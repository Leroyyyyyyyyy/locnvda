"""Real localhost sockets: httpx ASGITransport buffers SSE and misses disconnects.

Run: .venv-gateway/bin/python -m unittest discover -s tests -v
No GPU, external network, model downloads or credentials required.
"""
import asyncio
from contextlib import asynccontextmanager
import gzip
import os
import socket
import unittest
from unittest.mock import patch

from fastapi import FastAPI, Request
import httpx
from starlette.responses import Response, StreamingResponse
import uvicorn

from gateway.app import create_app, end_to_end_headers
from gateway.config import Settings
from gateway.mock_upstream import app as demo_upstream

TEST_KEYS = {"alice": "test-key", "bob": "second-test-key"}
UPSTREAM_KEY = "upstream-only-key"
AUTH_HEADERS = {"Authorization": "Bearer test-key"}


def gateway_settings(**overrides):
    # Regression tests exercise proxy/auth, not production quota defaults.
    return Settings(**{"api_keys": TEST_KEYS, "upstream_api_key": UPSTREAM_KEY,
                       "rate_limit_rps": 1000, "rate_limit_burst": 1000,
                       "max_in_flight": 1, **overrides})


@asynccontextmanager
async def running_server(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        app, log_level="error", access_log=False, lifespan="on",
        timeout_graceful_shutdown=1,
    ))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Test server did not start")
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 5)
        finally:
            sock.close()


class ConfigTests(unittest.TestCase):
    def test_env(self):
        with patch.dict(os.environ, {
            "GATEWAY_UPSTREAM_URL": "http://localhost:9000/prefix/",
            "GATEWAY_READ_TIMEOUT": "45",
            "GATEWAY_MAX_CONNECTIONS": "30",
            "GATEWAY_API_KEYS": '{"alice":"test-key","bob":"second-test-key"}',
            "GATEWAY_UPSTREAM_API_KEY": UPSTREAM_KEY,
            "GATEWAY_RATE_LIMIT_RPS": "2.5",
            "GATEWAY_RATE_LIMIT_BURST": "6",
            "GATEWAY_MAX_IN_FLIGHT": "3",
        }, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.upstream_url, "http://localhost:9000/prefix/")
        self.assertEqual(settings.read_timeout, 45)
        self.assertEqual(settings.max_connections, 30)
        self.assertEqual(dict(settings.api_keys), TEST_KEYS)
        self.assertEqual(settings.upstream_api_key, UPSTREAM_KEY)
        self.assertEqual(settings.rate_limit_rps, 2.5)
        self.assertEqual(settings.rate_limit_burst, 6)
        self.assertEqual(settings.max_in_flight, 3)

    def test_invalid_settings(self):
        cases = [
            {"upstream_url": "file:///tmp/model"},
            {"upstream_url": "http://user:secret@localhost:8000"},
            {"upstream_url": "http://localhost:8000/?x=1"},
            {"upstream_url": "http://localhost:8000/#fragment"},
            {"upstream_url": "http://localhost:99999"},
            {"read_timeout": 0}, {"connect_timeout": -1},
            {"read_timeout": float("nan")}, {"read_timeout": float("inf")},
            {"max_connections": 0}, {"max_keepalive_connections": -1},
            {"max_connections": 10, "max_keepalive_connections": 20},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                Settings(api_keys=TEST_KEYS, **case)

    def test_hop_headers_and_duplicate_headers(self):
        raw = [
            (b"Connection", b"keep-alive, X-Internal"), (b"X-Internal", b"secret"),
            (b"Transfer-Encoding", b"chunked"), (b"Host", b"downstream"),
            (b"Content-Length", b"42"), (b"Content-Encoding", b"gzip"),
            (b"Set-Cookie", b"a=1"), (b"Set-Cookie", b"b=2"),
        ]
        self.assertEqual(end_to_end_headers(raw), raw[3:])
        self.assertEqual(end_to_end_headers(raw, request=True), raw[5:])


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.records = []
        self.stream_started = asyncio.Event()
        self.stream_release = asyncio.Event()
        self.stream_closed = asyncio.Event()
        self.headers_started = asyncio.Event()
        self.headers_disconnected = asyncio.Event()
        mock = FastAPI()

        async def handler(request: Request):
            body = await request.body()
            self.records.append({
                "path": request.url.path, "query": request.url.query,
                "body": body, "headers": dict(request.headers),
            })
            mode = request.headers.get("x-test-mode", "json")
            if mode == "wait-headers":
                self.headers_started.set()
                # Observe the upstream TCP connection, not just gateway output.
                while (await request.receive())["type"] != "http.disconnect":
                    pass
                self.headers_disconnected.set()
                return Response()
            if mode == "sse":
                async def chunks():
                    try:
                        self.stream_started.set()
                        yield 'data: {"choices":[{"text":"你好"}]}\n\n'.encode()
                        # The test releases this only AFTER it sees chunk one.
                        await self.stream_release.wait()
                        yield b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                        yield b"data: [DONE]\n\n"
                    finally:
                        self.stream_closed.set()
                return StreamingResponse(chunks(), media_type="text/event-stream", headers={
                    "connection": "keep-alive, x-internal", "x-internal": "remove-me",
                    "x-upstream": "mock", "cache-control": "no-cache",
                })
            if mode == "gzip":
                return Response(gzip.compress(b'{"message":"compressed"}'),
                                media_type="application/json", headers={"content-encoding": "gzip"})
            if mode == "error":
                return Response(b'{"error":{"message":"upstream validation"}}',
                                status_code=422, media_type="application/json")
            if mode == "upstream-auth":
                return Response(b'{"error":{"code":"upstream_invalid_key"}}', status_code=401,
                                media_type="application/json", headers={"www-authenticate": "Bearer"})
            if mode == "limited":
                return Response(b'{"error":"busy"}', status_code=429,
                                media_type="application/json", headers={"retry-after": "3"})
            if mode == "redirect":
                return Response(status_code=307, headers={"location": "http://127.0.0.1:1/other"})
            if mode == "cookies":
                response = Response(b"{}", media_type="application/json")
                response.raw_headers.extend([(b"set-cookie", b"a=1; Path=/"),
                                             (b"set-cookie", b"b=2; Path=/")])
                return response
            if request.method == "GET":
                return Response(b'{"object":"list","data":[{"id":"mock-model"}]}',
                                media_type="application/json")
            return Response(body, media_type="application/json", headers={"x-upstream": "mock"})

        for path, method in (("/v1/models", "GET"), ("/v1/completions", "POST"),
                             ("/v1/chat/completions", "POST")):
            mock.add_api_route(path, handler, methods=[method])
        self.upstream_context = running_server(mock)
        self.upstream_url = await self.upstream_context.__aenter__()
        self.gateway_app = create_app(gateway_settings(
            upstream_url=self.upstream_url, read_timeout=2, pool_timeout=0.5,
            max_connections=1, max_keepalive_connections=1,
        ))
        self.request_states = []

        async def capture_identity(scope, receive, send):
            # Observe state without BaseHTTPMiddleware altering stream/receive behavior.
            async def observed_send(message):
                if scope["type"] == "http" and message["type"] == "http.response.start":
                    self.request_states.append(dict(scope.get("state", {})))
                await send(message)
            await self.gateway_app(scope, receive, observed_send)

        self.gateway_context = running_server(capture_identity)
        self.gateway_url = await self.gateway_context.__aenter__()
        self.client = httpx.AsyncClient(base_url=self.gateway_url, trust_env=False, timeout=3,
                                       headers=AUTH_HEADERS)

    async def asyncTearDown(self):
        self.stream_release.set()
        await self.client.aclose()
        await self.gateway_context.__aexit__(None, None, None)
        await self.upstream_context.__aexit__(None, None, None)
        self.assertTrue(self.gateway_app.state.upstream_client.is_closed)
        self.assertEqual(self.gateway_app.state.admission.in_flight, 0)

    async def wait_for_in_flight(self, app, count):
        async with asyncio.timeout(2):
            while app.state.admission.in_flight != count:
                await asyncio.sleep(0.01)

    @asynccontextmanager
    async def limited_gateway(self, **overrides):
        app = create_app(gateway_settings(upstream_url=self.upstream_url, **overrides))
        async with running_server(app) as url:
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=3, headers=AUTH_HEADERS) as client:
                yield app, client
        self.assertEqual(app.state.admission.in_flight, 0)

    async def test_rate_quota_shared_across_routes_but_independent_per_caller(self):
        async with self.limited_gateway(rate_limit_rps=0.5, rate_limit_burst=2) as (app, client):
            clock = [app.state.admission._clock()]
            app.state.admission._clock = lambda: clock[0]
            self.assertEqual((await client.get("/v1/models")).status_code, 200)
            self.assertEqual((await client.post("/v1/completions", content=b"{}")).status_code, 200)
            rejected = await client.post("/v1/chat/completions", content=b"{}")
            self.assertEqual(rejected.status_code, 429)
            self.assertEqual(rejected.json()["error"]["code"], "rate_limit_exceeded")
            self.assertEqual(rejected.headers["retry-after"], "2")
            self.assertEqual(len(self.records), 2)  # Rejection never reached upstream.
            self.assertEqual((await client.get("/v1/models", headers={
                "Authorization": "Bearer second-test-key",
            })).status_code, 200)
            clock[0] += 2
            self.assertEqual((await client.get("/v1/models")).status_code, 200)

    async def test_invalid_auth_and_health_do_not_spend_frequency_quota(self):
        async with self.limited_gateway(rate_limit_rps=0.001, rate_limit_burst=1) as (_, client):
            self.assertEqual((await client.get("/v1/models", headers={
                "Authorization": "Bearer wrong-key",
            })).status_code, 401)
            self.assertEqual((await client.get("/health")).status_code, 200)
            self.assertEqual(self.records, [])
            self.assertEqual((await client.get("/v1/models")).status_code, 200)
            self.assertEqual((await client.get("/v1/models")).status_code, 429)
            self.assertEqual((await client.get("/health")).status_code, 200)
            self.assertEqual(len(self.records), 1)

    async def test_rate_rejection_happens_before_reading_body(self):
        async with self.limited_gateway(rate_limit_rps=0.001, rate_limit_burst=1) as (_, client):
            self.assertEqual((await client.get("/v1/models")).status_code, 200)
            reader, writer = await asyncio.open_connection(client.base_url.host, client.base_url.port)
            try:
                writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                             b"Authorization: Bearer test-key\r\nContent-Length: 1000\r\n\r\n")
                await writer.drain()  # No body: reject rather than waiting for it.
                headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 1)
                self.assertIn(b"429 Too Many Requests", headers)
            finally:
                writer.close()
                await writer.wait_closed()
            self.assertEqual(len(self.records), 1)

    async def test_sse_holds_global_slot_until_complete_but_models_and_health_still_work(self):
        async with self.limited_gateway(max_in_flight=1, max_connections=2,
                                        max_keepalive_connections=2) as (app, client):
            async with client.stream("POST", "/v1/chat/completions", content=b"{}",
                                     headers={"x-test-mode": "sse"}) as stream:
                chunks = stream.aiter_bytes()
                await asyncio.wait_for(anext(chunks), 1)
                self.assertEqual(app.state.admission.in_flight, 1)
                busy = await client.post("/v1/completions", content=b"{}", headers={
                    "Authorization": "Bearer second-test-key",
                })
                self.assertEqual(busy.status_code, 503)
                self.assertEqual(busy.json()["error"]["code"], "gateway_busy")
                self.assertEqual(busy.headers["retry-after"], "1")
                self.assertEqual(len(self.records), 1)
                self.assertEqual((await client.get("/v1/models")).status_code, 200)
                self.assertEqual((await client.get("/health")).status_code, 200)
                self.assertEqual(app.state.admission.in_flight, 1)
                self.stream_release.set()
                rest = b"".join([chunk async for chunk in chunks])
                self.assertTrue(rest.endswith(b"data: [DONE]\n\n"))
            await self.wait_for_in_flight(app, 0)
            self.assertEqual((await client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_stream_disconnect_releases_slot_for_next_inference(self):
        async with self.limited_gateway(max_in_flight=1) as (app, client):
            async with client.stream("POST", "/v1/completions", content=b"{}",
                                     headers={"x-test-mode": "sse"}) as stream:
                await asyncio.wait_for(anext(stream.aiter_bytes()), 1)
                self.assertEqual(app.state.admission.in_flight, 1)
            await asyncio.wait_for(self.stream_closed.wait(), 2)
            await self.wait_for_in_flight(app, 0)
            self.assertFalse(self.stream_release.is_set())
            self.assertEqual((await client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_simultaneous_requests_cannot_exceed_global_cap(self):
        async with self.limited_gateway(max_in_flight=2) as (app, client):
            responses = await asyncio.gather(*(
                client.send(client.build_request("POST", "/v1/completions", content=b"{}",
                                                 headers={"x-test-mode": "sse"}), stream=True)
                for _ in range(6)
            ))
            try:
                statuses = [response.status_code for response in responses]
                self.assertEqual(statuses.count(200), 2)
                self.assertEqual(statuses.count(503), 4)
                self.assertEqual(app.state.admission.in_flight, 2)
                self.assertEqual(len(self.records), 2)
            finally:
                for response in responses:
                    await response.aclose()
            await self.wait_for_in_flight(app, 0)
            self.assertEqual((await client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_upload_disconnect_releases_slot_and_busy_rejection_skips_body_read(self):
        async with self.limited_gateway(max_in_flight=1) as (app, client):
            _, upload = await asyncio.open_connection(client.base_url.host, client.base_url.port)
            try:
                upload.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                             b"Authorization: Bearer test-key\r\nContent-Length: 1000\r\n\r\n")
                await upload.drain()
                await self.wait_for_in_flight(app, 1)
                reader, rejected = await asyncio.open_connection(client.base_url.host, client.base_url.port)
                try:
                    rejected.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                                   b"Authorization: Bearer second-test-key\r\nContent-Length: 1000\r\n\r\n")
                    await rejected.drain()
                    headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 1)
                    self.assertIn(b"503 Service Unavailable", headers)
                    self.assertEqual(self.records, [])
                finally:
                    rejected.close()
                    await rejected.wait_closed()
            finally:
                upload.close()
                await upload.wait_closed()
            await self.wait_for_in_flight(app, 0)
            self.assertEqual((await client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_unauthorized_requests_never_reach_upstream(self):
        async with httpx.AsyncClient(base_url=self.gateway_url, trust_env=False) as anonymous:
            cases = [[], [("Authorization", "Bearer wrong-key")],
                     [("Authorization", f"Bearer {UPSTREAM_KEY}")],
                     [("Authorization", "Bearer test-key"), ("Authorization", "Bearer second-test-key")],
                     [("Authorization", "Bearer test-key"), ("Authorization", "Bearer test-key")]]
            for path, method in (("/v1/models", "GET"), ("/v1/completions", "POST"),
                                 ("/v1/chat/completions", "POST")):
                for headers in cases:
                    with self.subTest(path=path, headers=headers):
                        response = await anonymous.request(method, path, headers=headers, content=b"{}")
                        self.assertEqual(response.status_code, 401)
                        self.assertEqual(response.headers["www-authenticate"], "Bearer")
                        self.assertEqual(response.json()["error"]["code"], "invalid_api_key")
                        for secret in (*TEST_KEYS.values(), UPSTREAM_KEY, "wrong-key"):
                            self.assertNotIn(secret, response.text)
                        self.assertNotIn("caller_id", self.request_states[-1])
            query_key = await anonymous.get("/v1/models?api_key=test-key")
            self.assertEqual(query_key.status_code, 401)
            self.assertEqual((await anonymous.get("/health")).status_code, 200)
        self.assertEqual(self.records, [])

    async def test_caller_identity_cannot_be_spoofed_and_keys_are_isolated(self):
        for caller, key in TEST_KEYS.items():
            response = await self.client.get("/v1/models", headers={
                "Authorization": f"Bearer {key}", "X-Gateway-Caller-ID": "admin",
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.request_states[-1]["caller_id"], caller)
            self.assertNotIn(key, repr(self.request_states[-1]))
            self.assertEqual(self.records[-1]["headers"]["authorization"], f"Bearer {UPSTREAM_KEY}")
            self.assertNotIn("x-gateway-caller-id", self.records[-1]["headers"])

    async def test_no_upstream_key_means_no_authorization_forwarded(self):
        app = create_app(gateway_settings(upstream_url=self.upstream_url, upstream_api_key=None))
        async with running_server(app) as url:
            async with httpx.AsyncClient(base_url=url, trust_env=False, headers=AUTH_HEADERS) as client:
                self.assertEqual((await client.get("/v1/models")).status_code, 200)
        self.assertNotIn("authorization", self.records[-1]["headers"])

    async def test_authentication_happens_before_reading_body(self):
        host, port = self.gateway_url.removeprefix("http://").split(":")
        reader, writer = await asyncio.open_connection(host, int(port))
        try:
            # Deliberately do not send the declared body: authentication must
            # reject now, not wait for the payload or acquire an upstream slot.
            writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\n\r\n")
            await writer.drain()
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 1)
            self.assertIn(b"401 Unauthorized", headers)
        finally:
            writer.close()
            await writer.wait_closed()
        self.assertEqual(self.records, [])

    async def test_upstream_401_is_not_mistaken_for_gateway_auth_failure(self):
        response = await self.client.post("/v1/completions", content=b"{}",
                                          headers={"x-test-mode": "upstream-auth"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "upstream_invalid_key")
        self.assertEqual(self.request_states[-1]["caller_id"], "alice")
        self.assertEqual(len(self.records), 1)

    async def test_json_and_fixed_routes(self):
        payload = b'{ "model": "mock-model", "prompt": [1,2,3], "stream": false, "extra": {"x":1} }'
        for path in ("/v1/completions", "/v1/chat/completions"):
            response = await self.client.post(path, content=payload, headers={
                "content-type": "application/json", "authorization": "Bearer test-key",
                "connection": "keep-alive, x-internal", "x-internal": "remove-me",
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, payload)
            self.assertEqual(response.headers["x-upstream"], "mock")
            record = self.records[-1]
            self.assertEqual(record["body"], payload)
            self.assertEqual(record["headers"]["authorization"], f"Bearer {UPSTREAM_KEY}")
            self.assertNotIn("x-internal", record["headers"])
            self.assertEqual(record["headers"]["host"], self.upstream_url.removeprefix("http://"))
        # JSON is intentionally opaque: vLLM owns payload validation.
        response = await self.client.post("/v1/completions", content=b"not-json")
        self.assertEqual(response.content, b"not-json")
        self.assertEqual((await self.client.get("/v1/completions")).status_code, 405)
        self.assertEqual((await self.client.post("/v1/arbitrary")).status_code, 404)

    async def test_models_query_and_health(self):
        response = await self.client.get("/v1/models?tag=a&tag=b&encoded=%2F")
        self.assertEqual(response.json()["data"][0]["id"], "mock-model")
        self.assertEqual(self.records[-1]["query"], "tag=a&tag=b&encoded=%2F")
        count = len(self.records)
        self.assertEqual((await self.client.get("/health")).json(), {"status": "ok"})
        self.assertEqual(len(self.records), count)

    async def test_errors_and_redirect_are_passed_through(self):
        for mode, status in (("error", 422), ("limited", 429), ("redirect", 307)):
            response = await self.client.post("/v1/completions", content=b"{}",
                                              headers={"x-test-mode": mode})
            self.assertEqual(response.status_code, status)
            if mode == "error":
                self.assertEqual(response.content, b'{"error":{"message":"upstream validation"}}')
            if mode == "limited":
                self.assertEqual(response.headers["retry-after"], "3")
            if mode == "redirect":
                self.assertEqual(response.headers["location"], "http://127.0.0.1:1/other")

    async def test_compressed_bytes_and_length(self):
        async with self.client.stream("POST", "/v1/completions", content=b"{}",
                                      headers={"x-test-mode": "gzip"}) as response:
            raw = b"".join([chunk async for chunk in response.aiter_raw()])
            self.assertEqual(response.headers["content-encoding"], "gzip")
            self.assertEqual(len(raw), int(response.headers["content-length"]))
            self.assertEqual(gzip.decompress(raw), b'{"message":"compressed"}')

    async def test_duplicate_headers_without_cross_caller_cookies(self):
        response = await self.client.post("/v1/completions", content=b"{}",
                                          headers={"x-test-mode": "cookies"})
        self.assertEqual(response.headers.get_list("set-cookie"), ["a=1; Path=/", "b=2; Path=/"])
        # A distinct caller has no cookies; the shared upstream pool must not add any.
        async with httpx.AsyncClient(base_url=self.gateway_url, trust_env=False, headers=AUTH_HEADERS) as other:
            await other.get("/v1/models")
        self.assertNotIn("cookie", self.records[-1]["headers"])

    async def test_sse_is_not_buffered_and_preserves_done(self):
        async with self.client.stream("POST", "/v1/chat/completions", content=b'{"stream":true}',
                                      headers={"x-test-mode": "sse"}) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers["content-type"])
            self.assertNotIn("x-internal", response.headers)
            self.assertEqual(response.headers["cache-control"], "no-cache")
            chunks = response.aiter_bytes()
            first = await asyncio.wait_for(anext(chunks), 1)
            self.assertEqual(first, 'data: {"choices":[{"text":"你好"}]}\n\n'.encode())
            self.assertFalse(self.stream_release.is_set())
            self.stream_release.set()
            rest = b"".join([chunk async for chunk in chunks])
            self.assertIn(b'"usage":{"completion_tokens":1}', rest)
            self.assertTrue(rest.endswith(b"data: [DONE]\n\n"))
        await asyncio.wait_for(self.stream_closed.wait(), 2)
        # max_connections=max_in_flight=1: release both pool and inference slots.
        await self.wait_for_in_flight(self.gateway_app, 0)
        self.assertEqual((await self.client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_disconnect_closes_upstream_stream_and_releases_pool(self):
        async with self.client.stream("POST", "/v1/completions", content=b"{}",
                                      headers={"x-test-mode": "sse"}) as response:
            await asyncio.wait_for(anext(response.aiter_bytes()), 1)
        await asyncio.wait_for(self.stream_closed.wait(), 2)
        self.assertFalse(self.stream_release.is_set())
        await self.wait_for_in_flight(self.gateway_app, 0)
        self.assertEqual((await self.client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_disconnect_while_waiting_for_headers(self):
        host, port = self.gateway_url.removeprefix("http://").split(":")
        _, writer = await asyncio.open_connection(host, int(port))
        try:
            writer.write(
                b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                b"Authorization: Bearer test-key\r\n"
                b"X-Test-Mode: wait-headers\r\nContent-Length: 2\r\n\r\n{}"
            )
            await writer.drain()
            await asyncio.wait_for(self.headers_started.wait(), 2)
        finally:
            writer.close()
            await writer.wait_closed()
        await asyncio.wait_for(self.headers_disconnected.wait(), 2)
        await self.wait_for_in_flight(self.gateway_app, 0)
        self.assertEqual((await self.client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_header_timeout_is_504_and_releases_connection(self):
        app = create_app(gateway_settings(upstream_url=self.upstream_url, read_timeout=0.1))
        async with running_server(app) as url:
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=3, headers=AUTH_HEADERS) as client:
                response = await client.post("/v1/completions", content=b"{}",
                                             headers={"x-test-mode": "wait-headers"})
                self.assertEqual(response.status_code, 504)
                self.assertEqual(response.json()["error"]["code"], "upstream_timeout")
                await asyncio.wait_for(self.headers_disconnected.wait(), 2)
                await self.wait_for_in_flight(app, 0)
                self.assertEqual((await client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_stream_timeout_aborts_instead_of_faking_done(self):
        app = create_app(gateway_settings(upstream_url=self.upstream_url, read_timeout=0.1))
        async with running_server(app) as url:
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=3, headers=AUTH_HEADERS) as client:
                # Uvicorn correctly reports an ASGI error when headers were already sent.
                with self.assertLogs("uvicorn.error", level="ERROR"), self.assertLogs("gateway.app", level="WARNING"):
                    async with client.stream("POST", "/v1/completions", content=b"{}",
                                             headers={"x-test-mode": "sse"}) as response:
                        chunks = response.aiter_bytes()
                        first = await anext(chunks)
                        self.assertEqual(response.status_code, 200)
                        self.assertNotIn(b"[DONE]", first)
                        with self.assertRaises(httpx.RemoteProtocolError):
                            await anext(chunks)
                await asyncio.wait_for(self.stream_closed.wait(), 2)
                await self.wait_for_in_flight(app, 0)
                self.assertEqual((await client.post("/v1/completions", content=b"{}")).status_code, 200)

    async def test_unreachable_upstream_is_502(self):
        # Reserve a port until the gateway is started, then close it. On macOS
        # a bound-but-not-listening socket may time out instead of refusing TCP.
        with socket.socket() as unused:
            unused.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{unused.getsockname()[1]}"
            app = create_app(gateway_settings(upstream_url=url))
            async with running_server(app) as gateway:
                unused.close()
                async with httpx.AsyncClient(base_url=gateway, trust_env=False, headers=AUTH_HEADERS) as client:
                    response = await client.post("/v1/completions", content=b"{}")
                    self.assertEqual(response.status_code, 502)
                    self.assertEqual(response.json()["error"]["code"], "upstream_unavailable")
                    self.assertNotIn(url, response.text)
                    await self.wait_for_in_flight(app, 0)
                    self.assertEqual((await client.get("/health")).status_code, 200)


class DemoTests(unittest.IsolatedAsyncioTestCase):
    async def test_documented_mock_through_gateway(self):
        async with running_server(demo_upstream) as upstream:
            async with running_server(create_app(gateway_settings(upstream_url=upstream))) as gateway:
                async with httpx.AsyncClient(base_url=gateway, trust_env=False, timeout=3, headers=AUTH_HEADERS) as client:
                    models = (await client.get("/v1/models")).json()
                    self.assertEqual(models["data"][0]["id"], "mock-model")
                    for path, field in (("/v1/completions", "text"), ("/v1/chat/completions", "message")):
                        normal = await client.post(path, json={"model": "mock-model", "stream": False})
                        self.assertEqual(normal.status_code, 200)
                        choice = normal.json()["choices"][0]
                        text = choice[field]["content"] if field == "message" else choice[field]
                        self.assertEqual(text, "你好，这是本地模拟响应。")
                        async with client.stream("POST", path, json={
                            "model": "mock-model", "stream": True,
                            "stream_options": {"include_usage": True},
                        }) as response:
                            lines = response.aiter_lines()
                            first = await asyncio.wait_for(anext(lines), 1)
                            self.assertIn("你好", first)
                            rest = [line async for line in lines]
                            self.assertIn("data: [DONE]", rest)
                            self.assertTrue(any('"usage"' in line for line in rest))


if __name__ == "__main__":
    unittest.main()
