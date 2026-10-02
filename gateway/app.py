"""Fixed-route async vLLM proxy with independent caller authentication (4a/4b)."""
from contextlib import asynccontextmanager
import logging

import anyio
from fastapi import FastAPI, Request
import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from gateway.auth import Authenticator
from gateway.config import Settings

logger = logging.getLogger(__name__)

# These describe one transport connection, not the end-to-end HTTP message.
HOP_BY_HOP = {
    b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
    b"te", b"trailer", b"transfer-encoding", b"upgrade", b"proxy-connection",
}


def end_to_end_headers(headers, *, request=False):
    blocked = set(HOP_BY_HOP)
    for name, value in headers:
        if name.lower() == b"connection":
            blocked.update(part.strip().lower() for part in value.split(b","))
    if request:
        # httpx generates the upstream Host and length from our buffered body.
        blocked.update({b"host", b"content-length", b"authorization", b"x-gateway-caller-id"})
    return [(name, value) for name, value in headers if name.lower() not in blocked]


def proxy_error(status, code, message):
    return JSONResponse(
        {"error": {"message": message, "type": "gateway_error", "code": code}},
        status_code=status,
    )


class ProxyResponse(Response):
    """Watch disconnects for the whole upstream lifetime, including header wait.

    A route-level `finally` would run before StreamingResponse starts sending.
    Cleanup therefore belongs here, after streaming finishes or is cancelled.
    """

    def __init__(self, client, upstream_request):
        super().__init__(content=b"")
        self.client = client
        self.upstream_request = upstream_request

    async def __call__(self, scope, receive, send):
        async def watch_disconnect(cancel_scope):
            while True:
                if (await receive())["type"] == "http.disconnect":
                    cancel_scope.cancel()
                    return

        async def forward():
            upstream = None
            try:
                try:
                    upstream = await self.client.send(self.upstream_request, stream=True)
                except httpx.TimeoutException:
                    await proxy_error(504, "upstream_timeout", "Upstream request timed out.")(scope, receive, send)
                    return
                except httpx.RequestError:
                    await proxy_error(502, "upstream_unavailable", "Unable to connect to upstream.")(scope, receive, send)
                    return

                response = StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code)
                # aiter_raw preserves compressed bytes; retain encoding and length.
                # Keep duplicate end-to-end headers (e.g. Set-Cookie) intact.
                response.raw_headers = end_to_end_headers(upstream.headers.raw)
                try:
                    # We own disconnect monitoring, including before headers arrive;
                    # do not start StreamingResponse's second receive consumer.
                    await response.stream_response(send)
                except httpx.RequestError as exc:
                    # Headers are already sent: fail the stream, never manufacture
                    # a successful EOF / [DONE] or try to send a second HTTP status.
                    logger.warning("Upstream stream interrupted (%s)", type(exc).__name__)
                    raise
            finally:
                if upstream is not None:
                    # Cancellation must not interrupt connection-pool cleanup.
                    with anyio.CancelScope(shield=True):
                        await upstream.aclose()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(watch_disconnect, tasks.cancel_scope)
            await forward()
            tasks.cancel_scope.cancel()


def create_app(settings=None):
    @asynccontextmanager
    async def lifespan(app):
        # Resolve environment once at startup, not at import or per request.
        # Missing/invalid keys fail startup rather than enabling anonymous access.
        active_settings = settings if settings is not None else Settings.from_env()
        app.state.settings = active_settings
        app.state.authenticator = Authenticator(active_settings.api_keys)
        timeout = httpx.Timeout(
            connect=active_settings.connect_timeout, read=active_settings.read_timeout,
            write=active_settings.write_timeout, pool=active_settings.pool_timeout,
        )
        limits = httpx.Limits(
            max_connections=active_settings.max_connections,
            max_keepalive_connections=active_settings.max_keepalive_connections,
        )
        # Ignore machine-wide HTTP_PROXY; do not follow redirects to other hosts.
        async with httpx.AsyncClient(timeout=timeout, limits=limits,
                                     trust_env=False, follow_redirects=False) as client:
            app.state.upstream_client = client
            yield

    app = FastAPI(title="LLM gateway — stages 4a/4b", lifespan=lifespan)

    @app.get("/health")
    async def health():
        # Liveness only, not an upstream readiness/GPU check.
        return {"status": "ok"}

    async def proxy(request: Request):
        caller_id = request.app.state.authenticator.authenticate(request.headers.getlist("authorization"))
        if caller_id is None:
            return JSONResponse(
                {"error": {"message": "Invalid or missing API key.",
                           "type": "authentication_error", "code": "invalid_api_key"}},
                status_code=401, headers={"WWW-Authenticate": "Bearer"},
            )
        # Trusted identity for later per-caller limits/logs. Do not use a caller-
        # supplied identity header, or retain the raw credential in request state.
        request.state.caller_id = caller_id
        body = await request.body()
        client = request.app.state.upstream_client
        active_settings = request.app.state.settings
        url = active_settings.upstream_url.rstrip("/") + request.url.path
        if request.url.query:
            url += "?" + request.url.query
        # Build Request directly: a shared client's cookie jar must never inject
        # one caller's upstream cookies into another caller's request.
        headers = end_to_end_headers(request.headers.raw, request=True)
        if active_settings.upstream_api_key is not None:
            headers.append((b"authorization", f"Bearer {active_settings.upstream_api_key}".encode("ascii")))
        upstream_request = httpx.Request(
            request.method, url, content=body, headers=headers,
            extensions={"timeout": client.timeout.as_dict()},
        )
        return ProxyResponse(client, upstream_request)

    app.add_api_route("/v1/models", proxy, methods=["GET"], response_class=Response)
    app.add_api_route("/v1/completions", proxy, methods=["POST"], response_class=Response)
    app.add_api_route("/v1/chat/completions", proxy, methods=["POST"], response_class=Response)
    return app


app = create_app()
