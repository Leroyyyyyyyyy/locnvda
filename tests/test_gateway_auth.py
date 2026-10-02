"""4b unit/config/startup tests. Integration cases live in test_gateway.py."""
import os
import unittest
from unittest.mock import patch

from gateway.app import create_app
from gateway.auth import Authenticator
from gateway.config import Settings, load_api_keys


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.auth = Authenticator({"alice": "alice-secret", "bob": "bob-secret"})

    def test_keys_identify_distinct_callers(self):
        for header, caller in (("Bearer alice-secret", "alice"),
                               ("bearer bob-secret", "bob"),
                               ("BEARER   alice-secret", "alice")):
            with self.subTest(header=header):
                self.assertEqual(self.auth.authenticate([header]), caller)
        self.assertNotIn("alice-secret", repr(self.auth._callers))
        self.assertNotIn("bob-secret", repr(self.auth._callers))

    def test_invalid_and_ambiguous_headers(self):
        cases = [
            [], [""], ["Bearer"], ["Bearer "], ["Basic alice-secret"],
            ["Bearer wrong"], ["Bearer ALICE-SECRET"], ["Bearer alice-secret extra"],
            ["Bearer alice-secret "], [" Bearer alice-secret"], ["Bearer\talice-secret"],
            ["Bearer 你好"], ["Bearer alice-secret\r\nInjected: yes"],
            ["Bearer " + "x" * 513], ["Bearer " + "x" * 2000],
            ["Bearer alice-secret", "Bearer bob-secret"],
            ["Bearer alice-secret", "Bearer alice-secret"],
            ["Bearer alice-secret", "Basic invalid"],
        ]
        for headers in cases:
            with self.subTest(headers=headers):
                self.assertIsNone(self.auth.authenticate(headers))


class AuthConfigTests(unittest.TestCase):
    def test_invalid_registries(self):
        cases = [
            None, {}, [], "secret", {"alice": ""}, {"alice": 42},
            {"alice": "contains space"}, {"alice": "你好"},
            {"alice": "secret\r\nHeader: injected"}, {"alice": "x" * 513},
            {"": "secret"}, {"bad caller": "secret"}, {"a" * 65: "secret"},
            {"alice": "same-secret", "bob": "same-secret"},
        ]
        for registry in cases:
            with self.subTest(registry=registry), self.assertRaises(ValueError):
                Settings(api_keys=registry)

    def test_invalid_json_and_duplicate_caller_ids(self):
        for raw in (None, "", "{", "null", "[]", "{}", '"secret"',
                    '{"alice":"first-secret","alice":"second-secret"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_api_keys(raw)

    def test_secrets_hidden_and_registry_immutable(self):
        keys = {"alice": "caller-secret"}
        settings = Settings(api_keys=keys, upstream_api_key="upstream-secret")
        keys["alice"] = "mutated-secret"
        self.assertEqual(settings.api_keys["alice"], "caller-secret")
        with self.assertRaises(TypeError):
            settings.api_keys["bob"] = "bob-secret"
        self.assertNotIn("caller-secret", repr(settings))
        self.assertNotIn("upstream-secret", repr(settings))

    def test_config_errors_do_not_echo_credentials(self):
        for env in (
            {"GATEWAY_API_KEYS": '{"alice":"sensitive secret"}'},
            {"GATEWAY_API_KEYS": '{"alice":"sensitive-secret", malformed'},
            {"GATEWAY_API_KEYS": '{"alice":"sensitive-secret","alice":"another-secret"}'},
            {"GATEWAY_API_KEYS": '{"alice":"valid-key"}',
             "GATEWAY_UPSTREAM_API_KEY": "sensitive secret"},
        ):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as result:
                    Settings.from_env()
                self.assertNotIn("sensitive", str(result.exception))
                self.assertNotIn("another-secret", str(result.exception))


class AuthStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_credentials_fail_startup_before_client_creation(self):
        app = create_app()
        with patch.dict(os.environ, {}, clear=True), patch("gateway.app.httpx.AsyncClient") as client:
            with self.assertRaisesRegex(ValueError, "GATEWAY_API_KEYS is required"):
                async with app.router.lifespan_context(app):
                    self.fail("Anonymous gateway must not start")
            client.assert_not_called()

    async def test_env_loaded_at_startup_not_import_and_client_closed(self):
        app = create_app()
        with patch.dict(os.environ, {
            "GATEWAY_API_KEYS": '{"startup-caller":"startup-key"}',
        }, clear=True):
            async with app.router.lifespan_context(app):
                self.assertEqual(app.state.authenticator.authenticate(["Bearer startup-key"]), "startup-caller")
                self.assertIsNone(app.state.settings.upstream_api_key)
                self.assertFalse(app.state.upstream_client.is_closed)
            self.assertTrue(app.state.upstream_client.is_closed)


if __name__ == "__main__":
    unittest.main()
