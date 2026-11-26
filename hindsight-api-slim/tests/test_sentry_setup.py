import builtins
import os
import sys
import types
import unittest
from unittest.mock import patch

from hindsight_api import __version__, sentry_setup


def _fake_sdk():
    module = types.ModuleType("sentry_sdk")
    module.calls = []
    module.init = lambda **kwargs: module.calls.append(kwargs)
    return module


class SentrySetupTests(unittest.TestCase):
    def setUp(self):
        sentry_setup._reset_for_tests()
        self.addCleanup(sentry_setup._reset_for_tests)

    def test_no_dsn_is_noop_and_does_not_import_sdk(self):
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "sentry_sdk":
                raise AssertionError("sentry_sdk imported without DSN")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"SENTRY_DSN": "  "}), patch("builtins.__import__", guard):
            self.assertFalse(sentry_setup.install())
        env = {k: v for k, v in os.environ.items() if k != "SENTRY_DSN"}
        with patch.dict(os.environ, env, clear=True), patch("builtins.__import__", guard):
            self.assertFalse(sentry_setup.install())

    def test_defaults_and_idempotent(self):
        sdk = _fake_sdk()
        env = {"SENTRY_DSN": "https://k@o1.ingest.sentry.io/1"}
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"sentry_sdk": sdk}):
            self.assertTrue(sentry_setup.install())
            self.assertTrue(sentry_setup.install())
        self.assertEqual(len(sdk.calls), 1)
        call = sdk.calls[0]
        self.assertEqual(call["environment"], "production")
        self.assertEqual(call["release"], f"hindsight-oracle@{__version__}")
        self.assertEqual(call["traces_sample_rate"], 0.1)
        self.assertIs(call["before_send"], sentry_setup._before_send)
        self.assertFalse(call["data_collection"]["user_info"])
        self.assertEqual(call["data_collection"]["http_bodies"], [])
        self.assertEqual(call["data_collection"]["cookies"], {"mode": "off"})

    def test_custom_env_and_rate_clamping(self):
        cases = {"-3": 0.0, "7": 1.0, "abc": 0.1, "nan": 0.1, "inf": 0.1, "0.5": 0.5, "": 0.1}
        for raw, expected in cases.items():
            sentry_setup._reset_for_tests()
            sdk = _fake_sdk()
            env = {
                "SENTRY_DSN": "https://k@o1.ingest.sentry.io/1",
                "SENTRY_ENVIRONMENT": "staging",
                "SENTRY_RELEASE": "r1",
                "SENTRY_TRACES_SAMPLE_RATE": raw,
            }
            with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"sentry_sdk": sdk}):
                self.assertTrue(sentry_setup.install())
            self.assertEqual(sdk.calls[0]["traces_sample_rate"], expected, raw)
            self.assertEqual(sdk.calls[0]["environment"], "staging")
            self.assertEqual(sdk.calls[0]["release"], "r1")

    def test_missing_sdk_returns_false(self):
        real_import = builtins.__import__

        def block(name, *args, **kwargs):
            if name == "sentry_sdk":
                raise ImportError("no sentry_sdk")
            return real_import(name, *args, **kwargs)

        env = {"SENTRY_DSN": "https://k@o1.ingest.sentry.io/1"}
        with patch.dict(os.environ, env, clear=True), patch("builtins.__import__", block):
            with self.assertLogs("hindsight_api.sentry_setup", level="WARNING"):
                self.assertFalse(sentry_setup.install())

    def test_init_failure_degrades_without_raising(self):
        sdk = _fake_sdk()

        def boom(**kwargs):
            raise ValueError("BadDsn")

        sdk.init = boom
        env = {"SENTRY_DSN": "not-a-dsn"}
        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"sentry_sdk": sdk}):
            with self.assertLogs("hindsight_api.sentry_setup", level="WARNING"):
                self.assertFalse(sentry_setup.install())
        self.assertFalse(sentry_setup._installed)

    def test_before_send_scrubs_credentials(self):
        event = {
            "request": {
                "headers": {
                    "Authorization": "Bearer x",
                    "X-API-Key": "k",
                    "Cookie": "a=b",
                    "Set-Cookie": "a=b",
                    "Content-Type": "application/json",
                    "User-Agent": "ua",
                },
                "cookies": {"a": "b"},
                "data": {"secret": 1},
            }
        }
        out = sentry_setup._before_send(event, {})
        self.assertEqual(out["request"]["headers"], {"Content-Type": "application/json", "User-Agent": "ua"})
        self.assertNotIn("cookies", out["request"])
        self.assertNotIn("data", out["request"])
        self.assertEqual(sentry_setup._before_send({"message": "m"}, {}), {"message": "m"})


if __name__ == "__main__":
    unittest.main()
