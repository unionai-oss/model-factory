"""Regression tests for the serving-app client's reload handshake.

The failure these guard against: ``/reload`` is fire-and-forget, so a POST
that dies at the gateway (the app was scaled to zero) starts no load at all.
Treating that timeout as "loading started" made the client poll a healthy but
idle app until its deadline and then fail with a misleading message — a
30-minute task timeout instead of a working reload.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from model_factory.shared import inference_client as ic

CKPT = "s3://bucket/policy-checkpoint"


class _FakeService:
    """Minimal stand-in for the serving app's /health and /reload."""

    def __init__(self, drop_posts: int = 0, load_s: float = 0.2, fail_load: bool = False):
        self.state = {
            "loaded": False,
            "base_model": None,
            "checkpoint_path": None,
            "loading": None,
            "reload_error": None,
        }
        self.posts = 0
        self._drop = drop_posts
        self._load_s = load_s
        self._fail = fail_load
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self._send(dict(outer.state))

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.posts += 1
                if outer._drop > 0:
                    # Gateway timeout: the app never saw this request.
                    outer._drop -= 1
                    self._send({"detail": "activator request timeout"}, code=504)
                    return
                outer.state["loading"] = CKPT

                def load():
                    time.sleep(outer._load_s)
                    if outer._fail:
                        outer.state.update(reload_error="OSError: no such checkpoint", loading=None)
                    else:
                        outer.state.update(
                            loaded=True, checkpoint_path=CKPT, base_model="Qwen/x", loading=None
                        )

                threading.Thread(target=load, daemon=True).start()
                self._send({"ok": True, "loading": True, "checkpoint_path": CKPT})

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._srv.server_address[1]}"

    def close(self):
        self._srv.shutdown()


@pytest.fixture
def service(request):
    svc = _FakeService(**getattr(request, "param", {}))
    yield svc
    svc.close()


def test_reload_reissues_a_post_lost_at_the_gateway(service):
    """A dropped /reload must be detected via /health and sent again."""
    service._drop = 1
    out = ic.reload_checkpoint(service.url, checkpoint_path=CKPT, deadline_s=30, poll_s=0.5,
                               ready_s=10)
    assert out["checkpoint_path"] == CKPT
    assert service.posts >= 2, "client did not re-issue the lost POST"


def test_reload_returns_once_the_checkpoint_is_served(service):
    out = ic.reload_checkpoint(service.url, checkpoint_path=CKPT, deadline_s=30, poll_s=0.5,
                               ready_s=10)
    assert out["ok"] and out["base_model"] == "Qwen/x"
    assert service.posts == 1


def test_reload_is_a_no_op_when_already_serving(service):
    service.state.update(loaded=True, checkpoint_path=CKPT, base_model="Qwen/x")
    out = ic.reload_checkpoint(service.url, checkpoint_path=CKPT, deadline_s=30, poll_s=0.5,
                               ready_s=10)
    assert out["checkpoint_path"] == CKPT


def test_reload_surfaces_a_server_side_load_failure(service):
    service._fail = True
    with pytest.raises(ic.InferenceServiceError, match="no such checkpoint"):
        ic.reload_checkpoint(service.url, checkpoint_path=CKPT, deadline_s=30, poll_s=0.5,
                             ready_s=10)


def test_wait_until_ready_fails_fast_when_app_never_schedules():
    """An unreachable app must fail with a pointed message, not hang."""
    with pytest.raises(ic.InferenceServiceError, match="did not become reachable"):
        # Port 9 (discard) refuses/blackholes; nothing is listening.
        ic.wait_until_ready("http://127.0.0.1:9", deadline_s=2, poll_s=0.5)
