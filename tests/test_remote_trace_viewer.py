from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts import remote_trace_viewer


class FakeProcess:
    def __init__(self, poll_values: list[int | None] | None = None) -> None:
        self._poll_values = list(poll_values or [])
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self._poll_values:
            value = self._poll_values.pop(0)
            if value is not None:
                self.returncode = value
            return value
        return self.returncode


class FakeRuntime:
    def __init__(
        self,
        *,
        tunnels: list[FakeProcess],
        api_health: Callable[[str], bool] = lambda _url: True,
        frontend_health: Callable[[str, str], bool] = lambda _url, _api: True,
        frontend_responds: Callable[[str], bool] = lambda _url: False,
        interrupt_after_sleeps: int = 1,
    ) -> None:
        self.tunnels = list(tunnels)
        self.api_health = api_health
        self.frontend_health = frontend_health
        self.frontend_responds = frontend_responds
        self.interrupt_after_sleeps = interrupt_after_sleeps
        self.process_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.stopped: list[FakeProcess] = []
        self.messages: list[str] = []
        self.sleeps: list[float] = []
        self.frontend_process: FakeProcess | None = None
        self.current_tunnel: FakeProcess | None = None

    def popen(self, command: list[str], **kwargs: Any) -> FakeProcess:
        self.process_calls.append((command, kwargs))
        if command[0] == "ssh":
            self.current_tunnel = self.tunnels.pop(0)
            return self.current_tunnel
        assert command[:3] == ["npm", "run", "dev"]
        self.frontend_process = FakeProcess()
        return self.frontend_process

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if len(self.sleeps) >= self.interrupt_after_sleeps:
            raise KeyboardInterrupt

    def stop_process(self, process: FakeProcess) -> None:
        self.stopped.append(process)
        process.returncode = -15

    def output(self, message: str) -> None:
        self.messages.append(message)


def test_cli_starts_keepalive_tunnel_waits_for_health_then_starts_vite_and_cleans_up(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    frontend_checks = iter((False, True))
    tunnel = FakeProcess()
    runtime = FakeRuntime(
        tunnels=[tunnel],
        frontend_health=lambda _url, _api: next(frontend_checks),
        interrupt_after_sleeps=2,
    )

    code = remote_trace_viewer.main(["--frontend-dir", str(frontend)], runtime=runtime)

    assert code == 0
    ssh_command, ssh_options = runtime.process_calls[0]
    assert ssh_command == [
        "ssh",
        "-N",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "TCPKeepAlive=yes",
        "-L",
        "127.0.0.1:18081:127.0.0.1:18080",
        "h20",
    ]
    assert ssh_options["start_new_session"] is True
    vite_command, vite_options = runtime.process_calls[1]
    assert vite_command == [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        "15174",
        "--strictPort",
    ]
    assert vite_options["cwd"] == frontend.resolve()
    assert vite_options["env"]["VITE_API_BASE_URL"] == "http://127.0.0.1:18081"
    assert runtime.stopped == [runtime.frontend_process, tunnel]
    assert any("Trace API ready" in message for message in runtime.messages)
    assert any("http://127.0.0.1:15174" in message for message in runtime.messages)
    assert any("Stopped" in message for message in runtime.messages)


def test_cli_restarts_an_exited_tunnel_with_backoff_and_reuses_existing_vite(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    first = FakeProcess([255])
    second = FakeProcess()
    runtime = FakeRuntime(
        tunnels=[first, second],
        frontend_health=lambda _url, _api: True,
        interrupt_after_sleeps=2,
    )

    code = remote_trace_viewer.main(["--frontend-dir", str(frontend)], runtime=runtime)

    assert code == 0
    assert [command[0] for command, _ in runtime.process_calls] == ["ssh", "ssh"]
    assert runtime.sleeps[0] == 1.0
    assert first not in runtime.stopped
    assert runtime.stopped == [second]
    assert any("exited with code 255" in message for message in runtime.messages)
    assert any("Reusing Vite" in message for message in runtime.messages)


class FakeHttpResponse:
    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.status = status
        self._data = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self._data


def test_api_health_probe_requires_the_versioned_read_only_health_contract() -> None:
    received: list[tuple[str, float, str]] = []

    def healthy_open(request: Any, *, timeout: float) -> FakeHttpResponse:
        received.append((request.full_url, timeout, request.headers["Accept"]))
        return FakeHttpResponse({"status": "ok", "read_only": True})

    assert remote_trace_viewer.probe_api_health(
        "http://127.0.0.1:18081/api/v1/health",
        opener=healthy_open,
    )
    assert received == [("http://127.0.0.1:18081/api/v1/health", 1.0, "application/json")]
    assert not remote_trace_viewer.probe_api_health(
        "http://127.0.0.1:18081/api/v1/health",
        opener=lambda *_args, **_kwargs: FakeHttpResponse({"status": "ok", "read_only": False}),
    )


def test_frontend_health_requires_the_expected_api_base_identity() -> None:
    matching = '<title>Fisheye Handpose · Trace Studio</title><meta name="fhp-api-base" content="http://127.0.0.1:18081">'
    stale = matching.replace("18081", "8000")

    def response(body: str) -> FakeHttpResponse:
        value = FakeHttpResponse({})
        value._data = body.encode("utf-8")
        return value

    assert remote_trace_viewer.probe_frontend_health(
        "http://127.0.0.1:15174",
        "http://127.0.0.1:18081",
        opener=lambda *_args, **_kwargs: response(matching),
    )
    assert not remote_trace_viewer.probe_frontend_health(
        "http://127.0.0.1:15174",
        "http://127.0.0.1:18081",
        opener=lambda *_args, **_kwargs: response(stale),
    )


def test_cli_rejects_unsafe_hosts_invalid_ports_and_a_missing_frontend(tmp_path: Path) -> None:
    for arguments in (
        ["--ssh-host=-oProxyCommand=bad"],
        ["--ssh-host", "two words"],
        ["--local-api-port", "0"],
        ["--frontend-port", "65536"],
        ["--health-timeout", "nan"],
    ):
        with pytest.raises(SystemExit):
            remote_trace_viewer.main(arguments, runtime=FakeRuntime(tunnels=[]))

    runtime = FakeRuntime(tunnels=[])
    assert (
        remote_trace_viewer.main(
            ["--frontend-dir", str(tmp_path / "missing")],
            runtime=runtime,
        )
        == 2
    )
    assert runtime.process_calls == []
    assert any("package.json" in message for message in runtime.messages)


def test_cli_reports_a_missing_ssh_executable_and_still_finishes_cleanup(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    runtime = FakeRuntime(tunnels=[])

    def missing_executable(_command: list[str], **_kwargs: Any) -> FakeProcess:
        raise FileNotFoundError("ssh not found")

    runtime.popen = missing_executable  # type: ignore[method-assign]

    assert remote_trace_viewer.main(["--frontend-dir", str(frontend)], runtime=runtime) == 2
    assert any("ssh not found" in message for message in runtime.messages)
    assert runtime.messages[-1] == "Stopped remote Trace viewer"


def test_cli_refuses_to_reuse_a_frontend_with_a_different_api_identity(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    tunnel = FakeProcess()
    runtime = FakeRuntime(
        tunnels=[tunnel],
        frontend_health=lambda _url, _api: False,
        frontend_responds=lambda _url: True,
        interrupt_after_sleeps=10,
    )

    assert remote_trace_viewer.main(["--frontend-dir", str(frontend)], runtime=runtime) == 2
    assert [command[0] for command, _ in runtime.process_calls] == ["ssh"]
    assert tunnel in runtime.stopped
    assert any("differently configured" in message for message in runtime.messages)
