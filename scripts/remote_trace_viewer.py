#!/usr/bin/env python3
"""Keep an H20 Trace API tunnel and the local React inspector available."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol


class Process(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...


class Runtime(Protocol):
    def popen(self, command: list[str], **kwargs: Any) -> Process: ...

    def sleep(self, seconds: float) -> None: ...

    def stop_process(self, process: Process) -> None: ...

    def output(self, message: str) -> None: ...

    def api_health(self, url: str) -> bool: ...

    def frontend_health(self, url: str, api_base_url: str) -> bool: ...

    def frontend_responds(self, url: str) -> bool: ...


def probe_api_health(
    url: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    """Accept only the versioned Trace API's explicit read-only health contract."""

    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with opener(request, timeout=1.0) as response:
            if getattr(response, "status", 200) != 200:
                return False
            payload = json.loads(response.read(4096))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return payload == {"status": "ok", "read_only": True}


def probe_frontend_health(
    url: str,
    api_base_url: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    """Distinguish this inspector from an unrelated process occupying the port."""

    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    try:
        with opener(request, timeout=1.0) as response:
            if getattr(response, "status", 200) != 200:
                return False
            body = response.read(64 * 1024).decode("utf-8", errors="replace")
    except OSError:
        return False
    identity = f'<meta name="fhp-api-base" content="{api_base_url}">'
    return "Fisheye Handpose · Trace Studio" in body and identity in body


class SystemRuntime:
    def popen(self, command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        return subprocess.Popen(command, **kwargs)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def stop_process(self, process: Process) -> None:
        if process.poll() is not None:
            return
        concrete = process
        pid = getattr(concrete, "pid", None)
        if not isinstance(pid, int):
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        wait = getattr(concrete, "wait", None)
        if not callable(wait):
            return
        try:
            wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            wait(timeout=5)

    def output(self, message: str) -> None:
        print(f"[remote-trace] {message}", flush=True)

    def api_health(self, url: str) -> bool:
        return probe_api_health(url)

    def frontend_health(self, url: str, api_base_url: str) -> bool:
        return probe_frontend_health(url, api_base_url)

    def frontend_responds(self, url: str) -> bool:
        return _frontend_port_responds(url)


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def _ssh_host(value: str) -> str:
    host = value.strip()
    if not host or host.startswith("-") or any(character.isspace() for character in host):
        raise argparse.ArgumentTypeError("SSH host must be a non-empty host or SSH alias")
    return host


def build_parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Keep an SSH tunnel to the remote Trace API and run the local React viewer"
    )
    parser.add_argument("--ssh-host", default="h20", type=_ssh_host)
    parser.add_argument("--remote-api-port", default=18080, type=_port)
    parser.add_argument("--local-api-port", default=18081, type=_port)
    parser.add_argument("--frontend-port", default=15174, type=_port)
    parser.add_argument("--frontend-dir", default=repository / "frontend", type=Path)
    parser.add_argument("--health-timeout", default=30.0, type=_positive_float)
    parser.add_argument("--reconnect-max-seconds", default=30.0, type=_positive_float)
    return parser


def _ssh_command(args: argparse.Namespace) -> list[str]:
    return [
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
        f"127.0.0.1:{args.local_api_port}:127.0.0.1:{args.remote_api_port}",
        args.ssh_host,
    ]


def _vite_command(args: argparse.Namespace) -> list[str]:
    return [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.frontend_port),
        "--strictPort",
    ]


def _wait_for_health(
    probe: Callable[[str], bool],
    url: str,
    runtime: Runtime,
    *,
    timeout: float,
    process: Process,
) -> bool:
    interval = 0.25
    attempts = max(1, math.ceil(timeout / interval))
    for _ in range(attempts):
        if process.poll() is not None:
            return False
        if probe(url):
            return True
        runtime.sleep(interval)
    return False


def _start_frontend(
    args: argparse.Namespace,
    runtime: Runtime,
    frontend_url: str,
    api_base_url: str,
) -> Process | None:
    if runtime.frontend_health(frontend_url, api_base_url):
        runtime.output(f"Reusing Vite at {frontend_url}")
        return None
    if runtime.frontend_responds(frontend_url):
        raise RuntimeError(
            f"frontend port is occupied by a stale or differently configured server: {frontend_url}"
        )
    environment = os.environ.copy()
    environment["VITE_API_BASE_URL"] = api_base_url
    runtime.output(f"Starting React inspector at {frontend_url}")
    process = runtime.popen(
        _vite_command(args),
        cwd=args.frontend_dir,
        env=environment,
        start_new_session=True,
    )
    if not _wait_for_health(
        lambda url: runtime.frontend_health(url, api_base_url),
        frontend_url,
        runtime,
        timeout=args.health_timeout,
        process=process,
    ):
        code = process.poll()
        runtime.output(
            "Vite did not become ready"
            + (" before the health timeout" if code is None else f"; exited with code {code}")
        )
        if code is None:
            runtime.stop_process(process)
        raise RuntimeError("local React inspector failed to start")
    return process


def _frontend_port_responds(url: str) -> bool:
    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    try:
        with urllib.request.urlopen(request, timeout=1.0) as response:
            return getattr(response, "status", 200) == 200
    except OSError:
        return False


def run(args: argparse.Namespace, runtime: Runtime) -> int:
    frontend_dir = args.frontend_dir.expanduser().resolve()
    if not (frontend_dir / "package.json").is_file():
        runtime.output(f"Frontend directory has no package.json: {frontend_dir}")
        return 2
    args.frontend_dir = frontend_dir
    api_base_url = f"http://127.0.0.1:{args.local_api_port}"
    api_health_url = f"{api_base_url}/api/v1/health"
    frontend_url = f"http://127.0.0.1:{args.frontend_port}"
    tunnel: Process | None = None
    frontend: Process | None = None
    reconnect_delay = 1.0
    try:
        while True:
            if runtime.api_health(api_health_url):
                runtime.output(f"Reusing existing SSH tunnel at 127.0.0.1:{args.local_api_port}")
                reconnect_delay = 1.0
                if frontend is None or frontend.poll() is not None:
                    frontend = _start_frontend(
                        args,
                        runtime,
                        frontend_url,
                        api_base_url,
                    )
                runtime.output(f"Viewer ready: {frontend_url} (Ctrl-C to stop)")
                while runtime.api_health(api_health_url):
                    if frontend is not None and frontend.poll() is not None:
                        runtime.output(f"Vite exited with code {frontend.returncode}; restarting")
                        frontend = _start_frontend(
                            args,
                            runtime,
                            frontend_url,
                            api_base_url,
                        )
                    runtime.sleep(0.5)
                runtime.output("Existing SSH tunnel is no longer healthy")
                runtime.output(f"Reconnecting in {reconnect_delay:.1f}s")
                runtime.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, args.reconnect_max_seconds)
                continue

            runtime.output(
                f"Starting SSH tunnel {args.ssh_host}:{args.remote_api_port} "
                f"-> 127.0.0.1:{args.local_api_port}"
            )
            tunnel = runtime.popen(_ssh_command(args), start_new_session=True)
            runtime.output(f"Waiting for Trace API health at {api_health_url}")
            if not _wait_for_health(
                runtime.api_health,
                api_health_url,
                runtime,
                timeout=args.health_timeout,
                process=tunnel,
            ):
                code = tunnel.poll()
                if code is None:
                    runtime.output("Trace API health timeout; restarting the SSH tunnel")
                    runtime.stop_process(tunnel)
                else:
                    runtime.output(f"SSH tunnel exited with code {code}")
                tunnel = None
                runtime.output(f"Reconnecting in {reconnect_delay:.1f}s")
                runtime.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, args.reconnect_max_seconds)
                continue

            # A different process may have answered the shared local health URL
            # just as our SSH child failed to bind.  Only a live child proves that
            # this process owns the newly established tunnel.
            if tunnel.poll() is not None:
                runtime.output(f"SSH tunnel exited with code {tunnel.returncode}")
                tunnel = None
                runtime.output(f"Reconnecting in {reconnect_delay:.1f}s")
                runtime.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2.0, args.reconnect_max_seconds)
                continue

            runtime.output(f"Trace API ready at {api_base_url}")
            reconnect_delay = 1.0
            if frontend is None or frontend.poll() is not None:
                frontend = _start_frontend(
                    args,
                    runtime,
                    frontend_url,
                    api_base_url,
                )
            runtime.output(f"Viewer ready: {frontend_url} (Ctrl-C to stop)")

            while tunnel.poll() is None:
                if frontend is not None and frontend.poll() is not None:
                    runtime.output(f"Vite exited with code {frontend.returncode}; restarting")
                    frontend = _start_frontend(
                        args,
                        runtime,
                        frontend_url,
                        api_base_url,
                    )
                runtime.sleep(0.5)
            runtime.output(f"SSH tunnel exited with code {tunnel.returncode}")
            tunnel = None
            runtime.output(f"Reconnecting in {reconnect_delay:.1f}s")
            runtime.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2.0, args.reconnect_max_seconds)
    except KeyboardInterrupt:
        runtime.output("Interrupt received; cleaning up")
        return 0
    except (OSError, RuntimeError) as error:
        runtime.output(f"Cannot start remote viewer: {error}")
        return 2
    finally:
        if frontend is not None and frontend.poll() is None:
            runtime.stop_process(frontend)
        if tunnel is not None and tunnel.poll() is None:
            runtime.stop_process(tunnel)
        runtime.output("Stopped remote Trace viewer")


def main(argv: Sequence[str] | None = None, *, runtime: Runtime | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args, SystemRuntime() if runtime is None else runtime)


if __name__ == "__main__":
    raise SystemExit(main())
