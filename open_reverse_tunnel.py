#!/usr/bin/env python3
import argparse
import re
import signal
import subprocess
import sys


URL_PATTERN = re.compile(r"(https://[a-zA-Z0-9.-]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open reverse SSH tunnel via localhost.run"
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=8000,
        help="Local HTTP port to expose (default: 8000)",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=80,
        help="Remote port on localhost.run (default: 80)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost.run",
        help="Tunnel host (default: localhost.run)",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="nokey",
        help="SSH user for tunnel host (default: nokey)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dest = f"{args.user}@{args.host}"
    remote_spec = f"{args.remote_port}:localhost:{args.local_port}"

    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ServerAliveInterval=30",
        "-R",
        remote_spec,
        dest,
    ]
    print("Running:", " ".join(cmd))
    print("Press Ctrl+C to stop tunnel.\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _stop(*_):
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    found_url = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            if found_url is None:
                m = URL_PATTERN.search(line)
                if m:
                    found_url = m.group(1)
                    print(f"\nPublic URL: {found_url}\n")
    finally:
        rc = proc.wait()
        print(f"Tunnel closed (exit code: {rc})")


if __name__ == "__main__":
    main()
