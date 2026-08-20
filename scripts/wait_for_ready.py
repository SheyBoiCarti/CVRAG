#!/usr/bin/env python3
"""Utility script to poll backend readiness endpoint until ready or timeout."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def wait_for_ready(url: str, timeout: float = 60.0, interval: float = 2.0) -> int:
    """Polls the readiness endpoint until state is 'ready' or timeout occurs."""
    start = time.time()
    print(f"Waiting for backend readiness at {url} (timeout: {timeout}s)...")

    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wait-for-ready/1.0"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("state") == "ready":
                        elapsed = round(time.time() - start, 2)
                        print(f"Backend is READY in {elapsed}s (version: {data.get('pipeline_version')})")
                        return 0
                    print(f"Backend returned 200 but state is '{data.get('state')}': {data.get('reason_codes')}")
        except urllib.error.HTTPError as exc:
            print(f"Readiness probe returned HTTP {exc.code} ({exc.reason})")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            print(f"Backend not reachable yet ({type(exc).__name__})")

        time.sleep(interval)

    print(f"Timed out after {timeout}s waiting for backend readiness at {url}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/v1/health/ready",
        help="Readiness endpoint URL to poll (default: http://localhost:8000/api/v1/health/ready)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Maximum duration in seconds to wait before failing (default: 60.0)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds between probe requests (default: 2.0)",
    )
    args = parser.parse_args()
    return wait_for_ready(url=args.url, timeout=args.timeout, interval=args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
