"""Docker soak test: drive the app with chat turns + note writes while sampling RAM.

Samples VmRSS/Pss/RssAnon of the app worker inside the container (docker exec)
plus `docker stats` memory usage, writes a CSV, and prints a summary (slope,
peak, decay residual). Designed for the production stack (or a scratch copy).

Usage (host):
    AUTH_TOKEN=... python3 apps/server/scripts/soak_ram.py \
        --container aikioku-server-1 --port 8869 \
        --turns 30 --interval 0.5 --note-every 5 \
        --idle-secs 5 --decay-secs 5 --out /tmp/soak.csv

Note writes POST {title, content, path} to /api/notes/ (title and path are
required fields). Chat turns POST {query, mode} to /api/chat. If AUTH_TOKEN is
unset and the API answers 401, the script prints instructions and exits 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def _sample(container: str) -> dict[str, Any]:
    stats = _sh(["docker", "stats", container, "--no-stream", "--format", "{{.MemUsage}}"])
    mem_usage = stats.strip()
    worker = json.loads(
        _sh(
            [
                "docker",
                "exec",
                container,
                "python3",
                "-c",
                (
                    "import json,os;"
                    "pids=[p for p in os.listdir('/proc') if p.isdigit()];"
                    "targets={};"
                    "for p in pids:"
                    "\n"
                    " try:"
                    "\n"
                    "  cmd=open(f'/proc/{p}/cmdline','rb').read().replace(b'\\x00',b' ').decode()"
                    "\n"
                    " except OSError:"
                    "\n"
                    "  continue"
                    "\n"
                    " if 'uvicorn' in cmd or 'spawn_main' in cmd:"
                    "\n"
                    "  try:"
                    "\n"
                    "   status=open(f'/proc/{p}/status').read()"
                    "\n"
                    "   smaps=open(f'/proc/{p}/smaps_rollup').read()"
                    "\n"
                    "   targets[p]={'cmd':cmd[:60],'status':status,'smaps':smaps}"
                    "\n"
                    "  except OSError:"
                    "\n"
                    "   pass"
                    "\n"
                    "print(json.dumps(targets))"
                ),
            ]
        )
    )
    rss = 0
    pss = 0
    anon = 0
    for info in worker.values():
        for line in info["status"].splitlines():
            if line.startswith("VmRSS:"):
                rss += int(line.split()[1])
        for line in info["smaps"].splitlines():
            if line.startswith("Rss:"):
                pss += int(line.split()[1])
            elif line.startswith("RssAnon:"):
                anon += int(line.split()[1])
    return {
        "t": round(time.monotonic(), 3),
        "rss_kb": rss,
        "rss_anon_kb": anon,
        "rss_smaps_kb": pss,
        "docker_mem": mem_usage,
    }


def _api(base: str, path: str, payload: dict[str, Any], token: str) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{base}{path}", data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        if exc.code == 401 and not token:
            print("API answered 401 — set AUTH_TOKEN to a valid bearer token and retry.")
            sys.exit(2)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="aikioku-server-1")
    parser.add_argument("--port", type=int, default=8869)
    parser.add_argument("--turns", type=int, default=90)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--note-every", type=int, default=5)
    parser.add_argument("--idle-secs", type=float, default=5.0)
    parser.add_argument("--decay-secs", type=float, default=5.0)
    parser.add_argument("--sample-secs", type=float, default=1.0)
    parser.add_argument("--out", default="/tmp/soak.csv")
    args = parser.parse_args(argv)

    import os

    token = os.environ.get("AUTH_TOKEN", "")
    base = f"http://127.0.0.1:{args.port}"

    samples: list[dict[str, Any]] = []
    note_counter = 0

    def collect(label: str, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            sample = _sample(args.container)
            sample["phase"] = label
            samples.append(sample)
            time.sleep(args.sample_secs)

    collect("idle", args.idle_secs)

    for turn in range(1, args.turns + 1):
        _api(base, "/api/chat", {"query": f"soak query {turn}", "mode": "simple"}, token)
        if turn % args.note_every == 0:
            note_counter += 1
            _api(
                base,
                "/api/notes/",
                {
                    "title": f"Soak note {note_counter}",
                    "content": f"Soak test content number {note_counter}.",
                    "path": f"soak-{note_counter}.md",
                },
                token,
            )
        sample = _sample(args.container)
        sample["phase"] = "ramp"
        samples.append(sample)
        time.sleep(args.interval)

    collect("decay", args.decay_secs)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)

    ramp = [s for s in samples if s["phase"] == "ramp"]
    if len(ramp) >= 2:
        first = ramp[0]
        last = ramp[-1]
        turns_delta = max(args.turns - 1, 1)
        slope_kb = (last["rss_kb"] - first["rss_kb"]) / turns_delta
    else:
        slope_kb = 0.0
    peak = max(s["rss_kb"] for s in samples)
    decay_end = [s for s in samples if s["phase"] == "decay"]
    decay_residual = decay_end[-1]["rss_kb"] - samples[0]["rss_kb"] if decay_end else 0

    print(f"soak.csv: {args.out}")
    print(f"rss start/end/peak (kB): {samples[0]['rss_kb']}/{samples[-1]['rss_kb']}/{peak}")
    print(f"slope (kB/turn): {slope_kb:.2f}")
    print(f"decay residual (kB): {decay_residual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
