"""Mirror the whole BOAS dataset (OpenNeuro ds005555, CC0) from OpenNeuro's public S3 bucket.

Files already complete are skipped and partial ones resumed; stalled transfers are restarted.
Writes the files into <data>/raw (see common.py) and manifest_full.json next to this file.
Usage:  python3 download.py [n_workers]
"""
from __future__ import annotations
import json, re, subprocess, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from common import HERE, RAW

BUCKET = "https://s3.amazonaws.com/openneuro.org"
PREFIX = "ds005555/"


def list_files() -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": PREFIX}
        if token:
            q["continuation-token"] = token
        x = urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}").read().decode()
        for block in re.findall(r"<Contents>(.*?)</Contents>", x, re.S):
            key = re.search(r"<Key>(.*?)</Key>", block).group(1)
            size = int(re.search(r"<Size>(\d+)</Size>", block).group(1))
            if not key.endswith("/"):
                out.append((key[len(PREFIX):], size))
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", x)
        if not m:
            return out
        token = m.group(1)


def download(item: tuple[str, int]) -> None:
    rel, size = item
    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET}/{PREFIX}{urllib.parse.quote(rel)}"
    while not (dest.exists() and dest.stat().st_size == size):
        subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--speed-limit", "1000", "--speed-time", "60",
                        "-C", "-", "-o", str(dest), url])


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    files = list_files()
    print(f"{len(files)} files, {sum(s for _, s in files) / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(download, files))
    (HERE / "manifest_full.json").write_text(json.dumps(
        dict(dataset="OpenNeuro ds005555", files=[dict(path=r, bytes=s) for r, s in files]), indent=1))
    print("download complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
