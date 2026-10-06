#!/usr/bin/env python3
"""Fetch the explicitly selected Flex Saline ERP data; Python standard library only.

Commands: manifest, download, verify, inventory, headers. Existing verified files are reused. Incomplete
downloads use adjacent .part files and HTTP Range when supported. No preprocessing.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import urllib.error
import urllib.request

BASE = "https://files.us.osf.io/v1/resources/zj3f5/providers/osfstorage"
RAW_ID = "/5e558fa28d8765025281489b/"
SCRIPTS_ID = "/5e5598b78d876502698129f5/"
SELECTED = {"N170", "oddball_active", "oddball_passive"}
LIMIT = 5_000_000_000
ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "selected_manifest.json"
STATUS = ROOT / "download_status.json"
USER_AGENT = "FlexSalineResearchDownloader/1.0 (public research dataset)"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps({"event": event, "utc": now(), **fields}), flush=True)


def save_generated(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def request(url, headers=None):
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})


def folder_metadata(folder_id):
    url = BASE + folder_id + "?meta="
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request(url), timeout=45) as response:
                obj = json.load(response)
            if not isinstance(obj.get("data"), list):
                raise ValueError("Expected complete folder metadata list: " + url)
            if obj.get("links", {}).get("next"):
                raise ValueError("Unexpected pagination; refusing incomplete inventory: " + url)
            return obj["data"]
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise
            emit("metadata_retry", url=url, attempt=attempt, error=str(error))
            time.sleep(attempt * 2)


def safe_path(path):
    value = PurePosixPath(path)
    if value.is_absolute() or ".." in value.parts or "\\" in path:
        raise ValueError("Unsafe source path: " + path)
    return value


def create_manifest():
    raw_entries = folder_metadata(RAW_ID)
    selected_entries = [entry for entry in raw_entries
                        if entry["attributes"]["kind"] == "folder"
                        and entry["attributes"]["name"] in SELECTED]
    if {entry["attributes"]["name"] for entry in selected_entries} != SELECTED:
        raise ValueError("Selected source folders are missing or renamed")
    pending = [(entry["attributes"]["path"], "Saline Raw Data/" + entry["attributes"]["name"])
               for entry in selected_entries]
    pending.append((SCRIPTS_ID, "scripts"))
    seen = set()
    files = []
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        active = {}
        while pending or active:
            while pending and len(active) < 3:
                folder_id, prefix = pending.pop()
                if folder_id in seen:
                    raise ValueError("Repeated directory in inventory: " + folder_id)
                seen.add(folder_id)
                active[executor.submit(folder_metadata, folder_id)] = (folder_id, prefix)
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                folder_id, prefix = active.pop(future)
                for entry in future.result():
                    attributes = entry["attributes"]
                    source_path = str(safe_path(prefix + "/" + attributes["name"]))
                    if attributes["kind"] == "folder":
                        pending.append((attributes["path"], source_path))
                        continue
                    if attributes["kind"] != "file":
                        raise ValueError("Unknown source entry type")
                    size = attributes.get("sizeInt")
                    if size is None:
                        size = attributes.get("size")
                    if not isinstance(size, int) or size < 0:
                        raise ValueError("Missing integer file size: " + source_path)
                    links = entry.get("links", {})
                    hashes = attributes.get("extra", {}).get("hashes", {})
                    if not hashes.get("sha256") and not hashes.get("md5"):
                        raise ValueError("Missing checksum: " + source_path)
                    files.append({"source_path": source_path, "local_path": "raw/" + source_path,
                                  "size_bytes": size, "hashes": hashes,
                                  "osf_path": attributes["path"],
                                  "download_url": links.get("download", BASE + attributes["path"]),
                                  "metadata_url": BASE + attributes["path"] + "?meta="})
                    total += size
                emit("folder_listed", folder=prefix, files_count=len(files), total_bytes=total)
                if total > LIMIT:
                    raise ValueError("Selected scope exceeds 5 GB; no EEG download permitted")
    if len({item["local_path"] for item in files}) != len(files):
        raise ValueError("Duplicate local file paths")
    result = {"schema_version": 1, "dataset": "Emotiv EPOC Flex Saline Validation — selected ERP paradigms",
              "source_url": "https://osf.io/zj3f5/", "created_utc": now(),
              "scope": {"included": ["Saline Raw Data/N170", "Saline Raw Data/oddball_active (P300)",
                                      "Saline Raw Data/oddball_passive (MMN)", "scripts"],
                        "participants": "All participants and files present in selected source folders",
                        "devices": ["emotiv", "neuroscan"],
                        "excluded": ["Saline Raw Data/ssvep", "Saline Raw Data/resting", "Gel Validation"]},
              "inventory_complete": True, "total_bytes": total, "limit_bytes": LIMIT,
              "files_count": len(files), "directories_queried": len(seen) + 1,
              "metadata_errors": [], "files": sorted(files, key=lambda item: item["local_path"])}
    if MANIFEST.exists():
        previous = json.loads(MANIFEST.read_text())
        if previous.get("files") != result["files"]:
            raise RuntimeError("Existing manifest differs from current source; review before replacing it")
        emit("manifest_unchanged", total_bytes=total, files_count=len(files))
    else:
        save_generated(MANIFEST, result)
        emit("manifest_created", path=str(MANIFEST), total_bytes=total, files_count=len(files))


def load_manifest():
    manifest = json.loads(MANIFEST.read_text())
    files = manifest["files"]
    if not manifest.get("inventory_complete") or sum(item["size_bytes"] for item in files) != manifest["total_bytes"]:
        raise ValueError("Incomplete or inconsistent manifest")
    if manifest["total_bytes"] > LIMIT:
        raise ValueError("Manifest exceeds selected-dataset byte limit")
    for item in files:
        path = safe_path(item["local_path"])
        if path.parts[0] != "raw":
            raise ValueError("Manifest destination must be inside raw/")
        allowed = str(path).startswith("raw/scripts/") or any(
            str(path).startswith("raw/Saline Raw Data/" + folder + "/") for folder in SELECTED)
        if not allowed:
            raise ValueError("Manifest file outside approved selected scope")
    return manifest


def verify_file(path, item):
    if not path.is_file() or path.stat().st_size != item["size_bytes"]:
        return False
    algorithms = [name for name in ("sha256", "md5") if item["hashes"].get(name)]
    if not algorithms:
        raise ValueError("No expected hashes: " + item["local_path"])
    digests = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            for digest in digests.values():
                digest.update(chunk)
    return all(digest.hexdigest().lower() == item["hashes"][name].lower()
               for name, digest in digests.items())


def fetch_file(item):
    destination = ROOT / item["local_path"]
    if not destination.resolve().is_relative_to(ROOT):
        raise RuntimeError("Destination resolves outside the dataset directory")
    if destination.exists():
        if verify_file(destination, item):
            return "already_verified"
        raise RuntimeError("Existing final file does not match manifest; left untouched: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if not partial.resolve().is_relative_to(ROOT):
        raise RuntimeError("Partial file resolves outside the dataset directory")
    for attempt in range(1, 4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item["size_bytes"]:
                raise RuntimeError("Oversized partial file; left untouched: " + str(partial))
            if offset < item["size_bytes"]:
                headers = {"Range": "bytes=" + str(offset) + "-"} if offset else {}
                with urllib.request.urlopen(request(item["download_url"], headers), timeout=60) as response:
                    if offset and response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith("bytes " + str(offset) + "-"):
                            raise ValueError("Unexpected Content-Range")
                        mode = "ab"
                    elif response.status == 200:
                        mode = "wb"  # Server ignored Range: restart only this helper's partial.
                    else:
                        raise ValueError("Unexpected download HTTP status: " + str(response.status))
                    with partial.open(mode) as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            if handle.tell() > item["size_bytes"]:
                                raise ValueError("Response exceeds manifest file size")
            if not verify_file(partial, item):
                if partial.stat().st_size == item["size_bytes"]:
                    # Preserve the failed transfer as evidence; do not touch existing final files.
                    quarantined = partial.with_name(partial.name + ".checksum_failed." + str(time.time_ns()))
                    os.replace(partial, quarantined)
                raise ValueError("Downloaded size or checksum mismatch")
            if not destination.resolve().is_relative_to(ROOT) or destination.exists():
                raise RuntimeError("Destination changed during transfer; refusing to overwrite it")
            # Atomic, no-clobber publication: link fails if any destination appeared.
            os.link(partial, destination)
            partial.unlink()
            return "downloaded_verified"
        except (OSError, ValueError) as error:
            emit("download_retry" if attempt < 3 else "download_failed", path=item["local_path"],
                 attempt=attempt, error=str(error))
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def process_files(download, workers):
    manifest = load_manifest()
    started = now()
    completed = []
    failed = []
    status = {"manifest": MANIFEST.name, "started_utc": started, "updated_utc": started,
              "mode": "download" if download else "verify", "state": "running",
              "files_total": manifest["files_count"], "bytes_total": manifest["total_bytes"],
              "files_verified": 0, "bytes_verified": 0, "failures": []}
    save_generated(STATUS, status)
    iterator = iter(manifest["files"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        exhausted = False
        while active or not exhausted:
            while not exhausted and not failed and len(active) < workers:
                item = next(iterator, None)
                if item is None:
                    exhausted = True
                    break
                if download:
                    future = executor.submit(fetch_file, item)
                else:
                    future = executor.submit(verify_file, ROOT / item["local_path"], item)
                active[future] = item
            if not active:
                break
            done, _ = concurrent.futures.wait(active, timeout=30,
                                            return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                emit("transfer_active", files_verified=len(completed), bytes_verified=status["bytes_verified"],
                     active_paths=[item["local_path"] for item in active.values()])
            for future in done:
                item = active.pop(future)
                try:
                    result = future.result()
                    if result is False:
                        raise ValueError("Missing file or incorrect size/checksum")
                    completed.append(item["local_path"])
                    status["files_verified"] += 1
                    status["bytes_verified"] += item["size_bytes"]
                    emit("file_verified", path=item["local_path"], result=result,
                         files_verified=len(completed), bytes_verified=status["bytes_verified"],
                         files_total=manifest["files_count"], bytes_total=manifest["total_bytes"])
                except Exception as error:
                    failure = {"path": item["local_path"], "error": str(error)}
                    failed.append(failure)
                    emit("file_failed", **failure)
            status.update(updated_utc=now(), failures=failed)
            save_generated(STATUS, status)
            if failed:
                exhausted = True  # Finish already-active requests, but schedule no more files.
    status.update(updated_utc=now(), finished_utc=now(),
                  state="complete" if len(completed) == manifest["files_count"] and not failed else "incomplete",
                  failures=failed, verified_files=completed)
    save_generated(STATUS, status)
    emit("finished", state=status["state"], files_verified=len(completed),
         bytes_verified=status["bytes_verified"], failures=failed)
    if status["state"] != "complete":
        raise SystemExit(1)


def create_inventory():
    """Summarize public filenames only; do not open or preprocess EEG recordings."""
    manifest = load_manifest()
    groups = {}
    for item in manifest["files"]:
        source_path = PurePosixPath(item["source_path"])
        if source_path.parts[0] != "Saline Raw Data":
            continue
        key = "/".join(source_path.parts[1:-1])
        group = groups.setdefault(key, {"file_count": 0, "size_bytes": 0,
                                       "filetypes": {}, "participant_ids": set(),
                                       "unparsed_filenames": []})
        group["file_count"] += 1
        group["size_bytes"] += item["size_bytes"]
        extension = source_path.suffix.lower()
        filetype = group["filetypes"].setdefault(extension, {"file_count": 0, "participant_ids": set()})
        filetype["file_count"] += 1
        match = re.match(r"(\d{4})(?:_|\.)", source_path.name)
        if match:
            group["participant_ids"].add(match[1])
            filetype["participant_ids"].add(match[1])
        else:
            group["unparsed_filenames"].append(source_path.name)
    comparisons = {}
    paired_sets = []
    for paradigm in sorted(SELECTED):
        emotiv = groups[paradigm + "/eeg/emotiv"]["filetypes"][".edf"]["participant_ids"]
        neuroscan = groups[paradigm + "/eeg/neuroscan"]["filetypes"][".dat"]["participant_ids"]
        behavioral = groups[paradigm + "/behav"]["participant_ids"]
        paired = emotiv & neuroscan
        paired_sets.append(paired)
        comparisons[paradigm] = {
            "paired_eeg_count": len(paired), "paired_eeg_ids": sorted(paired),
            "emotiv_only_ids": sorted(emotiv - neuroscan),
            "neuroscan_only_ids": sorted(neuroscan - emotiv),
            "behavior_without_paired_eeg_ids": sorted(behavioral - paired),
            "paired_eeg_without_behavior_ids": sorted(paired - behavioral)}
    for group in groups.values():
        group["participant_ids"] = sorted(group["participant_ids"])
        for filetype in group["filetypes"].values():
            filetype["participant_ids"] = sorted(filetype["participant_ids"])
    all_paradigms = set.intersection(*paired_sets)
    result = {"source_manifest": MANIFEST.name, "generated_utc": now(),
              "method": "Participant IDs inferred only from public original filenames; no EEG preprocessing or quality assessment.",
              "groups": groups, "paired_device_comparisons": comparisons,
              "paired_in_all_selected_paradigms_count": len(all_paradigms),
              "paired_in_all_selected_paradigms_ids": sorted(all_paradigms)}
    output = ROOT / "participant_inventory.json"
    save_generated(output, result)
    emit("inventory_created", path=str(output), paired_in_all_paradigms=len(all_paradigms))


def create_header_inventory():
    """Read only EDF fixed/signal headers, excluding identity fields and samples."""
    manifest = load_manifest()
    records = []
    exceptions = []
    for item in manifest["files"]:
        if not item["local_path"].lower().endswith(".edf"):
            continue
        path = ROOT / item["local_path"]
        if not path.resolve().is_relative_to(ROOT):
            raise RuntimeError("Header source resolves outside dataset directory")
        try:
            with path.open("rb") as handle:
                fixed = handle.read(256)
                signal_count = int(fixed[252:256].decode("ascii").strip())
                if not 1 <= signal_count <= 1024:
                    raise ValueError("Invalid EDF signal count")
                header_bytes = int(fixed[184:192].decode("ascii").strip())
                record_duration = float(fixed[244:252].decode("ascii").strip())
                signals = handle.read(signal_count * 256)
            if len(signals) != signal_count * 256 or record_duration <= 0:
                raise ValueError("Invalid or truncated EDF header")
            labels = [signals[i * 16:(i + 1) * 16].decode("ascii").strip()
                      for i in range(signal_count)]
            sample_offset = signal_count * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
            samples = [int(signals[sample_offset + i * 8:sample_offset + (i + 1) * 8].decode("ascii").strip())
                       for i in range(signal_count)]
            records.append({"local_path": item["local_path"], "header_bytes": header_bytes,
                            "bytes_read": 256 + len(signals), "signal_count": signal_count,
                            "record_duration_seconds": record_duration, "labels": labels,
                            "sample_rates_hz": [count / record_duration for count in samples]})
        except (OSError, ValueError) as error:
            exceptions.append({"local_path": item["local_path"], "error": str(error)})
    if not records:
        raise RuntimeError("No readable EDF headers")
    representative = records[0]
    channel_exceptions = [record["local_path"] for record in records
                          if record["labels"] != representative["labels"]]
    rate_exceptions = [record["local_path"] for record in records
                       if record["sample_rates_hz"] != representative["sample_rates_hz"]]
    result = {"source_manifest": MANIFEST.name, "generated_utc": now(),
              "method": "EDF fixed and per-signal headers only; no samples, identity fields, Curry parsing, or quality preprocessing.",
              "records_expected": sum(item["local_path"].lower().endswith(".edf") for item in manifest["files"]),
              "records_read": len(records), "total_header_bytes_read": sum(record["bytes_read"] for record in records),
              "representative_file": representative["local_path"],
              "representative_labels": representative["labels"],
              "representative_sample_rates_hz": sorted(set(representative["sample_rates_hz"])),
              "channel_label_exceptions": channel_exceptions, "sampling_rate_exceptions": rate_exceptions,
              "read_errors": exceptions, "records": records}
    output = ROOT / "header_inventory.json"
    save_generated(output, result)
    emit("header_inventory_created", path=str(output), records_read=len(records),
         signal_counts=sorted({record["signal_count"] for record in records}),
         channel_label_exceptions=channel_exceptions, sampling_rate_exceptions=rate_exceptions,
         read_errors=exceptions)
    if exceptions:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["manifest", "download", "verify", "inventory", "headers"])
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=3)
    arguments = parser.parse_args()
    if arguments.command == "manifest":
        create_manifest()
    elif arguments.command == "inventory":
        create_inventory()
    elif arguments.command == "headers":
        create_header_inventory()
    else:
        process_files(arguments.command == "download", arguments.workers)
