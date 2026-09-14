"""Download the two official v1.0 Feather files, verify publisher MD5, record SHA256."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import urllib.request

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": {
        "md5": "50a7718770c57220f160ba4f431ab89e",
        "bytes": 14483314,
        "generation": "1780494878811468",
        "crc32c": "vjz9cg==",
    },
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": {
        "md5": "f30e9dcca25cfd021bf1e7b3d975599e",
        "bytes": 1051241946,
        "generation": "1780494887545976",
        "crc32c": "dKRPVQ==",
    },
}


def verify(path, expected):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            md5.update(block)
            sha.update(block)
    if path.stat().st_size != expected["bytes"] or md5.hexdigest() != expected["md5"]:
        raise ValueError(f"Checksum/size mismatch: {path}. Remove and download again.")
    return sha.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("data/cache"))
    p.add_argument("--metadata", type=Path, default=Path("data/dataset.json"))
    a = p.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in FILES.items():
        dest = a.cache / name
        url = BASE + name
        if not dest.exists():
            print("Downloading", name, flush=True)
            req = urllib.request.Request(url + "?generation=" + expected["generation"])
            with urllib.request.urlopen(req, timeout=120) as response, dest.with_suffix(".part").open("wb") as f:
                for block in iter(lambda: response.read(8 * 1024**2), b""):
                    f.write(block)
            verify(dest.with_suffix(".part"), expected)
            os.replace(dest.with_suffix(".part"), dest)
        sha = verify(dest, expected)
        import pyarrow as pa

        with pa.memory_map(str(dest), "r") as f:
            reader = pa.ipc.open_file(f)
            schema = [{"name": field.name, "type": str(field.type)} for field in reader.schema]
            rows = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
        records.append(
            dict(
                file=name,
                url=url,
                generation_url=url + "?generation=" + expected["generation"],
                format="Apache Arrow IPC file / Feather v2",
                schema=schema,
                rows=rows,
                sha256=sha,
                publisher_md5_base64=base64.b64encode(bytes.fromhex(expected["md5"])).decode(),
                **expected,
            )
        )
        print("Verified", name, "SHA256", sha, flush=True)
    metadata = dict(
        dataset="MaleCNS",
        version="v1.0",
        neuprint_dataset="male-cns:v1.0",
        synapse_confidence_threshold=0.5,
        download_documentation="https://male-cns.janelia.org/download/",
        inspected_on="2026-09-13",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="FlyEM (HHMI Janelia), University of Cambridge, MRC Laboratory of Molecular Biology, Google Research",
        publication="https://doi.org/10.1016/j.cell.2026.08.015",
        checksum_source="Official Google Cloud Storage x-goog-hash and x-goog-generation headers; SHA256 computed locally.",
        files=records,
    )
    a.metadata.parent.mkdir(parents=True, exist_ok=True)
    a.metadata.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
