"""Download the fixed WebShop catalog and build its matching Lucene index."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from urllib.error import URLError
from urllib.request import Request, build_opener

from .launch import ROOT, execute, runtime_env
from .storage import preflight

# Byte-identical WebShop assets, pinned to the public MIT-licensed mirror.
# Provenance: https://huggingface.co/datasets/sparklabutah/timewarp-env-data
SOURCE_BASE = (
    "https://huggingface.co/datasets/sparklabutah/timewarp-env-data/resolve/"
    "bfed490a8b4b5dae94b126be6105b84445dcbc4b/webshop/"
)
FILES = {
    "items_shuffle_1000.json": "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269",
    "items_ins_v2_1000.json": "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f",
    "items_human_ins.json": "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
}
MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024


def download_json(name, partial):
    """Stream a fixed public asset without a persistent download cache."""
    request = Request(SOURCE_BASE + name, headers={"User-Agent": "Mozilla/5.0"})
    opener = build_opener()
    try:
        with opener.open(request, timeout=30) as response:
            if response.headers.get_content_type() == "text/html":
                raise RuntimeError("The asset mirror returned an HTML error page; retry later.")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("Download exceeds the 16 MiB limit for a catalog asset.")
            received = 0
            with partial.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > MAX_DOWNLOAD_BYTES:
                        raise RuntimeError("Download exceeds the 16 MiB limit for a catalog asset.")
                    output.write(chunk)
    except URLError as error:
        raise RuntimeError(f"WebShop asset download failed ({error}); check network access to "
                           "Hugging Face and retry. Any partial file was retained.") from error
    except OSError as error:
        raise RuntimeError(f"Cannot save WebShop asset {partial.name}: {error}. "
                           "Any partial file was retained.") from error
    if digest(partial) != FILES[name]:
        raise RuntimeError(f"SHA-256 mismatch for {name}; the partial file was retained.")
    try:
        with partial.open(encoding="utf-8") as downloaded:
            json.load(downloaded)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("The asset is not valid JSON; the partial file was retained.") from error


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(directory, record_dir):
    for name, expected_sha256 in FILES.items():
        target = directory / name
        if not target.exists():
            partial = directory / f"{name}.{uuid.uuid4().hex[:8]}.partial"
            download_json(name, partial)
            if target.exists():
                raise RuntimeError(f"{name} appeared during download; refusing to overwrite it.")
            partial.rename(target)
        if digest(target) != expected_sha256:
            raise RuntimeError(f"{name} does not match the fixed catalog; choose a new setup directory.")
        json.loads(target.read_text())
    products_raw = json.loads((directory / "items_shuffle_1000.json").read_text())
    if not isinstance(products_raw, list) or len(products_raw) != 1000:
        raise ValueError("The fixed recipe requires the 1000-product catalog.")
    hashes = {name: digest(directory / name) for name in FILES}
    marker = directory / "index_manifest.json"
    index = directory / "indexes"
    if index.exists() and any(index.iterdir()):
        if marker.is_file() and json.loads(marker.read_text()).get("source_sha256") == hashes:
            from pyserini.search.lucene import LuceneSearcher
            searcher = LuceneSearcher(str(index))
            expected = json.loads(marker.read_text())["indexed_products"]
            if searcher.num_docs == expected:
                print(f"Existing index verified: {expected} products.")
                return
        raise RuntimeError("Existing index does not match the supplied catalog; choose a new setup directory.")
    os.environ["WEBSHOP_DATA"] = str(directory)
    vendor = ROOT / "agent_system/environments/env_package/webshop/webshop"
    sys.path.insert(0, str(vendor))
    from web_agent_site.engine.engine import load_products
    products, *_ = load_products(filepath=str(directory / "items_shuffle_1000.json"),
                                 attrpath=str(directory / "items_ins_v2_1000.json"),
                                 num_products=None, human_goals=False)
    resources = record_dir / "resources"
    resources.mkdir()
    with (resources / "documents.jsonl").open("x") as output:
        for product in products:
            options = ", and ".join(f"{name}: {', '.join(values)}"
                                     for name, values in product.get("options", {}).items())
            bullets = product.get("BulletPoints", [])
            contents = " ".join([product["Title"], product["Description"],
                                 bullets[0] if bullets else "", options]).lower()
            output.write(json.dumps({"id": product["asin"], "contents": contents,
                                     "product": product}) + "\n")
    subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                    "--input", str(resources), "--index", str(index),
                    "--generator", "DefaultLuceneDocumentGenerator", "--threads", "1",
                    "--storePositions", "--storeDocvectors", "--storeRaw"], check=True)
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(str(index))
    if searcher.num_docs != len(products):
        raise RuntimeError(f"Index contains {searcher.num_docs} products; expected {len(products)}.")
    marker.write_text(json.dumps({"source_sha256": hashes, "catalog_products": 1000,
                                  "indexed_products": len(products)}, indent=2) + "\n")
    print(f"Indexed {len(products)} products.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=ROOT / "data/webshop")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--record-dir", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    directory = args.directory.expanduser().resolve()
    if args.worker:
        build(directory, args.record_dir)
        return
    missing = [name for name in ("pyserini", "spacy", "en_core_web_sm")
               if importlib.util.find_spec(name) is None]
    if missing:
        p.error("Install requirements-webshop.txt and the spaCy model first; missing: " + ", ".join(missing))
    if not shutil.which("java"):
        p.error("Java 11 is required; install it before building the WebShop index.")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record_dir = directory / "setup" / suffix
    record_dir.mkdir(parents=True, exist_ok=False)
    budget = preflight(directory, record_dir, False, (ROOT / "r", ROOT / "t"), additional_gib=2)
    cmd = [sys.executable, "-m", "cord.setup_webshop", "--worker", "--directory", str(directory),
           "--record-dir", str(record_dir)]
    execute(cmd, runtime_env(directory), directory, record_dir, budget)
    print(f'export WEBSHOP_DATA="{directory}"')
    print(f'export WEBSHOP_INDEX="{directory / "indexes"}"')


if __name__ == "__main__":
    main()
