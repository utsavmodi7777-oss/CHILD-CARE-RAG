"""
Extract chunk id + context_enriched_content from each JSON file.

Reads JSON files from: new_data/processed
Writes one JSON file per input into: new_data/qa_chunks
Example: input 'file.json' -> output 'file_qa.json'
"""

from pathlib import Path
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

INPUT_DIR = Path("../../new_data/processed")
OUTPUT_DIR = Path("../../new_data/qa_chunks")

if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
    logging.error("Input folder not found: %s", INPUT_DIR)
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files_processed = 0

for path in sorted(INPUT_DIR.glob("*.json")):
    files_processed += 1
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        logging.warning("Skipping %s (failed to read/parse): %s", path.name, e)
        continue

    chunks = data.get("chunks") or []
    if not isinstance(chunks, list):
        logging.warning("No 'chunks' list in %s; skipping", path.name)
        continue

    results = []
    for chunk in chunks:
        cid = chunk.get("id")
        text = chunk.get("context_enriched_content") or chunk.get("content")
        if cid and text:
            results.append({"id": cid, "context_enriched_content": text})

    if not results:
        logging.info("No valid chunks found in %s", path.name)
        continue

    out_file = OUTPUT_DIR / (path.stem + "_qa.json")

    try:
        with out_file.open("w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error("Failed to write output file %s: %s", out_file.name, e)
        continue

    logging.info("Wrote %d chunks to %s", len(results), out_file.name)

logging.info("Finished. Processed %d file(s).", files_processed)
