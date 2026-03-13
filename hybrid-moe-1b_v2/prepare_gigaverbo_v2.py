#!/usr/bin/env python3
"""
Prepare GigaVerbo-v2 for the V2 training pipeline.

Recommended path:
  1. Acquire/filter raw text offline into local JSONL shards.
  2. Tokenize shards once into tokens.bin + metadata.json.
  3. Train V2 only on local memmaps.

Streaming from Hugging Face is supported for acquisition, but never used in the
training hot path.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tiktoken
except ImportError:  # pragma: no cover - runtime dependency
    tiktoken = None


DEFAULT_TEXT_FIELDS = ("text", "content", "raw_content", "document", "body")
DEFAULT_GIGAVERBO_DATASET = "TucanoBR/GigaVerbo-v2"
DEFAULT_GIGAVERBO_SYNTH_DATASET = "TucanoBR/GigaVerbo-v2-Synth"
DEFAULT_SHARED_DATASET_ROOT = Path("/home/lgene/meu_modelo_temp/ai-model-forge/datasets")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _require_tiktoken():
    if tiktoken is None:
        raise RuntimeError("tiktoken is required. Install it in the project venv.")
    return tiktoken.get_encoding("gpt2")


def _safe_text(record: Dict, explicit_field: Optional[str]) -> Optional[str]:
    if explicit_field:
        value = record.get(explicit_field)
        if isinstance(value, str) and value.strip():
            return value
    for field in DEFAULT_TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _is_educational(record: Dict, threshold: int, score_field: str) -> bool:
    score = record.get(score_field)
    if score is None:
        return False
    try:
        return float(score) >= float(threshold)
    except (TypeError, ValueError):
        return False


def _iter_jsonl(paths: List[Path]) -> Iterator[Dict]:
    for path in paths:
        logger.info("Reading local shard: %s", path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _iter_hf_dataset(
    *,
    dataset_name: str,
    split: str,
    streaming: bool,
    cache_dir: Optional[str],
    config_name: Optional[str],
    retries: int,
    retry_sleep: float,
) -> Iterator[Dict]:
    from datasets import load_dataset

    kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": streaming,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if config_name:
        kwargs["name"] = config_name

    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            dataset = load_dataset(**kwargs)
            for record in dataset:
                yield dict(record)
            return
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            if attempt >= retries:
                break
            sleep_s = retry_sleep * (2 ** attempt)
            logger.warning(
                "HF dataset load failed (%s). retry=%d/%d sleep=%.1fs",
                exc,
                attempt + 1,
                retries,
                sleep_s,
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to read HF dataset {dataset_name}: {last_error}")


def acquire_raw_jsonl(
    *,
    output_dir: Path,
    subset: str,
    dataset_name: str,
    dataset_split: str,
    dataset_config: Optional[str],
    local_jsonl_glob: Optional[str],
    streaming: bool,
    cache_dir: Optional[str],
    text_field: Optional[str],
    edu_threshold: int,
    edu_score_field: str,
    shard_size: int,
    max_records: Optional[int],
    retries: int,
    retry_sleep: float,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    if local_jsonl_glob:
        paths = [Path(p) for p in sorted(glob.glob(local_jsonl_glob))]
        if not paths:
            raise FileNotFoundError(f"No local JSONL files matched: {local_jsonl_glob}")
        iterator = _iter_jsonl(paths)
        source = {"type": "local_jsonl", "glob": local_jsonl_glob}
    else:
        iterator = _iter_hf_dataset(
            dataset_name=dataset_name,
            split=dataset_split,
            streaming=streaming,
            cache_dir=cache_dir,
            config_name=dataset_config,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        source = {
            "type": "huggingface",
            "dataset": dataset_name,
            "split": dataset_split,
            "streaming": streaming,
            "config": dataset_config,
        }

    shard_idx = 0
    records_seen = 0
    records_kept = 0
    shard_records = 0
    shard_file = None
    shard_path: Optional[Path] = None
    shards: List[Dict] = []

    def open_next_shard() -> None:
        nonlocal shard_idx, shard_file, shard_path, shard_records
        if shard_file is not None:
            shard_file.close()
        shard_path = output_dir / f"raw_{subset}_{shard_idx:05d}.jsonl"
        shard_file = shard_path.open("w", encoding="utf-8")
        shard_records = 0
        shard_idx += 1

    open_next_shard()
    for record in iterator:
        records_seen += 1
        if max_records is not None and records_seen > max_records:
            break

        text = _safe_text(record, text_field)
        if not text:
            continue

        keep = True
        if subset == "edu":
            keep = _is_educational(record, edu_threshold, edu_score_field)
        elif subset == "non_edu":
            keep = not _is_educational(record, edu_threshold, edu_score_field)
        elif subset == "synth":
            keep = True

        if not keep:
            continue

        payload = {
            "text": text,
            "edu_int_score": record.get(edu_score_field),
            "source_dataset": dataset_name,
            "subset": subset,
        }
        shard_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        records_kept += 1
        shard_records += 1

        if shard_records >= shard_size:
            shards.append({"path": shard_path.name, "records": shard_records})
            open_next_shard()

    if shard_file is not None:
        shard_file.close()
        if shard_records > 0 and shard_path is not None:
            shards.append({"path": shard_path.name, "records": shard_records})
        elif shard_path is not None and shard_path.exists():
            shard_path.unlink()

    manifest = {
        "created_at": time.time(),
        "subset": subset,
        "source": source,
        "records_seen": records_seen,
        "records_kept": records_kept,
        "edu_threshold": edu_threshold,
        "edu_score_field": edu_score_field,
        "text_field": text_field,
        "shards": shards,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    logger.info(
        "Acquisition complete: subset=%s seen=%d kept=%d shards=%d",
        subset,
        records_seen,
        records_kept,
        len(shards),
    )
    return manifest


def tokenize_jsonl_shards(
    *,
    raw_dir: Path,
    output_dir: Path,
    subset_name: str,
    append_eot: bool,
) -> Dict:
    enc = _require_tiktoken()
    eot_id = enc.eot_token if hasattr(enc, "eot_token") else 50256
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {raw_dir}")
    manifest = json.load(manifest_path.open("r", encoding="utf-8"))
    shards = manifest.get("shards", [])
    if not shards:
        raise RuntimeError(f"No shards found in {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "tokens.bin"
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "tokenization_progress.json"

    progress = {
        "completed_shards": [],
        "total_tokens": 0,
        "total_samples": 0,
    }
    if progress_path.exists():
        progress = json.load(progress_path.open("r", encoding="utf-8"))

    completed = set(progress.get("completed_shards", []))
    mode = "ab" if tokens_path.exists() else "wb"
    total_tokens = int(progress.get("total_tokens", 0))
    total_samples = int(progress.get("total_samples", 0))

    with tokens_path.open(mode) as out_f:
        for shard in shards:
            shard_name = shard["path"]
            if shard_name in completed:
                continue
            shard_path = raw_dir / shard_name
            logger.info("Tokenizing shard: %s", shard_path)
            with shard_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get("text")
                    if not text:
                        continue
                    ids = enc.encode(text)
                    if append_eot:
                        ids.append(eot_id)
                    if not ids:
                        continue
                    arr = np.asarray(ids, dtype=np.uint16)
                    out_f.write(arr.tobytes())
                    total_tokens += len(arr)
                    total_samples += 1

            completed.add(shard_name)
            progress = {
                "completed_shards": sorted(completed),
                "total_tokens": total_tokens,
                "total_samples": total_samples,
            }
            with progress_path.open("w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2, sort_keys=True)
                f.write("\n")

    metadata = {
        "dataset_name": subset_name,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "vocab_size": enc.n_vocab,
        "max_seq_len": 2048,
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "prepared_from": str(raw_dir),
        "subset": manifest.get("subset"),
        "records_seen": manifest.get("records_seen"),
        "records_kept": manifest.get("records_kept"),
        "edu_threshold": manifest.get("edu_threshold"),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    logger.info(
        "Tokenization complete: subset=%s samples=%d tokens=%d output=%s",
        subset_name,
        total_samples,
        total_tokens,
        output_dir,
    )
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare GigaVerbo-v2 local artifacts for train_v2.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["acquire", "tokenize", "prepare"],
        help="acquire raw JSONL shards, tokenize existing shards, or do both",
    )
    parser.add_argument("--subset", choices=["edu", "synth", "non_edu"], default="edu")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_jsonl_glob", type=str, default=None)
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument("--edu_threshold", type=int, default=3)
    parser.add_argument("--edu_score_field", type=str, default="edu_int_score")
    parser.add_argument("--shard_size", type=int, default=25000)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--append_eot", action="store_true")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(args.verbose)

    raw_dir = Path(args.raw_dir) if args.raw_dir else DEFAULT_SHARED_DATASET_ROOT / "gigaverbo_v2" / "raw" / f"gigaverbo_v2_{args.subset}"
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_SHARED_DATASET_ROOT / "gigaverbo_v2" / "tokenized" / f"gigaverbo_v2_{args.subset}"
    dataset_name = args.dataset
    if dataset_name is None:
        dataset_name = (
            DEFAULT_GIGAVERBO_SYNTH_DATASET if args.subset == "synth" else DEFAULT_GIGAVERBO_DATASET
        )

    if args.command in {"acquire", "prepare"}:
        acquire_raw_jsonl(
            output_dir=raw_dir,
            subset=args.subset,
            dataset_name=dataset_name,
            dataset_split=args.split,
            dataset_config=args.dataset_config,
            local_jsonl_glob=args.local_jsonl_glob,
            streaming=args.streaming,
            cache_dir=args.cache_dir,
            text_field=args.text_field,
            edu_threshold=args.edu_threshold,
            edu_score_field=args.edu_score_field,
            shard_size=args.shard_size,
            max_records=args.max_records,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )

    if args.command in {"tokenize", "prepare"}:
        tokenize_jsonl_shards(
            raw_dir=raw_dir,
            output_dir=output_dir,
            subset_name=f"gigaverbo_v2_{args.subset}",
            append_eot=args.append_eot,
        )


if __name__ == "__main__":
    main()
