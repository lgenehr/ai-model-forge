#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import mmap
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:
    print("Missing dependency: numpy. Install with: pip install numpy", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    from langdetect import DetectorFactory, LangDetectException, detect
except ImportError as exc:
    print("Missing dependency: langdetect. Install with: pip install langdetect", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    print("Missing dependency: tqdm. Install with: pip install tqdm", file=sys.stderr)
    raise SystemExit(2) from exc


DetectorFactory.seed = 0

DEFAULT_DATASET_ROOT = Path("../bitnet-mamba-hybrid/data/tokenized")
LANGS = ("en", "pt")
SAMPLE_COUNT = 200
SAMPLE_LENGTH = 64
CHUNK_TOKENS = 4_000_000
QUICK_MAX_TOKENS_PER_LANG = 50_000_000
DECODE_CONTEXT_TOKENS = 16
RUN_ALERT_THRESHOLD = 128
HIGH_RUN_LENGTH_THRESHOLD = 512
HIGH_RUN_COUNT_THRESHOLD = 10
SUSPICIOUS_DOMINANCE_THRESHOLD = 0.10
REPETITION_WINDOW = 64
GARBAGE_CHAR_RATIO_THRESHOLD = 0.35
EDGE_TEXT_WINDOW = 8
LONG_RUN_CONTEXT_TOKENS = 12


def load_project_tokenizer():
    try:
        from inference import get_tokenizer  # type: ignore

        return get_tokenizer()
    except Exception:
        try:
            import tiktoken
        except ImportError as exc:
            print("Missing dependency: tiktoken. Install with: pip install tiktoken", file=sys.stderr)
            raise SystemExit(2) from exc

        return tiktoken.get_encoding("gpt2")


@dataclass
class DatasetSpec:
    language: str
    tokens_path: Path
    metadata_path: Path
    exists: bool
    metadata: Dict[str, Any]
    dtype: np.dtype
    file_size: int
    token_count: int
    estimated_tokens_from_size: int
    max_seq_len: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit tokenized CLM datasets.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing en/ and pt/ tokenized datasets.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLE_COUNT,
        help="Random decode samples per language.",
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=SAMPLE_LENGTH,
        help="Tokens per random sample.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_TOKENS,
        help="Streaming chunk size in tokens.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("dataset_report.json"),
        help="Optional JSON export path. Use --no-json to disable.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON export.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster audit on a capped number of tokens per language.",
    )
    parser.add_argument(
        "--quick-max-tokens-per-lang",
        type=int,
        default=QUICK_MAX_TOKENS_PER_LANG,
        help="Token cap per language when --quick is enabled.",
    )
    return parser.parse_args()


def dtype_from_metadata(dtype_name: Optional[str], file_size: int) -> np.dtype:
    if dtype_name:
        mapping = {
            "uint16": np.uint16,
            "int16": np.int16,
            "uint32": np.uint32,
            "int32": np.int32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported dtype in metadata: {dtype_name}")
        return np.dtype(mapping[dtype_name])

    for candidate in (np.uint16, np.int16, np.uint32, np.int32):
        if file_size % np.dtype(candidate).itemsize == 0:
            return np.dtype(candidate)
    raise ValueError("Unable to infer token dtype from file size.")


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def read_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset_spec(dataset_root: Path, language: str) -> DatasetSpec:
    lang_root = dataset_root / language
    tokens_path = lang_root / "tokens.bin"
    metadata_path = lang_root / "metadata.json"
    metadata = read_metadata(metadata_path)
    exists = tokens_path.exists() and metadata_path.exists()
    file_size = tokens_path.stat().st_size if tokens_path.exists() else 0
    dtype = dtype_from_metadata(metadata.get("dtype"), file_size) if tokens_path.exists() else np.dtype(np.uint16)
    estimated_tokens_from_size = file_size // dtype.itemsize if file_size else 0
    token_count = int(metadata.get("total_tokens", estimated_tokens_from_size))
    return DatasetSpec(
        language=language,
        tokens_path=tokens_path,
        metadata_path=metadata_path,
        exists=exists,
        metadata=metadata,
        dtype=dtype,
        file_size=file_size,
        token_count=token_count,
        estimated_tokens_from_size=estimated_tokens_from_size,
        max_seq_len=metadata.get("max_seq_len"),
    )


def detect_model_vocab_size(specs: Sequence[DatasetSpec]) -> int:
    for spec in specs:
        if spec.metadata.get("vocab_size") is not None:
            return int(spec.metadata["vocab_size"])
    return 50304


def memmap_tokens(spec: DatasetSpec) -> np.memmap:
    return np.memmap(spec.tokens_path, dtype=spec.dtype, mode="r")


def iter_chunks(arr: np.memmap, chunk_size: int, limit_tokens: Optional[int] = None) -> Iterable[Tuple[int, np.ndarray]]:
    total = len(arr) if limit_tokens is None else min(len(arr), limit_tokens)
    for start in range(0, total, chunk_size):
        yield start, np.asarray(arr[start : start + chunk_size])


def scan_long_runs(chunk: np.ndarray, carry_value: Optional[int], carry_length: int) -> Tuple[List[Dict[str, int]], Optional[int], int]:
    alerts: List[Dict[str, int]] = []
    if chunk.size == 0:
        return alerts, carry_value, carry_length

    changes = np.flatnonzero(chunk[1:] != chunk[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [chunk.size]))

    for start, end in zip(starts, ends):
        value = int(chunk[start])
        length = int(end - start)
        if carry_value == value and start == 0:
            length += carry_length
            run_start = -carry_length
        else:
            run_start = int(start)
        if length >= RUN_ALERT_THRESHOLD:
            alerts.append({"token_id": value, "run_length": length, "chunk_offset": run_start})

    last_value = int(chunk[starts[-1]])
    last_length = int(ends[-1] - starts[-1])
    if carry_value == last_value and starts[-1] == 0:
        last_length += carry_length
    return alerts, last_value, last_length


def looks_like_html(text: str) -> bool:
    return bool(re.search(r"</?[a-zA-Z][^>]{0,80}>", text))


def punctuation_run_length(text: str) -> int:
    matches = re.findall(r"[^\w\s]{4,}", text, flags=re.UNICODE)
    return max((len(match) for match in matches), default=0)


def control_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for ch in text if (ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 0xFFFD)
    return bad / len(text)


def estimate_garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    odd = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?\"'()[]{}-_/\\@#$%^&*+=<>`~|“”‘’…"))
    return odd / len(text)


def language_of_text(text: str) -> Optional[str]:
    stripped = text.strip()
    if len(stripped) < 20:
        return None
    try:
        return detect(stripped)
    except LangDetectException:
        return None


def special_token_map(tokenizer) -> Dict[str, Optional[int]]:
    eos_id = None
    eot = tokenizer._special_tokens.get("<|endoftext|>") if hasattr(tokenizer, "_special_tokens") else None
    if eot is not None:
        eos_id = int(eot)
    elif hasattr(tokenizer, "eot_token"):
        eos_id = int(tokenizer.eot_token)

    return {
        "bos": eos_id,
        "eos": eos_id,
        "pad": None,
        "unk": None,
    }


def top_k_from_counts(counts: np.ndarray, k: int) -> List[Dict[str, int]]:
    nonzero = np.flatnonzero(counts)
    if nonzero.size == 0:
        return []
    top_idx = nonzero[np.argsort(counts[nonzero])[-k:]][::-1]
    return [{"token_id": int(i), "count": int(counts[i])} for i in top_idx]


def rare_token_summary(counts: np.ndarray, rare_threshold: int = 10, limit: int = 50) -> Dict[str, Any]:
    rare_ids = np.flatnonzero((counts > 0) & (counts < rare_threshold))
    counts_by_freq = Counter(int(counts[i]) for i in rare_ids)
    return {
        "unique_rare_tokens": int(rare_ids.size),
        "count_buckets": dict(sorted(counts_by_freq.items())),
        "examples": [int(i) for i in rare_ids[:limit]],
    }


def analyze_decoded_text(text: str) -> Dict[str, Any]:
    flags: List[str] = []
    if "\ufffd" in text:
        first = text.find("\ufffd")
        last = text.rfind("\ufffd")
        near_left = first < EDGE_TEXT_WINDOW
        near_right = last >= max(0, len(text) - EDGE_TEXT_WINDOW)
        if near_left or near_right:
            flags.append("replacement_char_edge")
        if not (near_left or near_right):
            flags.append("replacement_char_mid")
    if looks_like_html(text):
        flags.append("html_fragment")
    punct_run = punctuation_run_length(text)
    if punct_run >= 8:
        flags.append("large_punctuation_run")
    ctrl_ratio = control_char_ratio(text)
    if ctrl_ratio > 0.02:
        flags.append("control_chars")
    garbage_ratio = estimate_garbage_ratio(text)
    if garbage_ratio > GARBAGE_CHAR_RATIO_THRESHOLD:
        flags.append("garbage_text")
    return {
        "flags": flags,
        "punctuation_run_length": punct_run,
        "control_char_ratio": ctrl_ratio,
        "garbage_ratio": garbage_ratio,
    }


def sample_start_indices(total_tokens: int, sample_length: int, num_samples: int, seed: int) -> List[int]:
    if total_tokens < sample_length:
        return []
    max_start = total_tokens - sample_length
    if max_start + 1 <= num_samples:
        return list(range(max_start + 1))
    rng = random.Random(seed)
    return sorted(rng.sample(range(max_start + 1), num_samples))


def decode_samples(
    spec: DatasetSpec,
    arr: np.memmap,
    tokenizer,
    sample_count: int,
    sample_length: int,
    tokenizer_vocab_size: int,
    effective_total_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    seed_map = {"en": 17, "pt": 29}
    total_tokens = len(arr) if effective_total_tokens is None else min(len(arr), effective_total_tokens)
    starts = sample_start_indices(total_tokens, sample_length, sample_count, seed=seed_map.get(spec.language, 0))
    results: List[Dict[str, Any]] = []
    language_counts: Counter[str] = Counter()
    mismatch_count = 0
    problematic_examples: List[Dict[str, Any]] = []
    informational_examples: List[Dict[str, Any]] = []
    contamination = 0

    expected_lang = {"en": "en", "pt": "pt"}.get(spec.language)

    for start in tqdm(starts, desc=f"Decoding {spec.language}", unit="sample"):
        sample = np.asarray(arr[start : start + sample_length], dtype=np.int64)
        sample_list = sample.tolist()
        in_tokenizer_range = [tok for tok in sample_list if 0 <= tok < tokenizer_vocab_size]
        invalid_utf8 = False
        left_ctx = min(DECODE_CONTEXT_TOKENS, start)
        right_limit = min(len(arr), total_tokens)
        right_ctx = min(DECODE_CONTEXT_TOKENS, max(0, right_limit - (start + sample_length)))
        ctx_tokens = np.asarray(
            arr[start - left_ctx : start + sample_length + right_ctx], dtype=np.int64
        ).tolist()
        ctx_tokens = [tok for tok in ctx_tokens if 0 <= tok < tokenizer_vocab_size]
        decoded = tokenizer.decode(in_tokenizer_range)
        decoded_with_context = tokenizer.decode(ctx_tokens) if ctx_tokens else decoded

        utf8_error_near_edge = False
        try:
            raw_bytes = tokenizer.decode_bytes(ctx_tokens if ctx_tokens else in_tokenizer_range)
            raw_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            invalid_utf8 = True
            error = None
            try:
                raw_bytes.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                error = exc
            if error is not None:
                utf8_error_near_edge = (
                    error.start < 4 or error.end >= max(0, len(raw_bytes) - 4)
                )
        except Exception:
            pass

        text_check = analyze_decoded_text(decoded)
        if invalid_utf8:
            if utf8_error_near_edge:
                text_check["flags"].append("invalid_utf8_edge")
            else:
                text_check["flags"].append("invalid_utf8_mid")
        if len(in_tokenizer_range) != len(sample_list):
            text_check["flags"].append("tokenizer_oob_in_sample")
        detected_lang = language_of_text(decoded)
        if detected_lang:
            language_counts[detected_lang] += 1
            if expected_lang and detected_lang != expected_lang:
                contamination += 1

        roundtrip_ok = False
        boundary_sensitive = False
        if all(0 <= tok < tokenizer_vocab_size for tok in sample_list):
            try:
                direct_roundtrip = tokenizer.encode(decoded) == sample_list
                if direct_roundtrip:
                    roundtrip_ok = True
                else:
                    boundary_sensitive = True
                    if ctx_tokens:
                        reencoded_ctx = tokenizer.encode(decoded_with_context)
                        if reencoded_ctx == ctx_tokens:
                            roundtrip_ok = True
            except Exception:
                roundtrip_ok = False
        if not roundtrip_ok:
            mismatch_count += 1
            if boundary_sensitive:
                text_check["flags"].append("boundary_sensitive_roundtrip")

        sample_result = {
            "start": int(start),
            "decoded_preview": decoded[:300],
            "decoded_with_context_preview": decoded_with_context[:300],
            "lang": detected_lang,
            "roundtrip_ok": roundtrip_ok,
            "flags": text_check["flags"],
        }
        results.append(sample_result)
        severe_flags = [
            flag for flag in text_check["flags"]
            if flag not in {"invalid_utf8_edge", "replacement_char_edge", "boundary_sensitive_roundtrip"}
        ]
        if severe_flags and len(problematic_examples) < 12:
            problematic_examples.append(sample_result)
        elif text_check["flags"] and len(informational_examples) < 12:
            informational_examples.append(sample_result)

    detected_total = sum(language_counts.values())
    contamination_pct = (contamination / detected_total * 100.0) if detected_total else 0.0
    mismatch_rate = (mismatch_count / len(starts) * 100.0) if starts else 0.0

    return {
        "sample_count": len(starts),
        "language_counts": dict(language_counts.most_common()),
        "contamination_pct": contamination_pct,
        "tokenizer_mismatch_rate_pct": mismatch_rate,
        "problematic_examples": problematic_examples,
        "informational_examples": informational_examples,
        "samples": results,
    }


def mmap_unique_counts(path: Path, dtype: np.dtype, token_count: int, vocab_size: int) -> Optional[int]:
    if token_count == 0:
        return 0
    try:
        with path.open("rb") as fh:
            mm = mmap.mmap(fh.fileno(), length=0, access=mmap.ACCESS_READ)
            arr = np.frombuffer(mm, dtype=dtype, count=token_count)
            unique = int(np.unique(arr).size)
            mm.close()
            return unique
    except Exception:
        return None


def audit_dataset(
    spec: DatasetSpec,
    tokenizer,
    model_vocab_size: int,
    chunk_size: int,
    sample_count: int,
    sample_length: int,
    quick: bool = False,
    quick_max_tokens_per_lang: Optional[int] = None,
) -> Dict[str, Any]:
    tokenizer_vocab_size = int(tokenizer.n_vocab)
    tokenizer_max_token_id = tokenizer_vocab_size - 1
    arr = memmap_tokens(spec)
    effective_total_tokens = len(arr)
    if quick and quick_max_tokens_per_lang is not None:
        effective_total_tokens = min(effective_total_tokens, quick_max_tokens_per_lang)

    counts = np.zeros(model_vocab_size, dtype=np.int64)
    negative_tokens = 0
    above_model_vocab = 0
    above_tokenizer_vocab = 0
    valid_token_total = 0
    same_as_prev_pairs = 0
    token_sum = 0.0
    token_sq_sum = 0.0
    min_token: Optional[int] = None
    max_token: Optional[int] = None
    long_runs: List[Dict[str, int]] = []
    carry_value: Optional[int] = None
    carry_length = 0

    progress = tqdm(total=effective_total_tokens, desc=f"Scanning {spec.language}", unit="tok")
    for _, chunk in iter_chunks(arr, chunk_size, limit_tokens=effective_total_tokens):
        chunk64 = chunk.astype(np.int64, copy=False)
        if chunk64.size == 0:
            continue

        chunk_min = int(chunk64.min())
        chunk_max = int(chunk64.max())
        min_token = chunk_min if min_token is None else min(min_token, chunk_min)
        max_token = chunk_max if max_token is None else max(max_token, chunk_max)
        token_sum += float(chunk64.sum(dtype=np.int64))
        token_sq_sum += float(np.square(chunk64, dtype=np.int64).sum(dtype=np.int64))

        negative_mask = chunk64 < 0
        above_model_mask = chunk64 >= model_vocab_size
        above_tokenizer_mask = chunk64 >= tokenizer_vocab_size

        negative_tokens += int(negative_mask.sum())
        above_model_vocab += int(above_model_mask.sum())
        above_tokenizer_vocab += int(above_tokenizer_mask.sum())

        valid_mask = (~negative_mask) & (~above_model_mask)
        valid = chunk64[valid_mask]
        valid_token_total += int(valid.size)
        if valid.size:
            counts[:model_vocab_size] += np.bincount(valid, minlength=model_vocab_size)

        if chunk64.size > 1:
            same_as_prev_pairs += int(np.sum(chunk64[1:] == chunk64[:-1]))
        if carry_value is not None and chunk64.size and int(chunk64[0]) == carry_value:
            same_as_prev_pairs += 1

        run_alerts, carry_value, carry_length = scan_long_runs(chunk64, carry_value, carry_length)
        for alert in run_alerts:
            if len(long_runs) < 50:
                token_id = alert["token_id"]
                token_text = None
                run_start = max(0, alert["chunk_offset"])
                context_start = max(0, run_start - LONG_RUN_CONTEXT_TOKENS)
                context_end = min(len(arr), run_start + alert["run_length"] + LONG_RUN_CONTEXT_TOKENS)
                run_context_preview = None
                if 0 <= token_id < tokenizer_vocab_size:
                    try:
                        token_text = tokenizer.decode([token_id]).replace("\n", "\\n")
                    except Exception:
                        token_text = None
                try:
                    context_tokens = np.asarray(arr[context_start:context_end], dtype=np.int64).tolist()
                    context_tokens = [tok for tok in context_tokens if 0 <= tok < tokenizer_vocab_size]
                    run_context_preview = tokenizer.decode(context_tokens).replace("\n", "\\n")[:240]
                except Exception:
                    run_context_preview = None
                alert["token_text"] = token_text
                alert["context_preview"] = run_context_preview
                long_runs.append(alert)

        progress.update(chunk64.size)
    progress.close()

    total_tokens = effective_total_tokens
    mean = token_sum / total_tokens if total_tokens else 0.0
    variance = (token_sq_sum / total_tokens - mean * mean) if total_tokens else 0.0
    std = math.sqrt(max(variance, 0.0))
    invalid_token_pct = ((negative_tokens + above_model_vocab) / total_tokens * 100.0) if total_tokens else 0.0
    tokenizer_oob_pct = (above_tokenizer_vocab / total_tokens * 100.0) if total_tokens else 0.0

    probs = counts[counts > 0] / max(valid_token_total, 1)
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
    unique_tokens = int(np.count_nonzero(counts))
    diversity = unique_tokens / max(valid_token_total, 1)
    dominant = top_k_from_counts(counts, 1)
    dominant_ratio = (dominant[0]["count"] / valid_token_total) if dominant and valid_token_total else 0.0
    repetition_ratio = same_as_prev_pairs / max(total_tokens - 1, 1)
    remainder = (total_tokens % spec.max_seq_len) if spec.max_seq_len else None

    decode_report = decode_samples(
        spec=spec,
        arr=arr,
        tokenizer=tokenizer,
        sample_count=sample_count,
        sample_length=sample_length,
        tokenizer_vocab_size=tokenizer_vocab_size,
        effective_total_tokens=effective_total_tokens,
    )

    specials = special_token_map(tokenizer)
    special_frequencies = {}
    for name, token_id in specials.items():
        special_frequencies[name] = {
            "token_id": token_id,
            "count": int(counts[token_id]) if token_id is not None and 0 <= token_id < len(counts) else None,
        }

    recommendations: List[str] = []
    if invalid_token_pct > 0:
        recommendations.append("Remove or regenerate out-of-range tokens before training.")
    if tokenizer_oob_pct > 0:
        recommendations.append("Tokenizer IDs exceed GPT-2 tokenizer range; verify tokenizer/model vocab padding assumptions.")
    if dominant_ratio >= SUSPICIOUS_DOMINANCE_THRESHOLD:
        recommendations.append("A small number of tokens dominate the corpus; inspect formatting artifacts and separator tokens.")
    severe_long_run_count = sum(1 for item in long_runs if item["run_length"] >= HIGH_RUN_LENGTH_THRESHOLD)
    if repetition_ratio > 0.10 or severe_long_run_count >= HIGH_RUN_COUNT_THRESHOLD:
        recommendations.append("High repetition detected; inspect duplicated shards, corrupted concatenation, or padding leakage.")
    if decode_report["contamination_pct"] > 10.0:
        recommendations.append("Language contamination is high enough to affect monolingual generation quality.")
    edge_flag_count = 0
    severe_flag_count = 0
    for sample in decode_report["samples"]:
        for flag in sample["flags"]:
            if flag in {"invalid_utf8_edge", "replacement_char_edge", "boundary_sensitive_roundtrip"}:
                edge_flag_count += 1
            else:
                severe_flag_count += 1

    if decode_report["tokenizer_mismatch_rate_pct"] > 5.0:
        if edge_flag_count >= severe_flag_count:
            recommendations.append("Tokenizer mismatches are mostly boundary-related in sampled windows; treat as informational unless confirmed on larger-context checks.")
        else:
            recommendations.append("Tokenizer roundtrip mismatches are elevated; verify whether samples are crossing byte-level token boundaries before treating this as corruption.")
    if not recommendations:
        recommendations.append("No severe structural issue detected; continue with targeted spot checks on decoded samples.")

    severe_decode_examples = len(decode_report["problematic_examples"])
    contamination_pct = decode_report["contamination_pct"]
    if invalid_token_pct > 0.001 or above_model_vocab > 0:
        status = "FAIL"
    elif contamination_pct > 10.0 or severe_decode_examples > 5:
        status = "FAIL"
    elif severe_long_run_count >= HIGH_RUN_COUNT_THRESHOLD or severe_decode_examples > 0:
        status = "PASS WITH MINOR WARNINGS"
    elif contamination_pct > 2.0:
        status = "PASS WITH MINOR WARNINGS"
    elif decode_report["tokenizer_mismatch_rate_pct"] > 5.0:
        status = "PASS WITH MINOR WARNINGS"
    else:
        status = "PASS"

    suspicious_tokens = []
    for item in top_k_from_counts(counts, 50):
        ratio = item["count"] / max(valid_token_total, 1)
        if ratio >= 0.01:
            suspicious_tokens.append({**item, "ratio": ratio})

    return {
        "language": spec.language,
        "status": status,
        "paths": {
            "tokens": str(spec.tokens_path),
            "metadata": str(spec.metadata_path),
        },
        "structure": {
            "exists": spec.exists,
            "file_size_bytes": spec.file_size,
            "file_size_human": human_bytes(spec.file_size),
            "dtype": str(spec.dtype),
            "estimated_tokens_from_size": spec.estimated_tokens_from_size,
            "metadata_total_tokens": spec.token_count,
            "token_count_matches_size": spec.token_count == spec.estimated_tokens_from_size,
            "max_seq_len": spec.max_seq_len,
            "audit_mode": "quick" if quick else "full",
            "audited_tokens": effective_total_tokens,
            "audited_fraction_pct": (effective_total_tokens / len(arr) * 100.0) if len(arr) else 0.0,
        },
        "range_validation": {
            "min_token_id": min_token,
            "max_token_id": max_token,
            "mean_token_value": mean,
            "std_token_value": std,
            "negative_tokens": negative_tokens,
            "above_model_vocab_tokens": above_model_vocab,
            "above_tokenizer_vocab_tokens": above_tokenizer_vocab,
            "invalid_token_pct": invalid_token_pct,
            "tokenizer_oob_pct": tokenizer_oob_pct,
            "model_vocab_size": model_vocab_size,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "tokenizer_max_token_id": tokenizer_max_token_id,
        },
        "distribution": {
            "top_50_tokens": top_k_from_counts(counts, 50),
            "rare_tokens": rare_token_summary(counts),
            "dominant_token_ratio": dominant_ratio,
            "suspicious_dominant_tokens": suspicious_tokens,
            "long_identical_runs": long_runs,
            "severe_long_run_count": severe_long_run_count,
        },
        "decode_analysis": {
            "sample_count": decode_report["sample_count"],
            "problematic_examples": decode_report["problematic_examples"],
            "informational_examples": decode_report["informational_examples"],
        },
        "language_validation": {
            "detected_languages": decode_report["language_counts"],
            "contamination_pct": decode_report["contamination_pct"],
        },
        "tokenizer_consistency": {
            "mismatch_rate_pct": decode_report["tokenizer_mismatch_rate_pct"],
        },
        "special_tokens": special_frequencies,
        "quality_metrics": {
            "entropy_bits": entropy,
            "token_diversity": diversity,
            "repetition_ratio": repetition_ratio,
            "unique_tokens": unique_tokens,
            "truncated_sequence_remainder": remainder,
        },
        "recommendations": recommendations,
    }


def format_findings(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Language: {report['language']}")
    lines.append(f"Status: {report['status']}")

    structure = report["structure"]
    lines.append("  Dataset structure")
    lines.append(f"  - exists: {structure['exists']}")
    lines.append(f"  - file size: {structure['file_size_human']} ({structure['file_size_bytes']:,} bytes)")
    lines.append(f"  - dtype: {structure['dtype']}")
    lines.append(f"  - estimated tokens: {structure['estimated_tokens_from_size']:,}")
    lines.append(f"  - metadata tokens: {structure['metadata_total_tokens']:,}")
    lines.append(f"  - count matches metadata: {structure['token_count_matches_size']}")
    lines.append(
        f"  - audit mode: {structure['audit_mode']} "
        f"({structure['audited_tokens']:,} tokens, {structure['audited_fraction_pct']:.4f}% of file)"
    )

    ranges = report["range_validation"]
    lines.append("  Token range validation")
    lines.append(
        f"  - min/max/mean/std: {ranges['min_token_id']} / {ranges['max_token_id']} / "
        f"{ranges['mean_token_value']:.2f} / {ranges['std_token_value']:.2f}"
    )
    lines.append(
        f"  - invalid tokens: {ranges['invalid_token_pct']:.6f}% "
        f"(negative={ranges['negative_tokens']:,}, >=model_vocab={ranges['above_model_vocab_tokens']:,})"
    )
    lines.append(
        f"  - tokenizer out-of-range: {ranges['tokenizer_oob_pct']:.6f}% "
        f"(>= {ranges['tokenizer_vocab_size']:,}: {ranges['above_tokenizer_vocab_tokens']:,})"
    )

    dist = report["distribution"]
    lines.append("  Token distribution")
    lines.append(f"  - dominant token ratio: {dist['dominant_token_ratio']:.4%}")
    lines.append(f"  - rare tokens (<10 occurrences): {dist['rare_tokens']['unique_rare_tokens']:,}")
    lines.append(f"  - long identical runs flagged: {len(dist['long_identical_runs'])}")
    lines.append(f"  - severe long runs (>= {HIGH_RUN_LENGTH_THRESHOLD}): {dist['severe_long_run_count']}")
    lines.append(f"  - top 10 tokens: {dist['top_50_tokens'][:10]}")
    if dist["long_identical_runs"]:
        lines.append(f"  - long run examples: {dist['long_identical_runs'][:5]}")

    quality = report["quality_metrics"]
    lines.append("  Dataset quality metrics")
    lines.append(
        f"  - entropy: {quality['entropy_bits']:.4f} bits | diversity: {quality['token_diversity']:.8f} | "
        f"repetition ratio: {quality['repetition_ratio']:.6f}"
    )
    lines.append(f"  - unique tokens: {quality['unique_tokens']:,}")
    lines.append(f"  - truncated remainder vs max_seq_len: {quality['truncated_sequence_remainder']}")

    lang = report["language_validation"]
    tok = report["tokenizer_consistency"]
    lines.append("  Decode and language checks")
    lines.append(f"  - detected languages: {lang['detected_languages']}")
    lines.append(f"  - language contamination: {lang['contamination_pct']:.2f}%")
    lines.append(f"  - tokenizer mismatch rate: {tok['mismatch_rate_pct']:.2f}%")

    lines.append("  Special tokens")
    for name, payload in report["special_tokens"].items():
        lines.append(f"  - {name.upper()}: id={payload['token_id']} count={payload['count']}")

    lines.append("  Problematic decode examples")
    examples = report["decode_analysis"]["problematic_examples"]
    if not examples:
        lines.append("  - none in sampled decodes")
    else:
        for example in examples[:5]:
            preview = example["decoded_preview"].replace("\n", "\\n")
            lines.append(
                f"  - offset {example['start']}: flags={example['flags']} lang={example['lang']} text={preview[:180]}"
            )
    info_examples = report["decode_analysis"]["informational_examples"]
    if info_examples:
        lines.append("  Informational decode examples")
        for example in info_examples[:5]:
            preview = example["decoded_preview"].replace("\n", "\\n")
            lines.append(
                f"  - offset {example['start']}: flags={example['flags']} lang={example['lang']} text={preview[:180]}"
            )

    lines.append("  Recommendations")
    for item in report["recommendations"]:
        lines.append(f"  - {item}")
    return "\n".join(lines)


def final_summary(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = {report["language"]: report["status"] for report in reports}
    if any(status == "FAIL" for status in statuses.values()):
        overall_status = "FAIL"
    elif any(status == "PASS WITH MINOR WARNINGS" for status in statuses.values()):
        overall_status = "PASS WITH MINOR WARNINGS"
    else:
        overall_status = "PASS"
    return {
        "overall_status": overall_status,
        "dataset_statuses": statuses,
        "dataset_summary": {
            report["language"]: {
                "status": report["status"],
                "file_size_human": report["structure"]["file_size_human"],
                "estimated_tokens": report["structure"]["estimated_tokens_from_size"],
                "audit_mode": report["structure"]["audit_mode"],
            "audited_tokens": report["structure"]["audited_tokens"],
            "audited_fraction_pct": report["structure"]["audited_fraction_pct"],
            "invalid_token_pct": report["range_validation"]["invalid_token_pct"],
            "contamination_pct": report["language_validation"]["contamination_pct"],
            "tokenizer_mismatch_rate_pct": report["tokenizer_consistency"]["mismatch_rate_pct"],
            }
            for report in reports
        },
        "recommendations": sorted({rec for report in reports for rec in report["recommendations"]}),
    }


def print_report(reports: Sequence[Dict[str, Any]], overall: Dict[str, Any]) -> None:
    print("=" * 100)
    print("DATASET AUDIT REPORT")
    print("=" * 100)
    print(f"OVERALL STATUS: {overall['overall_status']}")
    print(f"DATASET STATUSES: {json.dumps(overall['dataset_statuses'], indent=2)}")
    print("-" * 100)
    for report in reports:
        print(format_findings(report))
        print("-" * 100)

    print("FINAL REPORT")
    print(f"Dataset summary: {json.dumps(overall['dataset_summary'], indent=2)}")
    print(f"Recommendation list: {json.dumps(overall['recommendations'], indent=2)}")


def main() -> None:
    args = parse_args()
    print("Checking dependencies and dataset structure...")
    specs = [build_dataset_spec(args.dataset_root, lang) for lang in LANGS]

    missing = [spec for spec in specs if not spec.exists]
    if missing:
        for spec in missing:
            print(f"Missing expected dataset files for {spec.language}: {spec.tokens_path} / {spec.metadata_path}")
        raise SystemExit(1)

    tokenizer = load_project_tokenizer()
    print("Dependencies OK. Starting dataset audit...")
    if args.quick:
        print(
            f"Quick mode enabled: auditing up to {args.quick_max_tokens_per_lang:,} tokens per language."
        )
    model_vocab_size = detect_model_vocab_size(specs)
    reports = [
        audit_dataset(
            spec=spec,
            tokenizer=tokenizer,
            model_vocab_size=model_vocab_size,
            chunk_size=args.chunk_size,
            sample_count=args.samples,
            sample_length=args.sample_length,
            quick=args.quick,
            quick_max_tokens_per_lang=args.quick_max_tokens_per_lang,
        )
        for spec in specs
    ]
    overall = final_summary(reports)
    print_report(reports, overall)

    if not args.no_json:
        payload = {
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "dataset_root": str(args.dataset_root),
            "reports": reports,
            "final_report": overall,
        }
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
