"""Stream, Filter, and Upload FineWeb Climate & Nature Data
===========================================================

This script streams the HuggingFace FineWeb-Edu dataset, filters for climate and nature
keywords, and progressively uploads matched records to a Hugging Face dataset repository
using non-blocking async uploads for efficiency.

Features:
---------
* Streams any FineWeb-Edu subset (default: sample-10BT)
* Filters using academic climate/nature keywords with strength scoring
* Async uploads to HF Hub without blocking the pipeline
* Schema normalization to prevent dataset viewer errors
* Max samples parameter for testing before full runs
* Suitable for long-running SLURM jobs

Usage:
------
# Test run with 1000 samples
python stream_filter_upload_fineweb.py \\
    --hub-repo-id username/fineweb-climate \\
    --subset sample-10BT \\
    --max-samples 1000 \\
    --chunk-size 100

# Full production run
python stream_filter_upload_fineweb.py \\
    --hub-repo-id username/fineweb-climate \\
    --subset CC-MAIN-2024-10 \\
    --chunk-size 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from datasets import Dataset, Features, IterableDataset, Sequence as DsSequence, Value, load_dataset
except ImportError as exc:
    raise ImportError(
        "The 'datasets' package is required. Install it via 'pip install datasets'."
    ) from exc

try:
    from huggingface_hub import HfApi, login
except ImportError as exc:
    raise ImportError(
        "The 'huggingface_hub' package is required. Install it via 'pip install huggingface_hub'."
    ) from exc


# ============================================================================
# KEYWORD FILTERING LOGIC
# ============================================================================

STRENGTH_STRONG = "strong"
STRENGTH_WEAK = "weak"


@dataclass(frozen=True)
class KeywordRule:
    """Compiled regex keyword with a strength tag for filtering logic."""
    
    term: str
    group: str
    strength: str
    pattern: re.Pattern[str]


KeywordSpec = Tuple[str, str, str]  # (term, group, strength)


def _compile_pattern(term: str) -> re.Pattern[str]:
    """Compile a whitespace-tolerant, word-boundary-safe regex for term."""
    tokens = term.strip().split()
    joined = r"\s+".join(re.escape(tok) for tok in tokens)
    pattern = rf"(?<!\w){joined}(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


def _build_rules() -> Tuple[KeywordRule, ...]:
    """Create compiled keyword rules with strength metadata."""
    
    specs: Sequence[KeywordSpec] = (
        # Climate: strong signals
        ("climate change", "climate", STRENGTH_STRONG),
        ("global warming", "climate", STRENGTH_STRONG),
        ("greenhouse gas", "climate", STRENGTH_STRONG),
        ("ghg", "climate", STRENGTH_STRONG),
        ("carbon dioxide", "climate", STRENGTH_STRONG),
        ("co2", "climate", STRENGTH_STRONG),
        ("methane", "climate", STRENGTH_STRONG),
        ("nitrous oxide", "climate", STRENGTH_STRONG),
        ("carbon budget", "climate", STRENGTH_STRONG),
        ("net zero", "climate", STRENGTH_STRONG),
        ("decarbonization", "climate", STRENGTH_STRONG),
        ("renewable energy", "climate", STRENGTH_STRONG),
        ("carbon capture", "climate", STRENGTH_STRONG),
        ("carbon sequestration", "climate", STRENGTH_STRONG),
        ("negative emissions", "climate", STRENGTH_STRONG),
        ("emissions trading", "climate", STRENGTH_STRONG),
        ("carbon pricing", "climate", STRENGTH_STRONG),
        ("carbon neutrality", "climate", STRENGTH_STRONG),
        ("integrated assessment model", "climate", STRENGTH_STRONG),
        ("climate scenario", "climate", STRENGTH_STRONG),
        ("climate modelling", "climate", STRENGTH_STRONG),
        ("ipcc", "climate", STRENGTH_STRONG),
        ("el niño", "climate", STRENGTH_STRONG),
        ("enso", "climate", STRENGTH_STRONG),
        ("radiative forcing", "climate", STRENGTH_STRONG),
        ("radiative imbalance", "climate", STRENGTH_STRONG),
        ("arctic amplification", "climate", STRENGTH_STRONG),
        ("ocean heat content", "climate", STRENGTH_STRONG),
        ("sea level rise", "climate", STRENGTH_STRONG),
        ("permafrost", "climate", STRENGTH_STRONG),
        ("cryosphere", "climate", STRENGTH_STRONG),
        ("monsoon", "climate", STRENGTH_STRONG),
        ("flood risk", "climate", STRENGTH_STRONG),
        ("storm surge", "climate", STRENGTH_STRONG),
        ("extreme weather", "climate", STRENGTH_STRONG),
        ("heatwave", "climate", STRENGTH_STRONG),
        ("drought", "climate", STRENGTH_STRONG),
        ("climate hazard", "climate", STRENGTH_STRONG),
        ("loss and damage", "climate", STRENGTH_STRONG),
        ("loss and damage fund", "climate", STRENGTH_STRONG),
        ("1.5°c", "climate", STRENGTH_STRONG),
        ("2°c", "climate", STRENGTH_STRONG),
        ("paris agreement", "climate", STRENGTH_STRONG),
        ("global stocktake", "climate", STRENGTH_STRONG),
        ("nationally determined contribution", "climate", STRENGTH_STRONG),
        ("ndc", "climate", STRENGTH_STRONG),
        ("just transition", "climate", STRENGTH_STRONG),
        ("transition pathway", "climate", STRENGTH_STRONG),
        ("climate litigation", "climate", STRENGTH_STRONG),
        ("climate governance", "climate", STRENGTH_STRONG),
        ("climate finance", "climate", STRENGTH_STRONG),
        ("climate justice", "climate", STRENGTH_STRONG),
        
        # Climate: weaker / more ambiguous
        ("adaptation", "climate", STRENGTH_WEAK),
        ("climate resilience", "climate", STRENGTH_WEAK),
        ("climate risk", "climate", STRENGTH_WEAK),
        ("climate policy", "climate", STRENGTH_WEAK),
        ("climate mainstreaming", "climate", STRENGTH_WEAK),
        ("climate diplomacy", "climate", STRENGTH_WEAK),
        ("climate accountability", "climate", STRENGTH_WEAK),
        ("health impact", "climate", STRENGTH_WEAK),
        ("food security", "climate", STRENGTH_WEAK),
        ("water security", "climate", STRENGTH_WEAK),
        ("vector-borne disease", "climate", STRENGTH_WEAK),
        ("climate migration", "climate", STRENGTH_WEAK),
        ("disaster risk reduction", "climate", STRENGTH_WEAK),
        ("compound event", "climate", STRENGTH_WEAK),
        ("climate system", "climate", STRENGTH_WEAK),
        ("climate variability", "climate", STRENGTH_WEAK),
        ("climate dynamics", "climate", STRENGTH_WEAK),
        ("energy balance", "climate", STRENGTH_WEAK),
        ("water vapour feedback", "climate", STRENGTH_WEAK),
        ("atmospheric circulation", "climate", STRENGTH_WEAK),
        ("paleoclimate", "climate", STRENGTH_WEAK),
        ("proxy record", "climate", STRENGTH_WEAK),
        ("temperature rise", "climate", STRENGTH_WEAK),
        
        # Nature: strong signals
        ("biodiversity", "nature", STRENGTH_STRONG),
        ("ecosystem services", "nature", STRENGTH_STRONG),
        ("ecosystem integrity", "nature", STRENGTH_STRONG),
        ("ecosystem", "nature", STRENGTH_STRONG),
        ("ecosystems", "nature", STRENGTH_STRONG),
        ("habitat", "nature", STRENGTH_STRONG),
        ("species loss", "nature", STRENGTH_STRONG),
        ("endangered species", "nature", STRENGTH_STRONG),
        ("invasive species", "nature", STRENGTH_STRONG),
        ("conservation", "nature", STRENGTH_STRONG),
        ("restoration", "nature", STRENGTH_STRONG),
        ("reforestation", "nature", STRENGTH_STRONG),
        ("afforestation", "nature", STRENGTH_STRONG),
        ("nature-based solution", "nature", STRENGTH_STRONG),
        ("nature-based solutions", "nature", STRENGTH_STRONG),
        ("natural capital", "nature", STRENGTH_STRONG),
        ("wetland", "nature", STRENGTH_STRONG),
        ("wetlands", "nature", STRENGTH_STRONG),
        ("peatland", "nature", STRENGTH_STRONG),
        ("mangrove", "nature", STRENGTH_STRONG),
        ("mangroves", "nature", STRENGTH_STRONG),
        ("wildlife corridor", "nature", STRENGTH_STRONG),
        ("protected area", "nature", STRENGTH_STRONG),
        ("forest management", "nature", STRENGTH_STRONG),
        ("land degradation", "nature", STRENGTH_STRONG),
        ("soil health", "nature", STRENGTH_STRONG),
        ("pollinator", "nature", STRENGTH_STRONG),
        ("biodiversity finance", "nature", STRENGTH_STRONG),
        ("living planet index", "nature", STRENGTH_STRONG),
        ("ipbes", "nature", STRENGTH_STRONG),
        ("biocultural diversity", "nature", STRENGTH_STRONG),
        ("genetic diversity", "nature", STRENGTH_STRONG),
        ("deforestation", "nature", STRENGTH_STRONG),
        ("landscape connectivity", "nature", STRENGTH_STRONG),
        
        # Nature: weak / broader environmental terms
        ("ecological", "nature", STRENGTH_WEAK),
        ("ecosystem health", "nature", STRENGTH_WEAK),
        ("biodiversity loss", "nature", STRENGTH_WEAK),
    )
    
    rules = []
    for term, group, strength in specs:
        rules.append(
            KeywordRule(
                term=term,
                group=group,
                strength=strength,
                pattern=_compile_pattern(term)
            )
        )
    
    return tuple(rules)


KEYWORD_RULES: Tuple[KeywordRule, ...] = _build_rules()


@dataclass
class MatchResult:
    """Container describing keyword matches for a passage."""
    
    tags: List[str]
    matched_keywords: Dict[str, List[str]]
    strong_count: int
    weak_count: int
    total_count: int
    decision: str
    
    @property
    def has_match(self) -> bool:
        return bool(self.tags)


def normalise_text(text: str) -> str:
    """Return a lowercase, whitespace-collapsed string for regex checks."""
    return re.sub(r"\s+", " ", text.casefold())


def detect_keywords(text: str) -> MatchResult:
    """Identify climate and nature keywords using regex with strength logic."""
    
    lowered = normalise_text(text)
    matches: Dict[str, List[str]] = defaultdict(list)
    strong_hits = 0
    weak_hits = 0
    
    for rule in KEYWORD_RULES:
        if rule.pattern.search(lowered):
            matches[rule.group].append(rule.term)
            if rule.strength == STRENGTH_STRONG:
                strong_hits += 1
            else:
                weak_hits += 1
    
    total_hits = strong_hits + weak_hits
    decision = "rejected_insufficient"
    if strong_hits > 0:
        decision = "accepted_strong"
    elif total_hits >= 2:
        decision = "accepted_multi_weak"
    
    tags = sorted(matches) if decision.startswith("accepted") else []
    matches = {group: sorted(list(set(terms))) for group, terms in matches.items()} if tags else {}
    
    return MatchResult(
        tags=tags,
        matched_keywords=matches,
        strong_count=strong_hits,
        weak_count=weak_hits,
        total_count=total_hits,
        decision=decision,
    )


# ============================================================================
# STREAMING UPLOADER
# ============================================================================

class StreamingUploader:
    """
    Non-blocking uploader for streaming pipelines.
    
    Uploads shards to HF Hub without blocking the main processing loop.
    Uses run_as_future for async uploads and tracks pending uploads.
    """
    
    def __init__(self, repo_id: str, token: Optional[str] = None, temp_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.pending_uploads: List[Future] = []
        self.shard_counter = 0

        base_tmp = None
        if temp_dir:
            base_tmp = Path(temp_dir)
            base_tmp.mkdir(parents=True, exist_ok=True)

        self.temp_dir = tempfile.mkdtemp(prefix="fineweb_upload_", dir=str(base_tmp) if base_tmp else None)
        
        # Track nested dict keys for schema consistency across ALL shards
        # Pre-define schema for matched_keywords to prevent schema drift
        self.nested_schema_keys: Dict[str, set] = {
            "matched_keywords": {"climate", "nature"}
        }
        
        # Create repo if needed
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        logging.info(f"✓ Repository ready: https://huggingface.co/datasets/{repo_id}")
    
    def _normalize_schema(self, data: List[dict]) -> List[dict]:
        """
        Normalize schema across all records to prevent schema mismatch between shards.
        
        This ensures fields like 'matched_keywords' have consistent structure
        (e.g., always have both 'climate' and 'nature' keys, even if empty).
        """
        # First pass: collect all nested dict keys from this batch and update global tracker
        for record in data:
            for key, value in record.items():
                if isinstance(value, dict):
                    if key not in self.nested_schema_keys:
                        self.nested_schema_keys[key] = set()
                    self.nested_schema_keys[key].update(value.keys())
        
        # Second pass: normalize all records using the global schema
        normalized = []
        for record in data:
            new_record = record.copy()
            for key, all_subkeys in self.nested_schema_keys.items():
                if key in new_record and isinstance(new_record[key], dict):
                    # Make a copy and add missing subkeys with empty lists
                    new_record[key] = new_record[key].copy()
                    for subkey in all_subkeys:
                        if subkey not in new_record[key]:
                            new_record[key][subkey] = []
                elif key not in new_record:
                    # Field doesn't exist at all, add it with empty subkeys
                    new_record[key] = {subkey: [] for subkey in all_subkeys}
            normalized.append(new_record)
        
        return normalized
    
    def _get_features(self, sample_record: dict) -> Features:
        """
        Build explicit Features schema to ensure consistent types across shards.
        This prevents empty lists from being inferred as 'null' type.
        """
        features = {}
        
        # Define schema based on known FineWeb fields + our additions
        for key, value in sample_record.items():
            if key == "tags":
                # Always a list of strings
                features[key] = DsSequence(Value("string"))
            elif key == "matched_keywords":
                # Dict with string keys -> list of strings
                features[key] = {k: DsSequence(Value("string")) for k in value.keys()}
            elif key == "match_summary":
                # Nested dict with int/string values
                features[key] = {
                    "strong": Value("int64"),
                    "weak": Value("int64"),
                    "total": Value("int64"),
                    "decision": Value("string"),
                }
            elif isinstance(value, str):
                features[key] = Value("string")
            elif isinstance(value, int):
                features[key] = Value("int64")
            elif isinstance(value, float):
                features[key] = Value("float64")
            elif isinstance(value, list):
                # Assume list of strings
                features[key] = DsSequence(Value("string"))
            elif isinstance(value, dict):
                # Generic dict handling
                features[key] = {k: Value("string") for k in value.keys()}
            else:
                features[key] = Value("string")  # fallback
        
        return Features(features)
    
    def upload_shard(self, data: List[dict]) -> Future:
        """
        Upload a chunk of data as a Parquet shard.
        
        Returns immediately and upload happens in background.
        Returns the Future object so you can check status if needed.
        """
        shard_idx = self.shard_counter
        self.shard_counter += 1
        
        # Normalize schema to prevent mismatches between shards
        normalized_data = self._normalize_schema(data)
        
        # Build explicit features schema to ensure consistent types
        features = self._get_features(normalized_data[0])
        
        # Convert to Dataset with explicit schema and save as Parquet
        ds = Dataset.from_list(normalized_data, features=features)
        shard_name = f"train-{shard_idx:05d}-of-99999.parquet"
        parquet_path = Path(self.temp_dir) / shard_name
        ds.to_parquet(parquet_path)
        
        logging.info(f"📤 Queuing shard {shard_idx:05d} ({len(data)} records)...")
        
        # Upload non-blocking
        future = self.api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"data/{shard_name}",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add shard {shard_idx:05d}",
            run_as_future=True,  # 👈 Key for non-blocking!
        )
        
        self.pending_uploads.append(future)
        # Clean up completed futures
        self.pending_uploads = [f for f in self.pending_uploads if not f.done()]
        logging.info(f"   ↳ Queued (pending uploads: {len(self.pending_uploads)})")
        
        return future
    
    def wait_for_completion(self):
        """Wait for all pending uploads to complete."""
        if not self.pending_uploads:
            return
        
        logging.info(f"\n⏳ Waiting for {len(self.pending_uploads)} pending uploads...")
        for i, future in enumerate(self.pending_uploads):
            try:
                future.result()  # Block until complete
                logging.info(f"   ✓ Upload {i + 1}/{len(self.pending_uploads)} complete")
            except Exception as e:
                logging.error(f"   ✗ Upload {i + 1} failed: {e}")
        
        self.pending_uploads = []
        logging.info("✓ All uploads complete!")
    
    def cleanup(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def stream_fineweb(
    dataset_name: str,
    subset: str,
    split: str,
    token: Optional[str] = None,
    retries: int = 5,
    backoff: int = 10,
) -> IterableDataset:
    """Create a streaming dataset split with retry on network errors."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(
                dataset_name,
                name=subset,
                split=split,
                streaming=True,
                token=token,
                verification_mode="no_checks",
            )
        except Exception as exc:  # Keep broad to catch HF/network issues
            last_err = exc
            if attempt == retries:
                break
            wait_for = backoff * attempt
            logging.warning(
                "FineWeb stream load failed (attempt %s/%s): %s | retrying in %ss",
                attempt,
                retries,
                exc,
                wait_for,
            )
            time.sleep(wait_for)
    raise last_err  # type: ignore[misc]


def iter_filter_records(
    dataset: IterableDataset,
    text_column: str,
    max_samples: Optional[int] = None,
    filter_tags: Optional[List[str]] = None,
) -> Iterator[dict]:
    """Yield JSON-ready records that match the keyword criteria.
    
    Args:
        dataset: Streaming dataset to filter
        text_column: Column containing text to analyze
        max_samples: Optional limit on records to process
        filter_tags: If provided, only yield records with ALL specified tags
                    (e.g., ['climate'] for climate-only, ['nature'] for nature-only)
    """
    
    processed = 0
    matched = 0
    start_ts = time.time()
    for index, row in enumerate(dataset):
        processed += 1
        
        if max_samples is not None and processed > max_samples:
            logging.info(f"Reached max_samples limit: {max_samples}")
            break
        
        # Log progress every 5,000 records
        if processed % 5000 == 0:
            elapsed = max(time.time() - start_ts, 1e-6)
            rate = processed / elapsed
            logging.info(
                "Progress: processed=%s matched=%s rate=%.0f rec/s",
                f"{processed:,}",
                f"{matched:,}",
                rate,
            )
        
        text = row.get(text_column)
        if not isinstance(text, str):
            logging.debug(f"Skipping record {index}: missing text column '{text_column}'")
            continue
        
        match = detect_keywords(text)
        if not match.has_match:
            continue
        
        # Filter by specific tags if requested
        if filter_tags:
            if not all(tag in match.tags for tag in filter_tags):
                continue
        
        matched += 1
        # Create enriched record with match metadata
        enriched = dict(row)
        enriched["tags"] = match.tags
        enriched["matched_keywords"] = match.matched_keywords
        enriched["match_summary"] = {
            "strong": match.strong_count,
            "weak": match.weak_count,
            "total": match.total_count,
            "decision": match.decision,
        }
        
        yield enriched


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--hub-repo-id",
        required=True,
        help="Target HuggingFace dataset repository (e.g., username/fineweb-climate)"
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb-edu",
        help="Source dataset to stream (default: HuggingFaceFW/fineweb-edu)"
    )
    parser.add_argument(
        "--subset",
        default="sample-10BT",
        help="Dataset subset/config to stream (default: sample-10BT)"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream (default: train)"
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing text content to filter (default: text)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of filtered records per upload chunk (default: 500)"
    )
    parser.add_argument(
        "--temp-dir",
        default="/scratch/sraj",
        help="Directory for temporary parquet shards (default: /scratch/sraj)"
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=5,
        help="Retries for FineWeb streaming load (default: 5)"
    )
    parser.add_argument(
        "--retry-backoff",
        type=int,
        default=10,
        help="Backoff seconds multiplier between retries (default: 10)"
    )
    parser.add_argument(
        "--filter-tags",
        nargs="+",
        choices=["climate", "nature"],
        help="Only upload records with these tags (e.g., --filter-tags climate)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of records to process (for testing)"
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token (defaults to HF_TOKEN env var or hf_token.txt)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Get token
    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        token_file = Path(__file__).parent / "hf_token.txt"
        if token_file.exists():
            token = token_file.read_text().strip()
    
    if token:
        login(token=token)
        logging.info("✓ Logged in to Hugging Face")
    else:
        logging.warning("⚠ No token found. Some datasets may require authentication.")
    
    # Log configuration
    logging.info("=" * 70)
    logging.info("FineWeb Climate & Nature Filter Pipeline")
    logging.info("=" * 70)
    logging.info(f"Hub Repo:     {args.hub_repo_id}")
    logging.info(f"Dataset:      {args.dataset}")
    logging.info(f"Subset:       {args.subset}")
    logging.info(f"Split:        {args.split}")
    logging.info(f"Chunk Size:   {args.chunk_size}")
    temp_dir_arg = args.temp_dir if args.temp_dir else None
    logging.info(f"Max Samples:  {args.max_samples or 'unlimited'}")
    logging.info(f"Temp Dir:     {temp_dir_arg or 'system temp'}")
    logging.info(f"Retries:      {args.retry_attempts} (backoff {args.retry_backoff}s)")
    logging.info(f"Filter Tags:  {args.filter_tags or 'all (climate + nature)'}")
    logging.info("=" * 70)
    
    # Initialize uploader
    uploader = StreamingUploader(repo_id=args.hub_repo_id, token=token, temp_dir=temp_dir_arg)
    
    # Stream and filter
    logging.info("\n🚀 Starting streaming pipeline...")
    start_time = time.time()
    
    dataset = stream_fineweb(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        token=token,
        retries=args.retry_attempts,
        backoff=args.retry_backoff,
    )
    
    filtered_iter = iter_filter_records(
        dataset=dataset,
        text_column=args.text_column,
        max_samples=args.max_samples,
        filter_tags=args.filter_tags
    )
    
    # Process in chunks
    buffer = []
    total_matched = 0
    
    for record in filtered_iter:
        buffer.append(record)
        total_matched += 1
        
        # Upload when buffer reaches chunk size
        if len(buffer) >= args.chunk_size:
            uploader.upload_shard(buffer)
            buffer = []
            logging.info(f"   → Total matched records: {total_matched:,}")
    
    # Upload any remaining data
    if buffer:
        uploader.upload_shard(buffer)
    
    # Wait for all uploads to complete
    uploader.wait_for_completion()
    uploader.cleanup()
    
    elapsed = time.time() - start_time
    logging.info("\n" + "=" * 70)
    logging.info("🎉 Pipeline Complete!")
    logging.info(f"   Matched records: {total_matched:,}")
    logging.info(f"   Time elapsed:    {elapsed:.2f}s")
    logging.info(f"   View dataset:    https://huggingface.co/datasets/{args.hub_repo_id}")
    if total_matched == 0:
        logging.warning("No matches were found. Check subset/text column/token availability.")
    logging.info("=" * 70)
    logging.info(f"\n💡 Load with: load_dataset('{args.hub_repo_id}')")


if __name__ == "__main__":
    main()
