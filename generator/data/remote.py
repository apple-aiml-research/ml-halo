#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Where a dataset lives: ``data.data_dir`` is a local directory or an S3 URI.

- Local path: the dataset is read in place.
- ``s3://bucket/prefix/<dataset>``: the dataset is copied into
  ``<cache_dir>/bucket/prefix/<dataset>``, either all up front
  (``streaming_chunk_size: 0``, then read like a local dataset) or chunk by chunk
  (``streaming_chunk_size > 0``, see data/chunked_loader.py). Only that cache
  copy is ever deleted, and only when streaming.

Transfers use boto3 with the standard AWS credential chain.
"""

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def is_s3(data_dir: str) -> bool:
    """True when the dataset is an S3 URI, i.e. is streamed rather than read in place."""
    return str(data_dir).startswith("s3://")


def local_dir(data_dir: str, cache_dir: str) -> Path:
    """The local directory the dataset's files are read from: data_dir itself,
    or, for an S3 dataset, its copy under cache_dir (the full bucket/prefix path,
    so two datasets with the same name never share a cache)."""
    if is_s3(data_dir):
        return Path(cache_dir) / str(data_dir)[len("s3://"):].rstrip("/")
    return Path(data_dir)


# Concurrent transfers (download_episodes, download_prefix). LOADER_S3_WORKERS
# lowers it when the bucket is struggling.
S3_WORKERS = int(os.environ.get("LOADER_S3_WORKERS", "64"))

_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _client():
    """Shared in-process boto3 S3 client.

    In-process boto3 did ~478 obj/s on a ~21-core pod, against ~25-35 obj/s for
    per-object `aws s3 cp` processes (~0.6 s of CPU each for start-up and
    credential resolution).
    """
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                import boto3
                from botocore.config import Config
                _CLIENT = boto3.Session().client("s3", config=Config(
                    max_pool_connections=max(2 * S3_WORKERS, 64),
                    retries={"max_attempts": 10, "mode": "adaptive"}))
    return _CLIENT


def _split(s3_uri: str):
    bucket, _, key = s3_uri[len("s3://"):].partition("/")
    return bucket, key


def _retry(fn, what: str, max_retries: int = 8):
    """Run fn() with up to max_retries attempts, capped jittered exponential backoff.

    8 attempts: S3 transfers have timed out for hours at a time, and three
    strikes on ONE episode killed a whole job at its first chunk. Jitter keeps
    many threads from retrying in lockstep. Missing credentials fail at once.
    """
    from botocore.exceptions import NoCredentialsError
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1 or isinstance(e, NoCredentialsError):
                raise
            wait = min(120, 2 ** attempt) * (0.5 + random.random())
            print(f"S3 {what} failed ({e}), retry {attempt + 1}/{max_retries} "
                  f"in {wait:.0f}s...", flush=True)
            time.sleep(wait)


def download(s3_uri: str, local_path) -> None:
    """Download one object. It is written to a temp name and renamed, because
    callers treat "file exists" as "already downloaded" and must never see a
    partial file."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket, key = _split(s3_uri)
    tmp = local_path.with_name(local_path.name + f".part{threading.get_ident()}")

    def once():
        try:
            _client().download_file(bucket, key, str(tmp))
            os.replace(tmp, local_path)
        finally:
            tmp.unlink(missing_ok=True)

    _retry(once, f"download of {s3_uri}")


def download_prefix(s3_prefix: str, local: Path) -> None:
    """Download every object under s3_prefix into local, keeping relative paths.
    Files already in local are skipped, so an interrupted copy resumes."""
    bucket, prefix = _split(s3_prefix.rstrip("/") + "/")
    local = Path(local)

    def list_keys():
        pages = _client().get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix)
        return [o["Key"] for page in pages for o in page.get("Contents", [])]

    keys = _retry(list_keys, f"listing of {s3_prefix}")
    todo = [k for k in keys if not k.endswith("/")
            and not (local / k[len(prefix):]).exists()]
    with ThreadPoolExecutor(max_workers=S3_WORKERS) as pool:
        list(pool.map(lambda k: download(f"s3://{bucket}/{k}", local / k[len(prefix):]),
                      todo))


def exists(s3_uri: str) -> bool:
    from botocore.exceptions import ClientError
    bucket, key = _split(s3_uri)

    def head():
        try:
            _client().head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    return _retry(head, f"lookup of {s3_uri}")
