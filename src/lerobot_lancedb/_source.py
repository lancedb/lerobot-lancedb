#!/usr/bin/env python

# Copyright 2026 lerobot-lancedb contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Dataset source adapters for local, Hugging Face Hub, S3, and fsspec roots.

Allows ``lerobot-lance-convert --video-mode external`` to list objects,
materialize only ``meta/`` plus referenced ``data/*.parquet`` shards, and
range-read MP4 headers without downloading video payloads.
"""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

DEFAULT_IO_WORKERS = 16
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")


@dataclass(frozen=True)
class SourceObject:
    """One file discovered under a dataset root, relative to that root."""

    rel_path: str
    size: int


def is_uri(path: str | Path) -> bool:
    """Return True when ``path`` looks like a URI (contains ``://``)."""
    return "://" in str(path)


def is_remote_uri(path: str | Path) -> bool:
    """Return True for non-local URIs (``s3://``, ``gs://``, ``hf://``, etc.)."""
    raw = str(path)
    return "://" in raw and not raw.startswith("file://")


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected file:// URI, got {uri!r}")
    if parsed.netloc and parsed.netloc != "localhost":
        raise ValueError(f"Non-local file:// host is not supported: {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"Query and fragment are not supported in file:// URI: {uri!r}")
    if not parsed.path.startswith("/"):
        raise ValueError(f"Expected absolute file:// path, got {uri!r}")
    return Path(unquote(parsed.path))


def normalize_base_uri(uri: str) -> str:
    """Validate and canonicalize an external blob base URI for Lance ``initial_bases``."""
    raw = uri.strip()
    if not is_uri(raw):
        raise ValueError(
            "External video base URI must be an absolute URI with scheme "
            f"(e.g. s3://, hf://, file://), got {uri!r}"
        )
    parsed = urlsplit(raw)
    if not parsed.scheme:
        raise ValueError(f"Invalid external video base URI (missing scheme): {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(
            "External video base URI must not contain a query or fragment ('?' or '#'); "
            f"percent-encode literal characters in path segments instead: {uri!r}"
        )
    if parsed.scheme == "file":
        if parsed.netloc and parsed.netloc != "localhost":
            raise ValueError(f"Non-local file:// host is not supported: {uri!r}")
        if not parsed.path.startswith("/"):
            raise ValueError(f"file:// base URI must be an absolute path: {uri!r}")
    elif not parsed.netloc:
        raise ValueError(f"Invalid external video base URI (missing host/bucket): {uri!r}")
    safe = "/@" if parsed.scheme == "hf" else "/"
    clean_path = quote(unquote(parsed.path.rstrip("/")), safe=safe)
    if parsed.scheme == "file" and not clean_path:
        clean_path = "/"
    return f"{parsed.scheme}://{parsed.netloc}{clean_path}"


def validate_rel_path(rel_path: str) -> list[str]:
    """Validate a POSIX relative path and return its non-empty path segments."""
    if not rel_path or "\x00" in rel_path or "\\" in rel_path or rel_path.startswith("/"):
        raise ValueError(f"Unsafe relative path: {rel_path!r}")
    parts = rel_path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe relative path: {rel_path!r}")
    return parts


def _safe_staging_path(staging_dir: Path, rel_path: str) -> Path:
    """Validate ``rel_path`` and return a target strictly inside ``staging_dir``."""
    parts = validate_rel_path(rel_path)
    base = staging_dir.resolve()
    dst = staging_dir.joinpath(*parts)
    parent_resolved = dst.parent.resolve()
    if not parent_resolved.is_relative_to(base) or not (parent_resolved / dst.name).is_relative_to(base):
        raise ValueError(f"Relative path escapes staging directory: {rel_path!r}")
    return dst


def _validate_content_range(
    header: str | None, *, offset: int, max_length: int, label: str
) -> int:
    if not header:
        raise RuntimeError(f"Missing Content-Range header on HTTP 206 response for {label}")
    match = _CONTENT_RANGE_RE.match(header.strip())
    if not match:
        raise RuntimeError(f"Malformed Content-Range header {header!r} for {label}")
    start, end = int(match.group(1)), int(match.group(2))
    total_raw = match.group(3)
    if start != offset or end < start:
        raise RuntimeError(
            f"Unexpected Content-Range {header!r} (requested offset {offset}) for {label}"
        )
    if total_raw != "*":
        total = int(total_raw)
        if end >= total:
            raise RuntimeError(
                f"Invalid Content-Range {header!r} (last byte {end} >= total length {total}) for {label}"
            )
    expected_len = end - start + 1
    if expected_len > max_length:
        raise RuntimeError(
            f"Content-Range {header!r} returns {expected_len} bytes, exceeding requested {max_length} for {label}"
        )
    return expected_len


def parse_hf_dataset_uri(uri: str) -> tuple[str, str | None, str]:
    """Parse ``hf://datasets/<repo>[@rev][/subpath]`` into ``(repo_id, revision, subpath)``."""
    parsed = urlsplit(uri)
    if parsed.scheme != "hf" or parsed.netloc != "datasets":
        raise ValueError(f"Only hf://datasets/... URIs are supported, got {uri!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"Query and fragment are not supported in Hugging Face URI: {uri!r}")
    rest = parsed.path.strip("/")
    if not rest:
        raise ValueError(f"Empty Hugging Face dataset URI: {uri!r}")
    parts = rest.split("/")
    if len(parts) == 1:
        repo_part, raw_subpath = parts[0], ""
    elif "@" in parts[0]:
        repo_part, raw_subpath = parts[0], "/".join(parts[1:])
    else:
        repo_part = f"{parts[0]}/{parts[1]}"
        raw_subpath = "/".join(parts[2:])
    repo_id, _, rev = repo_part.partition("@")
    if not repo_id:
        raise ValueError(f"Invalid Hugging Face dataset URI: {uri!r}")
    return unquote(repo_id), (unquote(rev) or None), unquote(raw_subpath)


def _is_commit_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{40}", value) is not None


def _parse_retry_after(header_value: str | None, default: float) -> float:
    if not header_value:
        return min(default, 60.0)
    try:
        return max(0.0, min(float(header_value), 60.0))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(header_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delay = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, min(delay, 60.0))
    except (TypeError, ValueError):
        return min(default, 60.0)


def _default_base_from_root(root_uri: str, video_rels: list[str]) -> str:
    clean = root_uri.rstrip("/")
    if video_rels and all(rel.startswith("videos/") for rel in video_rels):
        return f"{clean}/videos"
    return clean


class DatasetSource:
    """Base interface for reading a LeRobot v3.0 dataset without copying MP4s."""

    root_label: str

    def __enter__(self) -> DatasetSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        return None

    def list_files(self) -> dict[str, SourceObject]:
        raise NotImplementedError

    def materialize_meta(self, staging_dir: Path) -> Path:
        """Return a local directory containing ``meta/`` (staging if remote)."""
        raise NotImplementedError

    def materialize_data(self, rel_paths: list[str], staging_dir: Path) -> list[Path]:
        """Return local paths for the requested ``data/*.parquet`` files in order."""
        raise NotImplementedError

    def read_range(self, rel_path: str, offset: int, length: int) -> bytes:
        """Read ``length`` bytes from ``rel_path`` starting at ``offset``."""
        raise NotImplementedError

    def default_video_base_uri(self, video_rels: list[str]) -> str:
        """Return the base URI registered in ``initial_bases`` for external videos."""
        raise NotImplementedError


class LocalSource(DatasetSource):
    """Dataset already on local disk (reads in place without staging copies)."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root_label = str(self.root)
        if not (self.root / "meta").is_dir():
            raise FileNotFoundError(f"{self.root} does not look like a LeRobot dataset (no meta/ dir)")

    def list_files(self) -> dict[str, SourceObject]:
        objects: dict[str, SourceObject] = {}
        for path in self.root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(self.root).as_posix()
                objects[rel] = SourceObject(rel_path=rel, size=path.stat().st_size)
        return objects

    def materialize_meta(self, staging_dir: Path) -> Path:
        return self.root

    def materialize_data(self, rel_paths: list[str], staging_dir: Path) -> list[Path]:
        return [_safe_staging_path(self.root, rel) for rel in rel_paths]

    def read_range(self, rel_path: str, offset: int, length: int) -> bytes:
        with open(_safe_staging_path(self.root, rel_path), "rb") as fh:
            fh.seek(offset)
            return fh.read(length)

    def default_video_base_uri(self, video_rels: list[str]) -> str:
        return _default_base_from_root(self.root.as_uri(), video_rels)


class HubSource(DatasetSource):
    """Hugging Face Hub dataset source pinned to an immutable 40-char commit SHA."""

    def __init__(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        subpath: str = "",
        token: str | None = None,
        max_workers: int = DEFAULT_IO_WORKERS,
    ) -> None:
        import huggingface_hub
        import requests
        from requests.adapters import HTTPAdapter

        self.repo_id = repo_id
        self.subpath = subpath.strip("/")
        self.max_workers = max(1, max_workers)
        self.token = token or os.environ.get("HF_TOKEN") or huggingface_hub.get_token()

        api = huggingface_hub.HfApi(token=self.token)
        info = api.dataset_info(repo_id, revision=revision or "main")
        if not info.sha:
            raise RuntimeError(f"Could not resolve commit SHA for {repo_id!r}@{revision or 'main'}")
        self.commit_sha = info.sha
        self._api = api

        encoded_subpath = quote(self.subpath, safe="/") if self.subpath else ""
        prefix = f"/{encoded_subpath}" if encoded_subpath else ""
        self.root_uri = f"hf://datasets/{self.repo_id}@{self.commit_sha}{prefix}"
        self.root_label = self.root_uri

        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.max_workers + 8,
            pool_maxsize=self.max_workers + 8,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        if self.token:
            session.headers["Authorization"] = f"Bearer {self.token}"
        self._session = session
        self._files_cache: dict[str, SourceObject] | None = None

    def close(self) -> None:
        self._session.close()

    def _repo_rel(self, rel_path: str) -> str:
        return f"{self.subpath}/{rel_path}" if self.subpath else rel_path

    def list_files(self) -> dict[str, SourceObject]:
        if self._files_cache is not None:
            return self._files_cache
        from huggingface_hub.hf_api import RepoFile

        objects: dict[str, SourceObject] = {}
        strip_prefix = f"{self.subpath}/" if self.subpath else ""
        for entry in self._api.list_repo_tree(
            self.repo_id,
            path_in_repo=self.subpath or None,
            revision=self.commit_sha,
            repo_type="dataset",
            recursive=True,
        ):
            if not isinstance(entry, RepoFile):
                continue
            if strip_prefix and not entry.path.startswith(strip_prefix):
                continue
            rel = entry.path[len(strip_prefix) :] if strip_prefix else entry.path
            objects[rel] = SourceObject(rel_path=rel, size=int(entry.size or 0))
        self._files_cache = objects
        return objects

    def _download_rel_paths(self, rel_paths: list[str], staging_dir: Path) -> list[Path]:
        import huggingface_hub

        targets = [(rel, _safe_staging_path(staging_dir, rel)) for rel in rel_paths]

        def _fetch(rel: str, dst: Path) -> Path:
            if dst.exists():
                return dst
            cached = Path(
                huggingface_hub.hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self._repo_rel(rel),
                    repo_type="dataset",
                    revision=self.commit_sha,
                    token=self.token,
                )
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.tmp-{threading.get_ident()}")
            try:
                try:
                    os.symlink(cached, tmp)
                except OSError:
                    import shutil

                    shutil.copy2(cached, tmp)
                tmp.replace(dst)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            return dst

        results: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(_fetch, rel, dst): rel for rel, dst in targets}
            for fut in as_completed(futs):
                rel = futs[fut]
                results[rel] = fut.result()
        return [results[rel] for rel in rel_paths]

    def materialize_meta(self, staging_dir: Path) -> Path:
        files = self.list_files()
        meta_rels = sorted(rel for rel in files if rel.startswith("meta/"))
        if not meta_rels:
            raise FileNotFoundError(f"{self.root_uri} does not look like a LeRobot dataset (no meta/ files)")
        self._download_rel_paths(meta_rels, staging_dir)
        return staging_dir

    def materialize_data(self, rel_paths: list[str], staging_dir: Path) -> list[Path]:
        return self._download_rel_paths(rel_paths, staging_dir)

    def read_range(self, rel_path: str, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        import requests

        repo_path = quote(self._repo_rel(rel_path), safe="/")
        endpoint = self._api.endpoint.rstrip("/")
        url = f"{endpoint}/datasets/{self.repo_id}/resolve/{self.commit_sha}/{repo_path}"
        end = offset + length - 1
        last_err: Exception | None = None
        for attempt in range(7):
            try:
                resp = self._session.get(
                    url,
                    headers={
                        "Range": f"bytes={offset}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    timeout=60.0,
                    allow_redirects=True,
                    stream=True,
                )
            except requests.RequestException as err:
                last_err = err
                if attempt == 6:
                    raise
                time.sleep(min(1.5 * (2**attempt), 15.0))
                continue

            retry_delay: float | None = None
            with resp:
                if resp.status_code == 206:
                    encoding = (resp.headers.get("Content-Encoding") or "").strip().lower()
                    if encoding and encoding != "identity":
                        raw_stream = getattr(resp, "raw", None)
                        if raw_stream is not None:
                            raw_stream.close()
                        raise RuntimeError(
                            f"Unexpected Content-Encoding {encoding!r} on HTTP 206 response for {url}"
                        )
                    expected_len = _validate_content_range(
                        resp.headers.get("Content-Range"),
                        offset=offset,
                        max_length=length,
                        label=url,
                    )
                    chunks: list[bytes] = []
                    received = 0
                    for chunk in resp.iter_content(chunk_size=min(65536, max(1, expected_len))):
                        if not chunk:
                            continue
                        received += len(chunk)
                        if received > expected_len:
                            raw_stream = getattr(resp, "raw", None)
                            if raw_stream is not None:
                                raw_stream.close()
                            raise RuntimeError(
                                f"HTTP 206 response exceeded Content-Range ({expected_len} bytes) for {url}"
                            )
                        chunks.append(chunk)
                    if received != expected_len:
                        raise RuntimeError(
                            f"Truncated HTTP 206 response ({received}/{expected_len} bytes) for {url}"
                        )
                    return b"".join(chunks)

                if resp.status_code == 200:
                    raw_stream = getattr(resp, "raw", None)
                    if raw_stream is not None:
                        raw_stream.close()
                    raise RuntimeError(
                        f"Server ignored Range request (returned HTTP 200 instead of 206) for {url}"
                    )

                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 6:
                    retry_delay = _parse_retry_after(resp.headers.get("Retry-After"), 1.5 * (2**attempt))
                else:
                    resp.raise_for_status()
                    raise RuntimeError(f"Unexpected HTTP {resp.status_code} for Range request on {url}")

            if retry_delay is not None:
                time.sleep(retry_delay)
        raise RuntimeError(f"Range GET failed for {url} after retries: {last_err}")

    def default_video_base_uri(self, video_rels: list[str]) -> str:
        return _default_base_from_root(self.root_uri, video_rels)


def _s3_client_kwargs(storage_options: dict[str, str] | None, max_workers: int) -> dict[str, Any]:
    from botocore import UNSIGNED
    from botocore.config import Config

    opts = {k.lower(): v for k, v in (storage_options or {}).items()}
    cfg_kwargs: dict[str, Any] = {"max_pool_connections": max_workers + 8}
    skip_sig = opts.get("aws_skip_signature") or opts.get("skip_signature") or ""
    if skip_sig.lower() in ("1", "true", "yes"):
        cfg_kwargs["signature_version"] = UNSIGNED
    kwargs: dict[str, Any] = {"config": Config(**cfg_kwargs)}

    region = opts.get("aws_region") or opts.get("region") or opts.get("aws_default_region")
    if region:
        kwargs["region_name"] = region
    endpoint = opts.get("aws_endpoint") or opts.get("aws_endpoint_url") or opts.get("endpoint") or opts.get("endpoint_url")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    if "aws_access_key_id" in opts:
        kwargs["aws_access_key_id"] = opts["aws_access_key_id"]
    if "aws_secret_access_key" in opts:
        kwargs["aws_secret_access_key"] = opts["aws_secret_access_key"]
    if "aws_session_token" in opts:
        kwargs["aws_session_token"] = opts["aws_session_token"]
    return kwargs


class S3Source(DatasetSource):
    """Amazon S3 (or S3-compatible) dataset source using boto3."""

    def __init__(
        self,
        uri: str,
        *,
        storage_options: dict[str, str] | None = None,
        max_workers: int = DEFAULT_IO_WORKERS,
    ) -> None:
        import boto3

        parsed = urlsplit(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Expected s3:// URI, got {uri!r}")
        if parsed.query or parsed.fragment:
            raise ValueError(
                "Query and fragment are not supported in s3:// root URI; "
                f"percent-encode literal '?' or '#' in prefix segments instead: {uri!r}"
            )
        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"Invalid S3 URI (missing bucket): {uri!r}")
        raw_prefix = unquote(parsed.path).strip("/")
        encoded_prefix = quote(raw_prefix, safe="/")
        self.bucket = bucket
        self.prefix = f"{raw_prefix}/" if raw_prefix else ""
        self.root_uri = f"s3://{bucket}/{encoded_prefix}" if encoded_prefix else f"s3://{bucket}"
        self.root_label = self.root_uri
        self.max_workers = max(1, max_workers)
        self._client_kwargs = _s3_client_kwargs(storage_options, self.max_workers)
        self._client = boto3.client("s3", **self._client_kwargs)
        self._files_cache: dict[str, SourceObject] | None = None

    def close(self) -> None:
        self._client.close()

    def list_files(self) -> dict[str, SourceObject]:
        if self._files_cache is not None:
            return self._files_cache
        objects: dict[str, SourceObject] = {}
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for entry in page.get("Contents", []):
                key = entry["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(self.prefix) :] if self.prefix else key
                objects[rel] = SourceObject(rel_path=rel, size=int(entry["Size"]))
        self._files_cache = objects
        return objects

    def _download_rel_paths(self, rel_paths: list[str], staging_dir: Path) -> list[Path]:
        targets = [(rel, _safe_staging_path(staging_dir, rel)) for rel in rel_paths]

        def _fetch(rel: str, dst: Path) -> Path:
            if dst.exists():
                return dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.tmp-{threading.get_ident()}")
            try:
                self._client.download_file(self.bucket, f"{self.prefix}{rel}", str(tmp))
                tmp.replace(dst)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            return dst

        results: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(_fetch, rel, dst): rel for rel, dst in targets}
            for fut in as_completed(futs):
                rel = futs[fut]
                results[rel] = fut.result()
        return [results[rel] for rel in rel_paths]

    def materialize_meta(self, staging_dir: Path) -> Path:
        files = self.list_files()
        meta_rels = sorted(rel for rel in files if rel.startswith("meta/"))
        if not meta_rels:
            raise FileNotFoundError(f"{self.root_uri} does not look like a LeRobot dataset (no meta/ files)")
        self._download_rel_paths(meta_rels, staging_dir)
        return staging_dir

    def materialize_data(self, rel_paths: list[str], staging_dir: Path) -> list[Path]:
        return self._download_rel_paths(rel_paths, staging_dir)

    def read_range(self, rel_path: str, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        key = f"{self.prefix}{rel_path}"
        label = f"s3://{self.bucket}/{key}"
        resp = self._client.get_object(
            Bucket=self.bucket,
            Key=key,
            Range=f"bytes={offset}-{offset + length - 1}",
        )
        body = resp["Body"]
        try:
            status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
            if status is not None and status != 206:
                raise RuntimeError(f"S3 server ignored Range request (HTTP {status}) for {label}")
            expected_len = _validate_content_range(
                resp.get("ContentRange"), offset=offset, max_length=length, label=label
            )
            data = body.read(expected_len + 1)
            if len(data) > expected_len:
                raise RuntimeError(
                    f"S3 Range response exceeded Content-Range ({expected_len} bytes) for {label}"
                )
            if len(data) < expected_len:
                raise RuntimeError(
                    f"Truncated S3 Range response ({len(data)}/{expected_len} bytes) for {label}"
                )
            return data
        finally:
            body.close()

    def default_video_base_uri(self, video_rels: list[str]) -> str:
        return _default_base_from_root(self.root_uri, video_rels)


class FsspecSource(DatasetSource):
    """Generic fsspec dataset source (gs://, az://, abfs://, memory://, or s3:// fallback)."""

    def __init__(
        self,
        uri: str,
        *,
        storage_options: dict[str, str] | None = None,
        max_workers: int = DEFAULT_IO_WORKERS,
    ) -> None:
        from fsspec.core import url_to_fs

        canonical = normalize_base_uri(uri)
        parsed = urlsplit(canonical)
        decoded_path = unquote(parsed.path)
        decoded_url = f"{parsed.scheme}://{parsed.netloc}{decoded_path}"
        self.root_uri = canonical
        self.root_label = canonical
        self.max_workers = max(1, max_workers)
        self._fs, root_path = url_to_fs(decoded_url, **(storage_options or {}))
        self._root_path = root_path.rstrip("/")
        self._is_gcs = parsed.scheme == "gs" and all(
            hasattr(self._fs, attr) for attr in ("_set_session", "url", "_get_headers", "_get_params", "loop")
        )
        self._files_cache: dict[str, SourceObject] | None = None

    def _full_path(self, rel_path: str) -> str:
        return f"{self._root_path}/{rel_path}" if self._root_path else rel_path

    def list_files(self) -> dict[str, SourceObject]:
        if self._files_cache is not None:
            return self._files_cache
        objects: dict[str, SourceObject] = {}
        prefix = f"{self._root_path}/" if self._root_path else ""
        details = self._fs.find(self._root_path, detail=True)
        for full_path, info in details.items():
            if info.get("type") == "directory":
                continue
            normalized = full_path.lstrip("/") if not prefix.startswith("/") else full_path
            rel = normalized[len(prefix) :] if prefix and normalized.startswith(prefix) else normalized
            rel = rel.lstrip("/")
            objects[rel] = SourceObject(rel_path=rel, size=int(info.get("size") or 0))
        self._files_cache = objects
        return objects

    def _download_rel_paths(self, rel_paths: list[str], staging_dir: Path) -> list[Path]:
        targets = [(rel, _safe_staging_path(staging_dir, rel)) for rel in rel_paths]

        def _fetch(rel: str, dst: Path) -> Path:
            if dst.exists():
                return dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(f".{dst.name}.tmp-{threading.get_ident()}")
            try:
                tmp.write_bytes(self._fs.cat_file(self._full_path(rel)))
                tmp.replace(dst)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            return dst

        results: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(_fetch, rel, dst): rel for rel, dst in targets}
            for fut in as_completed(futs):
                rel = futs[fut]
                results[rel] = fut.result()
        return [results[rel] for rel in rel_paths]

    def materialize_meta(self, staging_dir: Path) -> Path:
        files = self.list_files()
        meta_rels = sorted(rel for rel in files if rel.startswith("meta/"))
        if not meta_rels:
            raise FileNotFoundError(f"{self.root_uri} does not look like a LeRobot dataset (no meta/ files)")
        self._download_rel_paths(meta_rels, staging_dir)
        return staging_dir

    def materialize_data(self, rel_paths: list[str], staging_dir: Path) -> list[Path]:
        return self._download_rel_paths(rel_paths, staging_dir)

    def _read_gcs_range(self, rel_path: str, offset: int, length: int) -> bytes:
        import asyncio

        from fsspec.asyn import sync

        async def _fetch() -> bytes:
            await self._fs._set_session()
            full_path = self._full_path(rel_path)
            url = self._fs.url(full_path)
            end = offset + length - 1
            params = self._fs._get_params({})
            timeout = getattr(self._fs, "requests_timeout", 60.0)
            last_err: Exception | None = None

            for attempt in range(7):
                headers = self._fs._get_headers(
                    {"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"}
                )
                retry_delay: float | None = None
                try:
                    async with self._fs.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=timeout,
                        auto_decompress=False,
                    ) as resp:
                        if resp.status == 206:
                            encoding = (resp.headers.get("Content-Encoding") or "").strip().lower()
                            if encoding and encoding != "identity":
                                resp.close()
                                raise RuntimeError(
                                    f"Unexpected Content-Encoding {encoding!r} on GCS Range response for {url}"
                                )
                            expected_len = _validate_content_range(
                                resp.headers.get("Content-Range"),
                                offset=offset,
                                max_length=length,
                                label=url,
                            )
                            chunks: list[bytes] = []
                            received = 0
                            async for chunk in resp.content.iter_chunked(min(65536, max(1, expected_len))):
                                if not chunk:
                                    continue
                                received += len(chunk)
                                if received > expected_len:
                                    resp.close()
                                    raise RuntimeError(
                                        f"GCS HTTP 206 response exceeded Content-Range ({expected_len} bytes) "
                                        f"for {url}"
                                    )
                                chunks.append(chunk)
                            if received != expected_len:
                                raise RuntimeError(
                                    f"Truncated GCS HTTP 206 response ({received}/{expected_len} bytes) "
                                    f"for {url}"
                                )
                            return b"".join(chunks)

                        if resp.status == 200:
                            resp.close()
                            raise RuntimeError(
                                f"GCS server ignored Range request (returned HTTP 200 instead of 206) for {url}"
                            )

                        if resp.status in (429, 500, 502, 503, 504) and attempt < 6:
                            retry_delay = _parse_retry_after(
                                resp.headers.get("Retry-After"), 1.5 * (2**attempt)
                            )
                            resp.close()
                        else:
                            resp.close()
                            resp.raise_for_status()
                            raise RuntimeError(
                                f"Unexpected GCS HTTP {resp.status} for Range request on {url}"
                            )
                except (OSError, asyncio.TimeoutError) as err:
                    last_err = err
                    if attempt == 6:
                        raise
                    retry_delay = min(1.5 * (2**attempt), 15.0)

                if retry_delay is not None:
                    await asyncio.sleep(retry_delay)

            raise RuntimeError(f"GCS Range GET failed for {url} after retries: {last_err}")

        return sync(self._fs.loop, _fetch)

    def read_range(self, rel_path: str, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if self._is_gcs:
            return self._read_gcs_range(rel_path, offset, length)
        data = self._fs.cat_file(self._full_path(rel_path), start=offset, end=offset + length)
        if len(data) > length:
            raise RuntimeError(
                f"Range read returned {len(data)} bytes, exceeding requested {length} "
                f"for {self.root_uri}/{rel_path}"
            )
        return data

    def default_video_base_uri(self, video_rels: list[str]) -> str:
        return _default_base_from_root(self.root_uri, video_rels)


def open_source(
    *,
    repo_id: str | None = None,
    root: str | Path | None = None,
    revision: str | None = None,
    storage_options: dict[str, str] | None = None,
    max_workers: int = DEFAULT_IO_WORKERS,
) -> DatasetSource:
    """Open the appropriate :class:`DatasetSource` for ``root`` or ``repo_id``."""
    if root is not None:
        raw = str(root)
        if raw.startswith("file://"):
            return LocalSource(_file_uri_to_path(raw))
        if raw.startswith("hf://"):
            parsed_repo, uri_rev, subpath = parse_hf_dataset_uri(raw)
            if revision and uri_rev and revision != uri_rev:
                raise ValueError(
                    f"Conflicting revisions: --revision={revision!r} vs URI revision {uri_rev!r} in {raw!r}"
                )
            token = (storage_options or {}).get("token") or (storage_options or {}).get("hf_token")
            return HubSource(
                parsed_repo,
                revision=revision or uri_rev,
                subpath=subpath,
                token=token,
                max_workers=max_workers,
            )
        if raw.startswith("s3://"):
            try:
                import boto3  # noqa: F401
            except ImportError:
                try:
                    return FsspecSource(raw, storage_options=storage_options, max_workers=max_workers)
                except ImportError as err:
                    raise ImportError(
                        "Reading an s3:// source requires 'boto3' or 's3fs'. "
                        "Install S3 support with: pip install 'lerobot-lancedb[s3]'"
                    ) from err
            return S3Source(raw, storage_options=storage_options, max_workers=max_workers)
        if raw.startswith("gs://"):
            try:
                return FsspecSource(raw, storage_options=storage_options, max_workers=max_workers)
            except ImportError as err:
                raise ImportError(
                    "Reading a gs:// source requires 'gcsfs'. "
                    "Install GCS support with: pip install 'lerobot-lancedb[gcs]'"
                ) from err
        if is_uri(raw):
            return FsspecSource(raw, storage_options=storage_options, max_workers=max_workers)
        return LocalSource(Path(raw))

    if repo_id is not None:
        token = (storage_options or {}).get("token") or (storage_options or {}).get("hf_token")
        return HubSource(repo_id, revision=revision, token=token, max_workers=max_workers)

    raise ValueError("pass --repo-id or --root")
