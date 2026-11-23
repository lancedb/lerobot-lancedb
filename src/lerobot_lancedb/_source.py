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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

DEFAULT_IO_WORKERS = 16


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
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected file:// URI, got {uri!r}")
    if parsed.netloc and parsed.netloc != "localhost":
        raise ValueError(f"Non-local file:// host is not supported: {uri!r}")
    return Path(unquote(parsed.path))


def parse_hf_dataset_uri(uri: str) -> tuple[str, str | None, str]:
    """Parse ``hf://datasets/<repo>[@rev][/subpath]`` into ``(repo_id, revision, subpath)``."""
    prefix = "hf://datasets/"
    if not uri.startswith(prefix):
        raise ValueError(f"Only hf://datasets/... URIs are supported, got {uri!r}")
    rest = uri[len(prefix) :].strip("/")
    if not rest:
        raise ValueError(f"Empty Hugging Face dataset URI: {uri!r}")
    parts = rest.split("/")
    if len(parts) == 1:
        repo_part, subpath = parts[0], ""
    elif "@" in parts[0]:
        repo_part, subpath = parts[0], "/".join(parts[1:])
    else:
        repo_part = f"{parts[0]}/{parts[1]}"
        subpath = "/".join(parts[2:])
    repo_id, _, rev = repo_part.partition("@")
    if not repo_id:
        raise ValueError(f"Invalid Hugging Face dataset URI: {uri!r}")
    return repo_id, (rev or None), subpath


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
        return [self.root / rel for rel in rel_paths]

    def read_range(self, rel_path: str, offset: int, length: int) -> bytes:
        with open(self.root / rel_path, "rb") as fh:
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

        prefix = f"/{self.subpath}" if self.subpath else ""
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

        def _fetch(rel: str) -> Path:
            dst = staging_dir / rel
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
            tmp = dst.with_suffix(dst.suffix + f".tmp-{threading.get_ident()}")
            try:
                os.symlink(cached, tmp)
            except OSError:
                import shutil

                shutil.copy2(cached, tmp)
            tmp.replace(dst)
            return dst

        results: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(_fetch, rel): rel for rel in rel_paths}
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
                    headers={"Range": f"bytes={offset}-{end}"},
                    timeout=60.0,
                    allow_redirects=True,
                )
            except requests.RequestException as err:
                last_err = err
                if attempt == 6:
                    raise
                time.sleep(min(1.5 * (2**attempt), 15.0))
                continue

            if resp.status_code == 206:
                return resp.content
            if resp.status_code == 200:
                return resp.content[offset : offset + length]
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 6:
                delay = _parse_retry_after(resp.headers.get("Retry-After"), 1.5 * (2**attempt))
                time.sleep(delay)
                continue
            resp.raise_for_status()
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

        clean = uri.rstrip("/")
        if not clean.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got {uri!r}")
        bucket, _, prefix = clean[5:].partition("/")
        if not bucket:
            raise ValueError(f"Invalid S3 URI (missing bucket): {uri!r}")
        self.root_uri = clean
        self.root_label = clean
        self.bucket = bucket
        self.prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
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
        def _fetch(rel: str) -> Path:
            dst = staging_dir / rel
            if dst.exists():
                return dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + f".tmp-{threading.get_ident()}")
            self._client.download_file(self.bucket, f"{self.prefix}{rel}", str(tmp))
            tmp.replace(dst)
            return dst

        results: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(_fetch, rel): rel for rel in rel_paths}
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
        resp = self._client.get_object(
            Bucket=self.bucket,
            Key=f"{self.prefix}{rel_path}",
            Range=f"bytes={offset}-{offset + length - 1}",
        )
        body = resp["Body"]
        try:
            return body.read()
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
        import fsspec

        clean = uri.rstrip("/")
        self.root_uri = clean
        self.root_label = clean
        self.max_workers = max(1, max_workers)
        self._fs, root_path = fsspec.url_to_fs(clean, **(storage_options or {}))
        self._root_path = root_path.rstrip("/")
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
        def _fetch(rel: str) -> Path:
            dst = staging_dir / rel
            if dst.exists():
                return dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + f".tmp-{threading.get_ident()}")
            tmp.write_bytes(self._fs.cat_file(self._full_path(rel)))
            tmp.replace(dst)
            return dst

        results: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(_fetch, rel): rel for rel in rel_paths}
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
        return self._fs.cat_file(self._full_path(rel_path), start=offset, end=offset + length)

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
                        "Reading an s3:// source requires either 'boto3' or 's3fs'. "
                        "Install one of them (e.g. `pip install boto3`)."
                    ) from err
            return S3Source(raw, storage_options=storage_options, max_workers=max_workers)
        if is_uri(raw):
            return FsspecSource(raw, storage_options=storage_options, max_workers=max_workers)
        return LocalSource(Path(raw))

    if repo_id is not None:
        token = (storage_options or {}).get("token") or (storage_options or {}).get("hf_token")
        return HubSource(repo_id, revision=revision, token=token, max_workers=max_workers)

    raise ValueError("pass --repo-id or --root")
