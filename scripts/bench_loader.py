"""Compare LeRobot and Lance loaders on the same batches.

Run from the checkout with PYTHONPATH=$PWD/src. A uses a seeded global permutation;
B records the actual remote streaming batches and replays their episode/frame pairs.
Dataset conversion, sidecar preparation, correctness checks and imports are untimed.
TTFT starts before iter(DataLoader). Traffic includes startup, warmup and in-flight
prefetch through delivery of the last measured batch. Each row runs in a fresh process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager, redirect_stdout
from importlib.metadata import version
from pathlib import Path

import av
import fsspec
import numpy as np
import torch
from bench_range import BUCKET, range_server
from bench_workloads import PROFILES, source_path
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from torch.utils.data import DataLoader

import lerobot_lancedb
from lerobot_lancedb import LanceDBDataset
from lerobot_lancedb.convert import convert

# Explicit transport, including in spawned DataLoader workers. fsspec's synchronous
# HTTP backend uses requests, already shipped with LeRobot's dataset dependencies.
fsspec.register_implementation("sync-http", "fsspec.implementations.http_sync.HTTPFileSystem")

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("pusht", *PROFILES)
SCALAR_METRICS = (
    "ttft_s",
    "steady_samples_per_s",
    "p50_batch_wait_ms",
    "p95_batch_wait_ms",
    "peak_pss_mb",
    "shm_delta_mb",
)


def check_source():
    if not Path(lerobot_lancedb.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError(f"Run with PYTHONPATH={ROOT / 'src'} to measure this checkout")


def shm_used():
    try:
        return shutil.disk_usage("/dev/shm").used
    except OSError:
        return None


@contextmanager
def memory_peak():
    proc = Path("/proc/self").resolve()
    stop = threading.Event()
    shm_start = shm_used()
    peak = {"peak_pss_mb": None, "peak_rss_fallback_mb": None, "shm_delta_mb": None}

    def sample():
        pss = 0
        complete = True
        pending = [proc]
        while pending:
            directory = pending.pop()
            try:
                for children in directory.glob("task/*/children"):
                    pending.extend(Path("/proc") / pid for pid in children.read_text().split())
                with (directory / "smaps_rollup").open() as f:
                    pss += next(int(s.split()[1]) for s in f if s.startswith("Pss:")) * 1024
            except (FileNotFoundError, ProcessLookupError):
                if directory == proc:
                    complete = False
            except (OSError, StopIteration):
                complete = False
        if complete:
            peak["peak_pss_mb"] = max(peak["peak_pss_mb"] or 0, pss / 1e6)
        else:
            unit = 1 if sys.platform == "darwin" else 1024
            peak["peak_rss_fallback_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 1e6
        shm = shm_used()
        if shm is not None and shm_start is not None:
            peak["shm_delta_mb"] = max(peak["shm_delta_mb"] or 0, (shm - shm_start) / 1e6)

    def monitor():
        while not stop.wait(0.1):
            sample()

    sample()
    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        yield peak
    finally:
        stop.set()
        thread.join()
        sample()


def global_batches(meta, args):
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(meta.total_frames).tolist()
    return [indices[i : i + args.batch_size] for i in range(0, len(indices), args.batch_size)]


def make_loader(dataset, args, batches=None):
    options = {"num_workers": args.num_workers, "generator": torch.Generator().manual_seed(args.seed)}
    if args.num_workers:
        options.update(prefetch_factor=args.prefetch_factor, multiprocessing_context="spawn")
    if batches is None:
        return DataLoader(dataset, batch_size=args.batch_size, **options)
    return DataLoader(dataset, batch_sampler=batches, **options)


def anchors(batch):
    return list(zip(batch["episode_index"].tolist(), batch["frame_index"].tolist(), strict=True))


def indices_for(batches, meta):
    starts = meta.episodes["dataset_from_index"]
    return [[int(starts[ep]) + frame for ep, frame in batch] for batch in batches]


def assert_batch(actual, reference, cameras):
    if actual.keys() != reference.keys():
        raise AssertionError(f"Feature keys differ: {actual.keys() ^ reference.keys()}")
    for key, expected in reference.items():
        if torch.is_tensor(expected):
            # Identical source bytes; allow one RGB quantization level across decoder backends.
            torch.testing.assert_close(
                actual[key],
                expected,
                rtol=0,
                atol=1 if key in cameras else 0,
                msg=lambda reason, key=key: f"{key}: {reason}",
            )
        elif actual[key] != expected:
            raise AssertionError(f"{key}: {actual[key]} != {expected}")


def verify_boundary(dataset, reference, delta, args, *, streaming=False):
    meta = reference.meta
    ep = meta.episodes[0]
    length = ep["dataset_to_index"] - ep["dataset_from_index"]
    edges = {0, length - 1}
    if streaming:
        loader = DataLoader(dataset, batch_size=1)
    else:
        indices = [[ep["dataset_from_index"] + edge] for edge in sorted(edges)]
        loader = make_loader(dataset, args, indices)
    for row in loader:
        frame = int(row["frame_index"][0])
        if frame not in edges:
            continue
        edges.remove(frame)
        expected = torch.utils.data.default_collate([reference[ep["dataset_from_index"] + frame]])
        assert_batch(row, expected, meta.video_keys)
        for key, offsets in (delta or {}).items():
            wanted = [not 0 <= frame + round(t * meta.fps) < length for t in offsets]
            torch.testing.assert_close(row[f"{key}_is_pad"][0], torch.tensor(wanted))
        # Decoders return pixels, not container PTS. Locate the actual source PTS at
        # each requested (clamped) video timestamp independently of either loader.
        for key in meta.video_keys:
            with av.open(str(reference.root / meta.get_video_file_path(0, key))) as container:
                stream = container.streams.video[0]
                offset = ep[f"videos/{key}/from_timestamp"]
                pixels = row[key].reshape(-1, *row[key].shape[-3:])
                for image, delta_s in zip(pixels, (delta or {}).get(key, [0]), strict=True):
                    timestamp = offset + min(
                        max(float(row["timestamp"][0]) + delta_s, 0), (length - 1) / meta.fps
                    )
                    container.seek(int(timestamp / stream.time_base), stream=stream)
                    decoded = next(
                        f for f in container.decode(stream) if float(f.pts * f.time_base) >= timestamp - 1e-4
                    )
                    if abs(float(decoded.pts * decoded.time_base) - timestamp) > 1e-4:
                        raise AssertionError(f"{key}: no source frame at {timestamp}s")
                    if key not in meta.depth_keys:
                        source_rgb = torch.from_numpy(decoded.to_ndarray(format="rgb24")).permute(2, 0, 1)
                        torch.testing.assert_close(image, source_rgb, rtol=0, atol=1)
        if not edges:
            break
    if edges:
        raise AssertionError(f"Missing boundary anchors: {edges}")


def s3_options(endpoint):
    return dict(
        aws_endpoint=endpoint,
        aws_allow_http="true",
        aws_region="us-east-1",
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy",
    )


def run_row(args, source, work, repo_id, suite, mode, captured=None):
    check_source()
    torch.set_num_threads(args.torch_threads)
    meta = LeRobotDatasetMetadata(repo_id, root=source)
    delta = args.delta_timestamps
    if delta is None:
        delta = {
            k: [-1 / meta.fps, 0.0, 1 / meta.fps] for k in [*meta.video_keys, "action"] if k in meta.features
        }
    common = dict(repo_id=repo_id, delta_timestamps=delta or None, return_uint8=True)
    pool_mode = mode == "episode_pool"
    lookahead = mode.startswith("hybrid_lookahead_")
    iterable = mode == "streaming" or pool_mode
    with range_server(source, args.rtt_ms) as (endpoint, traffic), ExitStack() as resources:

        def parquet():
            return LeRobotDataset(root=source, video_backend="torchcodec", **common)

        def open_dataset(**overrides):
            if mode == "parquet":
                return parquet()
            if mode == "streaming":
                return StreamingLeRobotDataset(
                    root=source,
                    data_root=f"sync-{endpoint}/{BUCKET}",
                    video_backend="torchcodec",
                    seed=args.seed,
                    repeat=overrides.pop("repeat", True),
                    **overrides,
                    episode_pool_size=args.episode_pool_size,
                    **common,
                )
            if pool_mode:
                return lerobot_lancedb.LanceDBEpisodePoolDataset(
                    root=work / "hybrid", storage_options=s3_options(endpoint),
                    use_mini_mp4=not args.no_mini_mp4, seed=args.seed,
                    episode_pool_size=args.episode_pool_size, batch_size=args.batch_size,
                    **overrides, **common,
                )
            return LanceDBDataset(
                root=work / ("hybrid" if lookahead else mode),
                storage_options=s3_options(endpoint) if mode == "hybrid" or lookahead else None,
                use_mini_mp4=not args.no_mini_mp4,
                **common,
            )

        dataset = open_dataset()
        batches = global_batches(meta, args) if suite == "A" else None
        if captured is not None and not pool_mode:
            batches = indices_for(captured, meta)
        steps = args.warmup_steps + args.steps
        if batches is not None and len(batches) < steps:
            raise ValueError(f"Requested {steps} batches; only {len(batches)} available")
        loader = make_loader(dataset, args, batches)
        waits, observed = [], []
        before = traffic.snapshot()
        with memory_peak() as peak:
            start = time.perf_counter()
            if lookahead:
                depth = int(mode.removeprefix("hybrid_lookahead_"))
                iterator = resources.enter_context(dataset.lookahead_loader(
                    batch_sampler=batches, num_workers=args.num_workers, lookahead_batches=depth,
                    generator=torch.Generator().manual_seed(args.seed),
                    **({"prefetch_factor": args.prefetch_factor, "multiprocessing_context": "spawn"} if args.num_workers else {}),
                ))
            else:
                iterator = iter(loader)
            for step in range(steps):
                ready = time.perf_counter()
                batch = next(iterator)
                now = time.perf_counter()
                if step == 0:
                    ttft = now - start
                if step == args.warmup_steps:
                    steady_start = ready
                if step >= args.warmup_steps:
                    waits.append((now - ready) * 1000)
                observed.append(anchors(batch))
            after = traffic.snapshot()
        count = sum(map(len, observed[args.warmup_steps :]))
        io = {k: {m: v - before[k][m] for m, v in stats.items()} for k, stats in after.items()}
        if pool_mode and observed != captured[:steps]:
            raise AssertionError("Episode pool and streaming delivered different anchors")
        if batches is not None and indices_for(observed, meta) != batches[:steps]:
            raise AssertionError(f"{suite} {mode}: delivered batches differ from sampler")
        # Give replay workers the same number of prefetch slots beyond the timed window.
        replay = list(observed)
        if mode == "streaming":
            replay.extend(anchors(next(iterator)) for _ in range(args.num_workers * args.prefetch_factor))
        resources.close()
        del iterator, loader, batch
        if args.verify_correctness:
            reference = parquet()
            if mode == "streaming":
                boundary_dataset = open_dataset(episodes=[0], repeat=False)
            elif pool_mode:
                boundary_dataset = open_dataset(episodes=[0])
            else:
                boundary_dataset = dataset
            verify_boundary(boundary_dataset, reference, delta, args, streaming=iterable)
            # Restore at a consumed batch boundary and compare the remaining anchors AND
            # feature values against random access to the source on a fresh iterator.
            skip = min(1, steps - 1)
            restored = open_dataset()
            if iterable:
                restored.load_state_dict(
                    {"epoch": 0, "offset": skip * args.batch_size, "batch_size": args.batch_size}
                )
                resume_batches = None
            else:
                resume_batches = batches[skip:steps]
            it = iter(make_loader(restored, args, resume_batches))
            for offsets in indices_for(observed[skip:], meta):
                ref = torch.utils.data.default_collate([reference[i] for i in offsets])
                assert_batch(next(it), ref, meta.video_keys)
            del it
        row = dict(
            suite=suite,
            mode=mode,
            ttft_s=ttft,
            steady_samples_per_s=count / (now - steady_start),
            p50_batch_wait_ms=float(np.percentile(waits, 50)),
            p95_batch_wait_ms=float(np.percentile(waits, 95)),
            traffic=io,
            **peak,
            measured_samples=count,
            delta_timestamps=delta,
            anchor_batches=[[list(anchor) for anchor in batch] for batch in observed],
        )
        return row, replay


def prepare(args, workload, work):
    source = source_path(workload, work, args.dataset_path, args.repo_id)
    repo_id = args.repo_id or ("lerobot/pusht" if workload == "pusht" else f"local/{workload}")
    if "A" in args.suites:
        convert(work / "local", root=source, video_mode="external")
    with range_server(source, 0) as (endpoint, _):
        convert(
            work / "hybrid",
            root=source,
            video_mode="external",
            video_base_uri=f"s3://{BUCKET}/videos",
            source_storage_options=s3_options(endpoint),
        )
    # Warm the backing files for all modes and identify the actual input content for A/B.
    digest = hashlib.sha256()
    for root in [source, work / "local", work / "hybrid"]:
        if root == source:
            paths = (p for folder in ("meta", "data", "videos") for p in sorted((root / folder).rglob("*")))
        else:
            paths = root.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            if root == source:
                digest.update(str(path.relative_to(root)).encode() + b"\0")
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    if root == source:
                        digest.update(chunk)
    return source, repo_id, digest.hexdigest()


def render(rows):
    print(
        "| Workload | Suite | Loader | TTFT s | Samples/s | p50 ms | p95 ms | GETs | MB | Peak PSS MB | Δ shm MB |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        io = r["traffic"]["total"]
        values = [r[k] for k in SCALAR_METRICS[:4]] + [
            io["gets"],
            io["bytes"] / 1e6,
            r["peak_pss_mb"],
            r["shm_delta_mb"],
        ]
        print(
            "|",
            " | ".join(
                [
                    r["workload"],
                    r["suite"],
                    r["mode"],
                    *("n/a" if v is None else str(v) if isinstance(v, int) else f"{v:.2f}" for v in values),
                ]
            ),
            "|",
        )


def compare(base, candidate):
    def indexed(report):
        return {(r["workload"], r["suite"], r["mode"]): r for r in report["rows"]}

    left, right = indexed(base), indexed(candidate)
    if left.keys() != right.keys() or base["parameters"] != candidate["parameters"]:
        raise ValueError("Unmatched rows or runtime parameters")
    print(f"{base['commit']} → {candidate['commit']} (candidate minus baseline)")
    print(
        "| Workload | Suite | Loader | Δ TTFT s | Δ samples/s | Δ p50 ms | Δ p95 ms | Δ PSS MB | Δ shm MB | Δ GETs | Δ bytes |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, a in left.items():
        b = right[key]
        if any(a[k] != b[k] for k in ("source_sha256", "anchor_batches", "delta_timestamps")):
            raise ValueError(f"Unmatched data or anchors for {key}")
        diff = [f"{b[k] - a[k]:+.2f}" if None not in (a[k], b[k]) else "n/a" for k in SCALAR_METRICS]
        diff.extend(str(b["traffic"]["total"][k] - a["traffic"]["total"][k]) for k in ("gets", "bytes"))
        print("|", " | ".join([*key, *diff]), "|")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    aliases = {"droid": "droid-short-gop", "720p": "long-gop-720p"}
    parser.add_argument(
        "--workloads",
        "--workload",
        nargs="+",
        type=lambda w: aliases.get(w, w),
        choices=WORKLOADS,
        default=list(WORKLOADS),
    )
    parser.add_argument(
        "--suites", "--suite", nargs="+", type=str.upper, choices=["A", "B"], default=["A", "B"]
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--repo-id", help="Optional cached Hub dataset or identity of --dataset-path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--episode-pool-size", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--delta-timestamps", type=json.loads, help="JSON feature to seconds map; {} disables windows"
    )
    parser.add_argument("--rtt-ms", type=float, default=20.0)
    parser.add_argument("--no-mini-mp4", action="store_true", help="Use the sparse RGB decoder for indexed blobs")
    parser.add_argument("--lookahead-batches", type=int, help="Add a matched Suite A hybrid lookahead row")
    parser.add_argument("--episode-pool", action="store_true", help="Add a matched Suite B Lance episode pool row")
    parser.add_argument("--verify-correctness", action="store_true")
    parser.add_argument("--output-json", "--output", type=Path)
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="REPORT",
        help="Baseline for this run, or two existing reports",
    )
    parser.add_argument("--markdown", type=Path, help="Write the comparison table to this file")
    args = parser.parse_args(argv)

    def comparison(base, candidate):
        if args.markdown:
            with args.markdown.open("w") as stream, redirect_stdout(stream):
                compare(base, candidate)
        else:
            compare(base, candidate)

    if args.markdown and not args.compare:
        parser.error("--markdown requires --compare")
    if args.compare and len(args.compare) == 2:
        comparison(*(json.loads(p.read_text()) for p in args.compare))
        return
    if args.compare and len(args.compare) != 1:
        parser.error("--compare takes one baseline or two existing reports")
    check_source()
    if (
        min(args.batch_size, args.steps, args.prefetch_factor, args.episode_pool_size, args.torch_threads) < 1
        or min(args.num_workers, args.rtt_ms, args.warmup_steps) < 0
    ):
        parser.error("Counts must be positive; workers, RTT and warmup may also be zero")
    if (args.dataset_path or args.repo_id) and len(args.workloads) != 1:
        parser.error("Select one workload when overriding its source")
    parameters = {
        k: v
        for k, v in vars(args).items()
        if k
        not in {
            "output_json",
            "markdown",
            "compare",
            "dataset_path",
            "repo_id",
            "verify_correctness",
            "no_mini_mp4",
            "lookahead_batches",
            "episode_pool",
            "workloads",
            "suites",
        }
    }
    report = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        parameters=parameters,
        verify_correctness=args.verify_correctness,
        rows=[],
        runtime={
            "use_mini_mp4": not args.no_mini_mp4,
            "harness_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "scripts").glob("bench_*.py"))
            },
            "python": sys.version,
            "platform": platform.platform(),
            "versions": {
                k: version(k) for k in ("lerobot", "lancedb", "torch", "torchcodec", "av", "fsspec")
            },
            "threads_env": {
                k: os.getenv(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "LANCE_IO_THREADS")
            },
        },
    )
    for workload in args.workloads:
        with tempfile.TemporaryDirectory(prefix="lerobot-bench-") as tmp:
            work = Path(tmp)
            source, repo_id, digest = prepare(args, workload, work)
            for suite in args.suites:
                captured = None
                modes = ["parquet", "local", "hybrid"] if suite == "A" else ["streaming", "hybrid"]
                if suite == "A" and args.lookahead_batches is not None:
                    modes.extend(f"hybrid_lookahead_{k}" for k in dict.fromkeys((0, args.lookahead_batches)))
                if suite == "B" and args.episode_pool:
                    modes.append("episode_pool")
                for mode in modes:
                    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
                        row, replay = pool.submit(
                            run_row, args, source, work, repo_id, suite, mode, captured
                        ).result()
                    if mode == "streaming":
                        captured = replay
                    row.update(workload=workload, repo_id=repo_id, source_sha256=digest)
                    report["rows"].append(row)
    render(report["rows"])
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    if args.compare:
        comparison(json.loads(args.compare[0].read_text()), report)


if __name__ == "__main__":
    main()
