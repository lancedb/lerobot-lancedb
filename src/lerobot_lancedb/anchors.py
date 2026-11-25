"""Scalar predicate pushdown and weighted sampling over matched center frames."""

from collections import Counter
from copy import copy
from operator import index

import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler

from .reader import FRAMES_TABLE, _connect, _open_table


class AnchorSelection(Subset):
    """Map-style view. Samplers yield positions in this view; indices address its reader."""

    def __init__(self, dataset, indices, episode_indices, task_indices):
        super().__init__(dataset, indices)
        self.episode_indices = episode_indices
        self.task_indices = task_indices

    def sample_weights(self, *, episode_weights=None, task_weights=None, anchor_weights=None):
        """Multiply supplied factors. Missing groups in an explicit mapping have zero mass."""
        weights = torch.ones(len(self), dtype=torch.double)
        if task_weights is not None:
            task_weights = {_task_id(self.dataset, k): v for k, v in task_weights.items()}
        for ids, totals in ((self.episode_indices, episode_weights), (self.task_indices, task_weights)):
            if totals is not None:
                counts = Counter(ids)
                weights *= torch.tensor([totals.get(k, 0) / counts[k] for k in ids], dtype=torch.double)
        if anchor_weights is not None:
            direct = torch.as_tensor(anchor_weights, dtype=torch.double)
            if direct.shape != weights.shape:
                raise ValueError("Provide one anchor weight per matched anchor")
            weights *= direct
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Weights must be finite and nonnegative")
        return weights

    def weighted_sampler(self, num_samples=None, *, replacement=True, generator=None, **weights):
        return WeightedRandomSampler(
            self.sample_weights(**weights),
            len(self) if num_samples is None else num_samples,
            replacement,
            generator=generator,
        )


def _task_id(dataset, task):
    return index(dataset.meta.tasks.loc[task, "task_index"]) if isinstance(task, str) else index(task)


def select_anchors(dataset, *, where=None, episodes=None, tasks=None, cameras=None):
    predicates = [f"({where})"] if where else []
    for column, ids in (
        ("episode_index", dataset.episodes),
        ("episode_index", episodes),
        ("task_index", tasks),
    ):
        if ids is not None:
            values = sorted({_task_id(dataset, i) if column == "task_index" else index(i) for i in ids})
            predicates.append(f"{column} IN ({','.join(map(str, values))})" if values else "false")
    if cameras is not None:
        cameras = set(cameras)
        unknown = cameras.difference(dataset._camera_keys)
        if unknown:
            raise ValueError(f"Unknown or unselected cameras: {sorted(unknown)}")
        dataset = copy(dataset)  # independent worker handles and decoder caches
        dataset._selected = (set(dataset.features) - set(dataset._camera_keys)) | cameras
        if dataset._include_task:
            dataset._selected.add("task")
        dataset._camera_keys = [k for k in dataset._camera_keys if k in cameras]
        dataset._video_keys = [k for k in dataset._video_keys if k in cameras]
        dataset._depth_keys = [k for k in dataset._depth_keys if k in cameras]
        dataset._video_locator = {k: v for k, v in dataset._video_locator.items() if k in cameras}
        if dataset.delta_indices:
            dataset.delta_indices = {k: v for k, v in dataset.delta_indices.items() if k in dataset._selected}
    options = dataset._storage_options
    table = _open_table(_connect(dataset._db_uri, options), FRAMES_TABLE, options or None)
    matched = (
        table.search()
        .where(" AND ".join(predicates) or "true")
        .select(["episode_index", "frame_index", "timestamp", "task_index"])
        .with_row_id(True)
        .limit(None)
        .to_arrow()
        .sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])
    )
    eps = matched["episode_index"].to_numpy()
    absolute = dataset._ep_from[eps] + matched["frame_index"].to_numpy()
    relative = (
        absolute
        if dataset._subset_index is None
        else np.array([dataset._subset_index[i] for i in absolute], dtype=np.int64)
    )
    return AnchorSelection(dataset, relative, eps, matched["task_index"].to_numpy())
