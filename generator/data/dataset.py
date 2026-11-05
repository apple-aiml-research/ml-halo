#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
MotionDataset: PyTorch Dataset for loading vision-language-motion data.

Loads parquet files with motion data and extracts video frames for training
a vision-language to motion generator model.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
import json
import time
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from . import remote
from .remote import is_s3, local_dir


# Only these three parquet columns are read by the training path: __getitem__
# feeds them to data/traj_extract.py (joint_pos + pelvis pos/quat). Everything
# else in the file (action, info.*,
# timestamp, frame_index, episode_index, index, task_index) is never touched —
# `language` comes from meta (_ep_by_index), not the parquet.
_PARQUET_TRAIN_COLS = (
    "observation.joint_pos",   # [L, 29]
    "observation.body_pos",    # [L, B, 3]  only body 0 (pelvis) is used
    "observation.body_quat",   # [L, B, 4]  only body 0 (pelvis) is used
)


def _arrow_is_list(t) -> bool:
    """True for any Arrow list flavour (list / large_list / fixed_size_list)."""
    return bool(
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_fixed_size_list(t)
    )


def _arrow_to_numpy(col) -> np.ndarray:
    """A (possibly nested-list) Arrow column -> one dense float32 ndarray.

    list<double> -> [L, n]; list<list<double>> -> [L, n, m]. Flattening the
    Arrow buffers avoids materializing the intermediate Python objects that
    to_pylist() would create.
    """
    arr = col.combine_chunks()
    n_rows = len(arr)
    dims = []
    while _arrow_is_list(arr.type):
        inner = arr.flatten()
        dims.append(len(inner) // len(arr))
        arr = inner
    flat = arr.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return flat.reshape((n_rows, *dims))


def read_episode_parquet(path) -> Dict[str, np.ndarray]:
    """Read the training columns of an episode parquet as dense float32 arrays."""
    return _table_to_numpy_columns(pq.read_table(path))


def _table_to_numpy_columns(table) -> Dict[str, np.ndarray]:
    """Dense float32 arrays for the columns the trainer reads.

    About 1 KB per frame, versus ~14 KB for the nested Python lists that
    to_pylist() would build, and no per-sample list->array conversion in
    __getitem__.
    """
    out: Dict[str, np.ndarray] = {}
    for name in _PARQUET_TRAIN_COLS:
        a = _arrow_to_numpy(table[name])
        # Only body index 0 (pelvis) is read; keep the body axis for the
        # `_gather(...)[:, 0]` indexing in traj_extract.
        if a.ndim == 3:
            a = np.ascontiguousarray(a[:, :1])
        out[name] = a
    return out


@dataclass
class EpisodeInfo:
    """Episode metadata."""
    episode_index: int
    tasks: List[str]
    length: int


class MotionDataset(Dataset):
    """
    Dataset for motion generation from vision and language.

    Each sample provides:
    - Input: First frame RGB (resized to image_size) + language command
    - Output: 30-frame motion trajectory (1 second at 30fps)

    Output format per frame: [joint_pos (29D) + pelvis_pos (3D) + pelvis_quat (4D)] = 36D
    Total output shape: [30, 36]
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        trajectory_length: int = 30,
        transform=None,
        normalize_motion: bool = False,
        cache_frames: bool = False,
        cache_dir: str = "data/cache",
        streaming: bool = False,
        max_episodes: int = 0,
        prev_chunk_frames: int = 15,
        cold_start_augment: bool = False,
        fit_normalizer: bool = True,
    ):
        """
        Initialize MotionDataset.

        Args:
            data_dir: Local dataset directory, or an s3:// URI. An S3 dataset is
                copied into its cache_dir copy: all of it up front, then read like
                a local dataset, or (streaming) only meta/, with ChunkedDataLoader
                fetching the episodes chunk by chunk.
            split: Data split (currently unused, full dataset used)
            image_size: Target image size (H, W)
            trajectory_length: Number of frames in output trajectory (default 30 = 1 sec)
            transform: Optional image transform
            normalize_motion: Whether to normalize motion trajectories
            cache_frames: Preload all video frames and parquet data into RAM
            cache_dir: Local copy root for an S3 dataset (see data/remote.py).
                Unused for a local dataset.
            streaming: S3 dataset only: leave the episodes to ChunkedDataLoader
                instead of downloading them all up front.
            max_episodes: When >0, use only the first N episodes (smoke tests).
            fit_normalizer: Fit the motion normalizer here (when
                normalize_motion). False leaves it unset for the caller to supply
                via set_normalizer() (distributed ranks receive rank 0's fit).
        """
        # The dataset's S3 URI, or None for a local dataset; local_dir is where
        # its files are read from (data_dir itself, or the S3 dataset's cache).
        self.s3_uri = str(data_dir).rstrip("/") if is_s3(data_dir) else None
        self.local_dir = local_dir(data_dir, cache_dir)
        self.streaming = streaming
        self.split = split
        self.image_size = image_size
        self.trajectory_length = trajectory_length
        self.transform = transform
        self.normalize_motion = normalize_motion
        self.cache_frames = cache_frames
        self.prev_chunk_frames = prev_chunk_frames
        self.cold_start_augment = cold_start_augment

        if self.s3_uri:
            self._download_metadata()

        # Load metadata
        self.info = self._load_info()
        self.episodes = self._load_episodes()
        if max_episodes > 0:
            self.episodes = self.episodes[:max_episodes]
        self.tasks = self._load_tasks()

        # Episodes carry their global episode_index. The staged render assigns
        # episode_index = shard * 1000 + local, which is not contiguous (and a gated
        # subset has arbitrary holes). Look an episode up by that value, not by list
        # position: self.episodes[val] raises IndexError past the first gap.
        self._ep_by_index = {ep.episode_index: ep for ep in self.episodes}

        # Build sample index: (episode_idx, start_frame_idx)
        self.samples = self._build_sample_index()

        # Caches for fast data loading
        self._parquet_cache = {}  # {episode_idx: dict}
        self._episode_frames = {} # {episode_idx: [N, 3, H, W] uint8 tensor}

        # A non-streamed S3 dataset is copied whole, then read like a local one.
        if self.s3_uri and not streaming:
            self.download_episodes([ep.episode_index for ep in self.episodes])

        # Extract video frames to .pt files (a streamed dataset is extracted per
        # chunk by ChunkedDataLoader).
        if not streaming:
            self._ensure_frames_extracted()

        if cache_frames:
            # Load everything into RAM (frames + parquets)
            self._load_frames_into_memory()
            self._preload_parquets()

        # Motion normalizer (optional). Fitted from this run's episodes every time
        # and never written to disk: it travels with the model checkpoint, so it
        # cannot go stale against a changed dataset or max_episodes.
        self.normalizer = None
        if normalize_motion and fit_normalizer:
            self._fit_normalizer()

    def _download_metadata(self):
        """Download the S3 dataset's meta/ directory into local_dir."""
        src = f"{self.s3_uri}/meta"
        print(f"Downloading dataset metadata from {src}/...")
        remote.download_prefix(src, self.local_dir / "meta")
        # Create directories for temporary per-chunk files
        (self.local_dir / "videos" / "observation.images.fpv_cam").mkdir(parents=True, exist_ok=True)
        (self.local_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.local_dir / "frames" / "observation.images.fpv_cam").mkdir(parents=True, exist_ok=True)
        print("Metadata download complete.")

    def download_episodes(self, episode_ids: List[int]):
        """Fetch the files training reads for these episodes from S3 into
        local_dir: the parquet and the fpv video. Files already there are kept,
        so an interrupted download resumes."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=remote.S3_WORKERS) as pool:
            list(pool.map(self._download_episode, episode_ids))

    def _download_episode(self, ep_idx: int):
        name = f"episode_{ep_idx:06d}"
        for rel in (f"data/{name}.parquet",
                    f"videos/observation.images.fpv_cam/{name}.mp4"):
            local = self.local_dir / rel
            if not local.exists():
                remote.download(f"{self.s3_uri}/{rel}", local)

    def _load_info(self) -> Dict:
        """Load dataset info from info.json."""
        info_path = self.local_dir / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        with open(info_path) as f:
            return json.load(f)

    def _load_episodes(self) -> List[EpisodeInfo]:
        """Load episode metadata from episodes.jsonl."""
        episodes_path = self.local_dir / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"episodes.jsonl not found at {episodes_path}")

        episodes = []
        with open(episodes_path) as f:
            for line in f:
                data = json.loads(line)
                episodes.append(EpisodeInfo(
                    episode_index=data["episode_index"],
                    tasks=data["tasks"],
                    length=data["length"],
                ))
        return episodes

    def _load_tasks(self) -> Dict[int, str]:
        """Load task definitions from tasks.jsonl."""
        tasks_path = self.local_dir / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"tasks.jsonl not found at {tasks_path}")

        tasks = {}
        with open(tasks_path) as f:
            for line in f:
                task = json.loads(line)
                tasks[task["task_index"]] = task["task"]
        return tasks

    def _build_sample_index(self) -> List[Tuple[int, int, int]]:
        """
        Build index of valid samples as (episode_idx, start_frame, cold_k).

        Every frame in every episode is a valid sample (cold_k = -1).
        Trajectories that extend past the episode end are padded with the last
        frame's pose.

        When cold_start_augment is set, each episode start gets prev_chunk_frames
        extra samples (cold_k = 0 .. prev_chunk_frames-1). A cold-start sample
        blends a static hold of frame 0 with the future plan: the first
        (prev_chunk_frames - cold_k) frames hold frame 0, then the future plan
        follows. The normal start_frame=0 sample is the cold_k = prev_chunk_frames
        case (no hold), so we only add 0..prev_chunk_frames-1 here.
        """
        samples = []
        for ep in self.episodes:
            ep_idx = ep.episode_index
            ep_len = ep.length
            if ep_len >= 1:
                for start_frame in range(ep_len):
                    samples.append((ep_idx, start_frame, -1))
                if self.cold_start_augment:
                    for cold_k in range(self.prev_chunk_frames):
                        samples.append((ep_idx, 0, cold_k))

        return samples

    def _fit_normalizer(self):
        """Fit motion normalizer on dataset using sampled trajectories.

        For a streamed S3 dataset, first syncs every parquet into local_dir.
        """
        from .transforms import MotionNormalizer

        # Collect samples for fitting (every 10th to avoid redundant overlapping windows)
        fit_samples = self.samples[::10]

        print("Fitting motion normalizer...")
        if self.streaming:
            remote.download_prefix(f"{self.s3_uri}/data", self.local_dir / "data")

        all_trajectories = []
        for ep_idx, start_frame, cold_k in fit_samples:
            if ep_idx not in self._parquet_cache:
                parquet_path = self.local_dir / "data" / f"episode_{ep_idx:06d}.parquet"
                self._parquet_cache[ep_idx] = self._read_parquet(parquet_path)
            traj = self._extract_sample_trajectory(self._parquet_cache[ep_idx], start_frame, cold_k)
            all_trajectories.append(traj.numpy())

        all_trajectories = np.stack(all_trajectories)
        self.normalizer = MotionNormalizer()
        self.normalizer.fit(all_trajectories)

        # Free parquet cache (files stay on disk for other ranks to read)
        self._parquet_cache.clear()

        print("Motion normalizer fitted.")

    def _read_parquet(self, path: Path) -> Dict[str, np.ndarray]:
        return read_episode_parquet(path)

    def _extract_sample_trajectory(
        self,
        data: Any,
        start_frame: int,
        cold_k: int,
    ) -> torch.Tensor:
        """
        Extract the target trajectory for a sample.

        For normal samples (cold_k < 0) this is just the trajectory window
        starting at start_frame. For cold-start augmented samples (cold_k >= 0,
        start_frame == 0) it blends a static hold of frame 0 with the future
        plan: the first (prev_chunk_frames - cold_k) frames repeat frame 0, then
        the future plan follows — all expressed relative to frame 0. The first
        prev_chunk_frames of the result are exactly the prev_chunk the model is
        inpainted on, so no separate prev_chunk tensor is needed.

        Delegates to data/traj_extract.py.
        """
        from .traj_extract import extract_sample_trajectory_vectorized
        return extract_sample_trajectory_vectorized(
            data, start_frame, cold_k, self.trajectory_length, self.prev_chunk_frames
        )

    def _ensure_frames_extracted(self):
        """Extract video frames to .pt files if not already done.

        Checks each episode for a pre-extracted .pt file. If missing,
        decodes the MP4, resizes to image_size, and saves as uint8
        tensor [N, 3, H, W].  Uses multiprocessing to parallelize.
        """
        frames_dir = self.local_dir / "frames" / "observation.images.fpv_cam"
        frames_dir.mkdir(parents=True, exist_ok=True)

        episode_indices = sorted(set(ep_idx for ep_idx, *_ in self.samples))
        needs_extraction = []
        for ep_idx in episode_indices:
            pt_path = frames_dir / f"episode_{ep_idx:06d}.pt"
            if not pt_path.exists():
                needs_extraction.append(ep_idx)

        if not needs_extraction:
            return

        import multiprocessing as mp

        num_workers = min(usable_cpus(), len(needs_extraction))
        print(
            f"Extracting frames for {len(needs_extraction)} episodes "
            f"({num_workers} workers)..."
        )

        args = [
            (ep_idx, self.local_dir, self.image_size)
            for ep_idx in needs_extraction
        ]
        with mp.Pool(num_workers) as pool:
            for ep_idx, n_frames in pool.imap_unordered(
                _extract_episode_frames, args
            ):
                print(f"  Extracted episode {ep_idx}: {n_frames} frames")

        print("Frame extraction complete.")

    def _load_frames_into_memory(self):
        """Load all pre-extracted .pt frame files into memory."""
        frames_dir = self.local_dir / "frames" / "observation.images.fpv_cam"
        episode_indices = sorted(set(ep_idx for ep_idx, *_ in self.samples))
        print(f"Loading frames into memory for {len(episode_indices)} episodes...")
        for ep_idx in episode_indices:
            pt_path = frames_dir / f"episode_{ep_idx:06d}.pt"
            self._episode_frames[ep_idx] = torch.load(
                pt_path, map_location="cpu", weights_only=True,
            )
        print(f"Frames loaded: {len(self._episode_frames)} episodes")

    def _preload_parquets(self):
        """Load all parquet files into memory."""
        episode_indices = sorted(set(ep_idx for ep_idx, *_ in self.samples))
        print(f"Loading parquets for {len(episode_indices)} episodes...")
        for ep_idx in episode_indices:
            if ep_idx not in self._parquet_cache:
                parquet_path = self.local_dir / "data" / f"episode_{ep_idx:06d}.parquet"
                self._parquet_cache[ep_idx] = self._read_parquet(parquet_path)
        print(f"Parquets ready: {len(self._parquet_cache)} episodes")

    def _load_video_frame(
        self,
        episode_idx: int,
        frame_idx: int
    ) -> torch.Tensor:
        """
        Load single RGB frame from video.

        Reads the in-memory episode tensor, else the episode's extracted .pt
        file (every episode is extracted before training reads it).

        Args:
            episode_idx: Episode index
            frame_idx: Frame index within episode

        Returns:
            torch.Tensor: [3, H, W] RGB frame in range [0, 1]
        """
        # In-memory episode tensor (cache_frames, or the active streamed chunk)
        if episode_idx in self._episode_frames:
            return self._episode_frames[episode_idx][frame_idx].float() / 255.0

        pt_path = (
            self.local_dir / "frames" / "observation.images.fpv_cam"
            / f"episode_{episode_idx:06d}.pt"
        )
        episode_tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
        if self.cache_frames:
            self._episode_frames[episode_idx] = episode_tensor
        return episode_tensor[frame_idx].float() / 255.0

    def _read_chunk_from_disk(self, episode_ids: List[int]):
        """Load episodes' frames + parquets from disk into fresh dicts.

        Does not touch the active caches, so this is safe to call from a
        background thread while training reads the current chunk. Returns
        (frames_dict, parquet_dict) for the caller to swap in via
        activate_chunk().
        """
        frames: Dict[int, Any] = {}
        parquets: Dict[int, Any] = {}
        for ep_idx in episode_ids:
            pt_path = (
                self.local_dir / "frames" / "observation.images.fpv_cam"
                / f"episode_{ep_idx:06d}.pt"
            )
            if pt_path.exists():
                frames[ep_idx] = torch.load(
                    pt_path, map_location="cpu", weights_only=True,
                )
            parquet_path = self.local_dir / "data" / f"episode_{ep_idx:06d}.parquet"
            if parquet_path.exists():
                parquets[ep_idx] = self._read_parquet(parquet_path)
        return frames, parquets

    def activate_chunk(self, frames: Dict[int, Any], parquets: Dict[int, Any]):
        """Swap in a preloaded chunk's caches, dropping the previous chunk.

        The reassignment is atomic from __getitem__'s perspective (it reads
        whatever self._episode_frames/_parquet_cache currently point at), so the
        loader can swap chunks between batches without tearing.
        """
        self._episode_frames = frames
        self._parquet_cache = parquets

    def unload_chunk(self):
        """Free all cached episode data."""
        self._episode_frames.clear()
        self._parquet_cache.clear()
        import gc
        gc.collect()
        # On Linux, force glibc to return freed pages to the OS
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dict with:
                - rgb: [3, H, W] RGB frame
                - language: str task description
                - trajectory: [30, 36] motion trajectory
                - episode_idx: int
                - frame_idx: int
        """
        ep_idx, start_frame, cold_k = self.samples[idx]

        # Load parquet data (from cache or disk)
        if ep_idx in self._parquet_cache:
            df = self._parquet_cache[ep_idx]
        else:
            parquet_path = self.local_dir / "data" / f"episode_{ep_idx:06d}.parquet"
            df = self._read_parquet(parquet_path)

        # Extract trajectory (normal window, or cold-start blend when cold_k >= 0)
        trajectory = self._extract_sample_trajectory(df, start_frame, cold_k)

        # Normalize trajectory if enabled
        if self.normalize_motion and self.normalizer is not None:
            trajectory = self.normalizer.normalize(trajectory)

        # RGB frame at start_frame (the first frame of the trajectory)
        rgb_frame = self._load_video_frame(ep_idx, start_frame)
        if self.transform:
            rgb_frame = self.transform(rgb_frame)

        # Get language command
        language_cmd = self._ep_by_index[ep_idx].tasks[0]

        return {
            "rgb": rgb_frame,                    # [3, H, W]
            "language": language_cmd,            # str
            "trajectory": trajectory,            # [T, 36]
            "episode_idx": ep_idx,
            "frame_idx": start_frame,
        }

    def get_normalizer(self):
        """Get the motion normalizer (for saving/loading)."""
        return self.normalizer

    def set_normalizer(self, normalizer):
        """Set the motion normalizer."""
        self.normalizer = normalizer
        self.normalize_motion = normalizer is not None


def usable_cpus() -> int:
    """CPUs this process may run on: in a container ``os.cpu_count()`` is the host's (e.g. 192 on a 21-CPU pod)."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def _extract_episode_frames(args):
    """Worker function for parallel frame extraction (module-level for pickling)."""
    from .video_reader import read_frames

    torch.set_num_threads(1)  # one process per CPU already; torch's own threads would oversubscribe them

    ep_idx, data_dir, image_size = args
    data_dir = Path(data_dir)

    video_path = (
        data_dir / "videos" / "observation.images.fpv_cam"
        / f"episode_{ep_idx:06d}.mp4"
    )
    frames = read_frames(video_path, image_size)  # [N, 3, H, W] uint8

    pt_path = (
        data_dir / "frames" / "observation.images.fpv_cam"
        / f"episode_{ep_idx:06d}.pt"
    )
    torch.save(frames, pt_path)
    return ep_idx, len(frames)


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for MotionDataset.

    Handles string language commands separately from tensors.
    """
    language = [item["language"] for item in batch]
    trajectory = torch.stack([item["trajectory"] for item in batch])
    episode_idx = torch.tensor([item["episode_idx"] for item in batch])
    frame_idx = torch.tensor([item["frame_idx"] for item in batch])

    return {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "language": language,
        "trajectory": trajectory,
        "episode_idx": episode_idx,
        "frame_idx": frame_idx,
    }


def _share_normalizer(dataset: MotionDataset, timeout_s: int = 4 * 3600):
    """Give every rank global rank 0's normalizer fit, over the TCPStore.

    Polls the store instead of using an NCCL collective: the fit (in streaming
    mode, a whole-dataset parquet download) can outlast the 60 min NCCL timeout.
    The float32 stats round-trip through JSON exactly.
    """
    import torch.distributed as dist
    from .transforms import MotionNormalizer

    store = dist.distributed_c10d._get_default_store()
    key = "halo_motion_normalizer"
    if dist.get_rank() == 0:
        stats = dataset.get_normalizer().state_dict()
        store.set(key, json.dumps({k: v.tolist() for k, v in stats.items()}))
        return
    t0 = time.time()
    while not store.check([key]):
        time.sleep(5)
        if time.time() - t0 > timeout_s:
            raise RuntimeError(
                f"[Rank {dist.get_rank()}] timed out waiting for rank 0's normalizer fit")
    stats = json.loads(store.get(key))
    normalizer = MotionNormalizer()
    normalizer.load_state_dict(
        {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()})
    dataset.set_normalizer(normalizer)


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.9,
    image_size: Tuple[int, int] = (224, 224),
    trajectory_length: int = 30,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize_motion: bool = False,
    cache_frames: bool = False,
    distributed: bool = False,
    cache_dir: str = "data/cache",
    streaming_chunk_size: int = 0,
    max_episodes: int = 0,
    prev_chunk_frames: int = 15,
    cold_start_augment: bool = False,
):
    """
    Create training and validation data loaders.

    A local data_dir gets a standard DataLoader (plus a validation loader). An
    s3:// data_dir is copied under cache_dir: whole, then loaded like a local
    dataset (streaming_chunk_size 0), or streamed by ChunkedDataLoader,
    streaming_chunk_size episodes at a time.

    Args:
        data_dir: Local dataset directory, or an s3:// URI
        batch_size: Batch size (per GPU when distributed)
        train_split: Fraction of data for training
        image_size: Target image size (H, W)
        trajectory_length: Trajectory length in frames
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        normalize_motion: Whether to normalize motion data
        cache_frames: Preload all data into RAM
        distributed: Use DistributedSampler for multi-GPU training
        cache_dir: Local copy root for an S3 dataset
        streaming_chunk_size: Episodes per streamed chunk (S3 dataset only);
            0 = no streaming

    Returns:
        Tuple of (train_loader, val_loader)
    """
    import torch.distributed as dist
    from torch.utils.data import DataLoader, random_split
    from torch.utils.data.distributed import DistributedSampler
    from .transforms import get_vlm_image_transform

    streaming = streaming_chunk_size > 0
    if streaming and not is_s3(data_dir):
        raise ValueError(
            f"data.streaming_chunk_size={streaming_chunk_size} streams from S3, but "
            f"data_dir={data_dir!r} is local; use streaming_chunk_size: 0.")

    transform = get_vlm_image_transform(image_size)
    root = local_dir(data_dir, cache_dir)

    dataset_kwargs = dict(
        data_dir=data_dir,
        image_size=image_size,
        trajectory_length=trajectory_length,
        transform=transform,
        normalize_motion=normalize_motion,
        cache_frames=cache_frames,
        cache_dir=cache_dir,
        streaming=streaming,
        max_episodes=max_episodes,
        prev_chunk_frames=prev_chunk_frames,
        cold_start_augment=cold_start_augment,
    )

    # When distributed, each NODE's local-rank-0 prepares that node's local
    # disk first (downloads an S3 dataset, or its metadata when streaming, and
    # extracts frames) and
    # signals via a node-local sentinel; the node's other ranks then proceed.
    # Disk is node-local on multi-node, so we gate on LOCAL_RANK (not global
    # rank). We use a file sentinel instead of dist.barrier() because dataset
    # prep can exceed the NCCL barrier timeout. Only global rank 0 fits the
    # normalizer; the other ranks receive its fit (_share_normalizer), so every
    # rank normalizes identically.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        import time as _time
        is_rank0 = dist.get_rank() == 0
        sentinel = root / ".dataset_ready"
        if local_rank == 0:
            sentinel.unlink(missing_ok=True)
            full_dataset = MotionDataset(**dataset_kwargs, fit_normalizer=is_rank0)
            sentinel.touch()
        else:
            # Wait for this node's local-rank-0 to finish preparation
            while not sentinel.exists():
                _time.sleep(5)
            full_dataset = MotionDataset(**dataset_kwargs, fit_normalizer=False)
        if normalize_motion:
            _share_normalizer(full_dataset)
        # Global CPU barrier so all ranks across all nodes are in sync
        dist.barrier()
        # Drop the parquets the normalizer fit synced into a streamed dataset's
        # cache; training streams them back chunk by chunk.
        if local_rank == 0 and streaming:
            for f in (root / "data").glob("*.parquet"):
                f.unlink(missing_ok=True)
    else:
        full_dataset = MotionDataset(**dataset_kwargs)

    # Split into train/val
    total_samples = len(full_dataset)
    train_size = int(total_samples * train_split)
    val_size = total_samples - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Streamed S3 dataset: one chunk of episodes in RAM at a time
    if streaming:
        from .chunked_loader import ChunkedDataLoader

        train_loader = ChunkedDataLoader(
            underlying_dataset=full_dataset,
            sample_indices=list(train_dataset.indices),
            batch_size=batch_size,
            chunk_episodes=streaming_chunk_size,
            pin_memory=pin_memory,
            distributed=distributed,
        )

        # No validation loader for a streamed dataset — val episodes aren't on
        # disk and streaming val through ChunkedDataLoader conflicts with train.
        # Use open-loop eval (eval_every) for periodic evaluation instead.
        val_loader = None

        return train_loader, val_loader

    # When caching, all data is in RAM — no I/O to parallelize.
    # Using num_workers>0 would duplicate the cache across worker processes.
    if cache_frames:
        num_workers = 0

    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
