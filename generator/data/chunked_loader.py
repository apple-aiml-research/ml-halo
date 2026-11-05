#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Chunked streaming DataLoader for large datasets.

Streams an s3:// dataset (see data/remote.py) per chunk with a RAM double buffer: while all
ranks train on the current chunk (resident in RAM), a background thread
downloads + extracts the next chunk and loads it into a separate set of
RAM dicts. At the chunk boundary the prefetched buffer is swapped in
(activate_chunk) instead of being read from disk, so the disk->RAM load
is overlapped with training rather than blocking. Costs ~2x chunk RAM.

Rank 0 is the sole downloader; other ranks wait on a file sentinel (not
an NCCL collective, so it can wait arbitrarily long) before loading the
chunk from the shared filesystem. dist.barrier() calls stay on the main
thread and only gate disk cleanup, never the slow download.
"""

import math
import os
import threading
import time as _time
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist

from .dataset import MotionDataset, collate_fn, _extract_episode_frames, usable_cpus


class _DatasetLenProxy:
    def __init__(self, length: int):
        self._length = length

    def __len__(self) -> int:
        return self._length


class ChunkedDataLoader:
    """Streams dataset from S3 in memory-cached chunks with double buffering.

    Drop-in replacement for DataLoader: supports set_epoch, __iter__,
    __len__, and .dataset.
    """

    def __init__(
        self,
        underlying_dataset: MotionDataset,
        sample_indices: List[int],
        batch_size: int,
        chunk_episodes: int,
        collate_fn: Callable = collate_fn,
        pin_memory: bool = True,
        distributed: bool = False,
    ):
        self.underlying_dataset = underlying_dataset
        self.batch_size = batch_size
        self.chunk_episodes = chunk_episodes
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.distributed = distributed

        self._epoch = 0
        self._rank = dist.get_rank() if distributed else 0
        self._world_size = dist.get_world_size() if distributed else 1
        # Threads used to assemble (and pin) batches ahead of the GPU.
        # Value-neutral: batches are yielded in submission order. Overridable via env.
        self._assembly_workers = int(os.environ.get("LOADER_ASSEMBLY_WORKERS", "8"))
        # PERSIST_CHUNKS=1 keeps per-chunk files on node-local disk across epochs
        # instead of deleting + re-downloading them every epoch. Opt-in: the
        # extracted frames are large.
        self._persist_chunks = os.environ.get("PERSIST_CHUNKS", "0") == "1"
        # Whole dataset to node-local disk before the first step (_predownload_all).
        # Needs PERSIST_CHUNKS=1, or chunk cleanup deletes it again after one use.
        self._predownload = os.environ.get("PREDOWNLOAD_ALL", "0") == "1"
        self._predownload_done = False
        if self._predownload and not self._persist_chunks:
            raise ValueError("PREDOWNLOAD_ALL=1 requires PERSIST_CHUNKS=1")
        # Node-local rank (torchrun sets LOCAL_RANK per process). Disk
        # downloads/extraction/cleanup are gated on local-rank 0 so EACH node
        # populates its own (node-local) local_dir and its own file sentinels.
        # Global _rank/_world_size still drive cross-node sample sharding.
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))

        ep_to_samples: Dict[int, List[int]] = defaultdict(list)
        for si in sample_indices:
            ep_idx = underlying_dataset.samples[si][0]
            ep_to_samples[ep_idx].append(si)

        all_episodes = sorted(ep_to_samples.keys())

        self._chunks: List[List[int]] = []
        for i in range(0, len(all_episodes), chunk_episodes):
            self._chunks.append(all_episodes[i : i + chunk_episodes])

        self._chunk_samples: List[List[int]] = []
        for chunk_eps in self._chunks:
            chunk_si = []
            for ep_idx in chunk_eps:
                chunk_si.extend(ep_to_samples[ep_idx])
            self._chunk_samples.append(chunk_si)

        self._total_samples = len(sample_indices)
        self.dataset = _DatasetLenProxy(self._total_samples)

        # Cross-epoch carryover: the next epoch's first chunk, prefetched into
        # RAM during this epoch's last chunk so the next epoch doesn't cold-start.
        self._carryover = None

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        samples_per_rank = math.ceil(self._total_samples / self._world_size)
        return math.ceil(samples_per_rank / self.batch_size)

    # ------------------------------------------------------------------
    # S3 download / extract / cleanup
    # ------------------------------------------------------------------

    def _download_and_extract_chunk(self, episode_ids: List[int]):
        self._download_chunk_files(episode_ids)
        self._extract_chunk_frames(episode_ids)

    def _download_chunk_files(self, episode_ids: List[int]):
        self.underlying_dataset.download_episodes(episode_ids)

    def _extract_chunk_frames(self, episode_ids: List[int]):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ds = self.underlying_dataset
        args = [
            (ep_idx, ds.local_dir, ds.image_size)
            for ep_idx in episode_ids
            if not (ds.local_dir / "frames" / "observation.images.fpv_cam"
                    / f"episode_{ep_idx:06d}.pt").exists()
        ]
        if args:
            num_workers = min(usable_cpus(), len(args))
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
                list(pool.map(_extract_episode_frames, args))

    def _predownload_all(self):
        """PREDOWNLOAD_ALL=1: put the WHOLE dataset on node-local disk (videos,
        parquets AND extracted frames) before the first step.

        The streaming path otherwise downloads + decodes every chunk inside the
        training loop, every epoch (the live-RGB path deletes its files after
        each chunk). On one run that prefetch took ~15 min per 3800-episode
        chunk, longer than the ~10 min of training on it, adding a ~290 s stall
        at every chunk swap. With everything on disk (and PERSIST_CHUNKS=1 so it
        stays there), a chunk prefetch is only the disk -> RAM load.

        Local rank 0 of each node pulls its node's copy; downloading chunk i+1
        overlaps decoding chunk i, with no training competing for the CPUs.
        Other local ranks poll a file sentinel. Nodes then meet on a TCPStore
        counter instead of an NCCL barrier, whose 60 min timeout a slow node's
        download could exceed.
        """
        from concurrent.futures import ThreadPoolExecutor

        ds = self.underlying_dataset
        sentinel = ds.local_dir / ".predownload_all_ready"
        t0 = _time.time()
        if self._local_rank == 0:
            # Pull chunks in the order this epoch visits them, so that with
            # PREDOWNLOAD_MAX_CHUNKS (smoke/profile runs) the chunks training
            # touches first are the ones already on disk.
            # PREDOWNLOAD_MAX_CHUNKS: smoke tests only — stop after this many
            # chunks to measure per-chunk time and S3 health cheaply.
            order = [self._chunks[c] for c in self._chunk_order(self._epoch)]
            n = len(order)
            n = min(n, int(os.environ.get("PREDOWNLOAD_MAX_CHUNKS", n)))
            with ThreadPoolExecutor(max_workers=1) as dl:
                fut = dl.submit(self._download_chunk_files, order[0])
                for i, chunk in enumerate(order[:n]):
                    fut.result()
                    if i + 1 < n:
                        fut = dl.submit(self._download_chunk_files, order[i + 1])
                    self._extract_chunk_frames(chunk)
                    print(f"[predownload] node of rank {self._rank}: chunk {i + 1}/{n} "
                          f"on disk ({(_time.time() - t0) / 60:.1f} min)", flush=True)
            sentinel.touch()
        else:
            while not sentinel.exists():
                _time.sleep(10)
                if _time.time() - t0 > 12 * 3600:
                    raise RuntimeError(f"[Rank {self._rank}] predownload timeout (12h)")
        if self.distributed:
            store = dist.distributed_c10d._get_default_store()
            store.add("predownload_all_done", 1)
            while store.add("predownload_all_done", 0) < self._world_size:
                _time.sleep(10)
                if _time.time() - t0 > 13 * 3600:
                    raise RuntimeError(f"[Rank {self._rank}] predownload sync timeout")
        if self._rank == 0:
            print(f"[predownload] all {self._world_size} ranks have the full dataset on "
                  f"local disk after {(_time.time() - t0) / 60:.1f} min", flush=True)

    def _cleanup_chunk_disk(self, episode_ids: List[int]):
        # When persisting (PERSIST_CHUNKS=1), keep the downloaded
        # files on node-local disk so later epochs reuse them instead of
        # re-streaming from S3. This is what lets the per-step speedup flow
        # through to the whole-job wall-clock.
        if self._persist_chunks:
            return
        ds = self.underlying_dataset
        for ep_idx in episode_ids:
            paths = [
                ds.local_dir / "videos" / "observation.images.fpv_cam" / f"episode_{ep_idx:06d}.mp4",
                ds.local_dir / "data" / f"episode_{ep_idx:06d}.parquet",
                ds.local_dir / "frames" / "observation.images.fpv_cam" / f"episode_{ep_idx:06d}.pt",
            ]
            for path in paths:
                path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _yield_chunk_batches(self, chunk_idx):
        sample_indices = self._chunk_samples[chunk_idx]

        sg = torch.Generator()
        sg.manual_seed(self._epoch * len(self._chunks) + chunk_idx)
        perm = torch.randperm(len(sample_indices), generator=sg).tolist()
        shuffled = [sample_indices[i] for i in perm]

        rank_samples = shuffled[self._rank :: self._world_size]

        if self.distributed and rank_samples:
            num_batches = math.ceil(
                len(shuffled) / (self._world_size * self.batch_size)
            )
            target_len = num_batches * self.batch_size
            orig_len = len(rank_samples)
            while len(rank_samples) < target_len:
                rank_samples.append(rank_samples[len(rank_samples) % orig_len])

        batch_index_lists = [
            rank_samples[i : i + self.batch_size]
            for i in range(0, len(rank_samples), self.batch_size)
        ]
        yield from self._yield_assembled(batch_index_lists)

    def _yield_assembled(self, batch_index_lists):
        """Assemble (and pin) batches, concurrently prefetched ahead of the GPU.

        The heavy per-sample work (vectorized numpy trajectory extraction,
        frame tensor indexing, pin_memory) releases the GIL, so threads give
        real overlap. Batches are yielded in submission order, so data order is
        identical to the serial path — only the wall-clock changes.
        """
        def _assemble(batch_indices):
            batch = [self.underlying_dataset[si] for si in batch_indices]
            collated = self.collate_fn(batch)
            if self.pin_memory:
                collated = _pin_batch(collated)
            return collated

        if self._assembly_workers <= 1:
            for bi in batch_index_lists:
                yield _assemble(bi)
            return

        from concurrent.futures import ThreadPoolExecutor
        from collections import deque as _deque

        prefetch = self._assembly_workers + 2
        with ThreadPoolExecutor(max_workers=self._assembly_workers) as pool:
            it = iter(batch_index_lists)
            futures = _deque()
            for _ in range(prefetch):
                try:
                    futures.append(pool.submit(_assemble, next(it)))
                except StopIteration:
                    break
            while futures:
                nxt = futures.popleft()
                try:
                    futures.append(pool.submit(_assemble, next(it)))
                except StopIteration:
                    pass
                yield nxt.result()

    def _prefetch_chunk_to_ram(self, chunk_idx: int, holder: dict, epoch: int):
        """Background worker: ensure a chunk is on disk, then load it to RAM.

        Runs on every rank. Rank 0 downloads + extracts the chunk and signals
        completion via a file sentinel; other ranks poll that sentinel (a plain
        file check — safe off the main thread, unlike an NCCL collective) before
        loading. The resulting (frames, parquets) dicts land in `holder["buf"]`
        for the main loop to activate; any exception lands in `holder["error"]`.

        `epoch` names the sentinel so a cross-epoch prefetch (next epoch's first
        chunk) doesn't collide with this epoch's sentinels.
        """
        try:
            episode_ids = self._chunks[chunk_idx]
            sentinel = (
                self.underlying_dataset.local_dir
                / f".chunk_{chunk_idx}_epoch_{epoch}_ready"
            )
            if self._local_rank == 0:
                self._download_and_extract_chunk(episode_ids)
                if self.distributed:
                    sentinel.touch()
            elif self.distributed:
                waited = 0
                while not sentinel.exists():
                    _time.sleep(1)
                    waited += 1
                    if waited > 21600:
                        raise RuntimeError(
                            f"[Rank {self._rank}] Timed out waiting for chunk "
                            f"{chunk_idx} download (6h)"
                        )
            holder["buf"] = self.underlying_dataset._read_chunk_from_disk(
                episode_ids
            )
        except Exception as e:
            print(f"[Rank {self._rank}] Prefetch error: {e}")
            traceback.print_exc()
            holder["error"] = e

    def _store_barrier(self, name: str, timeout_s: int = 4 * 3600):
        """All-rank barrier on the TCPStore: no NCCL collective, so no 60 min
        NCCL timeout while a node is slow (e.g. retrying S3). `name` must be
        unique per call site and iteration."""
        store = dist.distributed_c10d._get_default_store()
        key = f"barrier_{name}"
        store.add(key, 1)
        t0 = _time.time()
        while store.add(key, 0) < self._world_size:
            _time.sleep(1)
            if _time.time() - t0 > timeout_s:
                raise RuntimeError(f"[Rank {self._rank}] store barrier {name} timed out")

    def _start_prefetch(self, chunk_idx: int, epoch: int = None):
        if epoch is None:
            epoch = self._epoch
        holder: dict = {}
        thread = threading.Thread(
            target=self._prefetch_chunk_to_ram,
            args=(chunk_idx, holder, epoch),
            daemon=True,
        )
        thread.start()
        return thread, holder

    def _join_prefetch(self, thread, holder):
        thread.join(timeout=21600)
        if thread.is_alive():
            raise RuntimeError("Prefetch thread timed out after 6 hours")
        if "error" in holder:
            raise RuntimeError(
                f"Prefetch failed: {holder['error']}"
            ) from holder["error"]
        return holder["buf"]

    def _chunk_order(self, epoch: int):
        """Deterministic chunk visitation order for an epoch (same on all ranks)."""
        g = torch.Generator()
        g.manual_seed(epoch)
        return torch.randperm(len(self._chunks), generator=g).tolist()

    def __iter__(self):
        if self._predownload and not self._predownload_done:
            self._predownload_all()
            self._predownload_done = True
        chunk_order = self._chunk_order(self._epoch)
        first_chunk = chunk_order[0]

        # Reuse the chunk prefetched at the end of the previous epoch if it
        # matches; otherwise cold-start (only happens on the very first epoch,
        # or if the trainer jumps epochs non-sequentially).
        carry = self._carryover
        self._carryover = None
        if carry is not None and carry[0] == self._epoch and carry[1] == first_chunk:
            buf = carry[2]
        else:
            carry = None  # stale — drop it so its RAM is reclaimed
            thread, holder = self._start_prefetch(first_chunk)
            buf = self._join_prefetch(thread, holder)

        for i, chunk_idx in enumerate(chunk_order):
            # All ranks now hold `chunk_idx` in RAM (joined above, or at the end
            # of the previous iteration). Barrier before rank 0 deletes the disk
            # files so no rank is still reading them. It CAN wait on a slow
            # download — another node's — so it is a TCPStore barrier, not an
            # NCCL one: a run died here of the 60 min NCCL timeout while one
            # node retried S3 timeouts.
            if self.distributed:
                self._store_barrier(f"chunk_e{self._epoch}_i{i}")
            if self._local_rank == 0:
                self._cleanup_chunk_disk(self._chunks[chunk_idx])
                sentinel = (
                    self.underlying_dataset.local_dir
                    / f".chunk_{chunk_idx}_epoch_{self._epoch}_ready"
                )
                sentinel.unlink(missing_ok=True)

            # Swap the freshly loaded chunk in (drops the previous chunk's RAM).
            self.underlying_dataset.activate_chunk(*buf)
            buf = None

            # Prefetch the next chunk in the background while we train on the
            # current one. For the last chunk of the epoch, prefetch the NEXT
            # epoch's first chunk instead and carry it over — this hides the
            # cross-epoch cold start behind the last chunk's training.
            next_thread = next_holder = None
            cross_epoch = i + 1 == len(chunk_order)
            if cross_epoch:
                next_epoch = self._epoch + 1
                next_chunk = self._chunk_order(next_epoch)[0]
                next_thread, next_holder = self._start_prefetch(next_chunk, next_epoch)
            else:
                next_chunk = chunk_order[i + 1]
                next_thread, next_holder = self._start_prefetch(next_chunk)

            yield from self._yield_chunk_batches(chunk_idx)

            # Free the chunk we just trained on, then collect the prefetched one
            # (usually already done, so this join returns immediately).
            self.underlying_dataset.unload_chunk()
            nbuf = self._join_prefetch(next_thread, next_holder)
            if cross_epoch:
                self._carryover = (next_epoch, next_chunk, nbuf)
            else:
                buf = nbuf

        # Leave the epoch together. The cross-epoch prefetch joined above can be
        # slow on one node, and the trainer's epoch-end collectives are NCCL
        # (60 min timeout), so meet here on the store first.
        if self.distributed:
            self._store_barrier(f"epoch_end_e{self._epoch}")


def _pin_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    pinned = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            pinned[k] = v.pin_memory()
        else:
            pinned[k] = v
    return pinned
