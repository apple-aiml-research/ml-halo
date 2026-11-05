#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Video decoding with PyAV: one sequential pass over the video.

For the dataset's H.264 videos PyAV decodes to the same RGB pixels as decord,
which the shipped models and VLM caches were produced with.
"""

from pathlib import Path
from typing import Iterator, Tuple, Union

import av
import torch
import torchvision.transforms.functional as TF


def iter_frames(video_path: Union[str, Path]) -> Iterator[torch.Tensor]:
    """Yield every frame of the video in order, as a [3, H, W] uint8 RGB tensor."""
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            yield torch.from_numpy(frame.to_rgb().to_ndarray()).permute(2, 0, 1)


def read_frames(video_path: Union[str, Path], image_size: Tuple[int, int]) -> torch.Tensor:
    """All frames resized to image_size -> uint8 [N, 3, H, W].

    Each frame goes /255 -> TF.resize -> *255 -> uint8, the pipeline the
    extracted training frames and the VLM cache are built with.
    """
    frames = [
        (TF.resize(f.float() / 255.0, list(image_size)) * 255).to(torch.uint8)
        for f in iter_frames(video_path)
    ]
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return torch.stack(frames)
