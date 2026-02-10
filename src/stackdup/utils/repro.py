from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class ReproMeta:
    seed: int
    deterministic: bool
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_reproducibility(seed: int, deterministic: bool) -> ReproMeta:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Some ops may still be nondeterministic depending on hardware/drivers.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    return ReproMeta(
        seed=seed,
        deterministic=deterministic,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=getattr(torch.version, "cuda", None),
        cudnn_version=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    )
