"""Device selection for CPU / CUDA / NPU. NPU backend uses torch_npu API."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_NPU_AVAILABLE: bool = False


def _check_npu_available() -> bool:
    """Lazy check: torch_npu (NPU API) is imported only when NPU is requested."""
    global _NPU_AVAILABLE
    if _NPU_AVAILABLE:
        return True
    try:
        import torch_npu  # noqa: F401  # registers torch.npu
        import torch
        _NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        _NPU_AVAILABLE = False
    return _NPU_AVAILABLE


def get_device(device_str: str) -> "torch.device":
    """
    Resolve device from config string. NPU uses torch_npu API (Ascend).

    Args:
        device_str: "auto", "cpu", "cuda", "cuda:0", "npu", "npu:0", etc.

    Returns:
        torch.device to use for model and tensors.

    Raises:
        RuntimeError: If requested device is not available (e.g. npu without torch_npu).
    """
    import torch

    s = (device_str or "auto").strip().lower()
    if s in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _check_npu_available():
            return torch.device("npu")
        return torch.device("cpu")

    if s == "cpu":
        return torch.device("cpu")

    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Install PyTorch with CUDA or set device to cpu."
            )
        return torch.device(s if ":" in s else "cuda")

    if s.startswith("npu"):
        if not _check_npu_available():
            raise RuntimeError(
                "NPU requested but torch_npu (NPU API) is not installed. "
                "Install torch_npu for Ascend NPU support, or set device to cpu/cuda."
            )
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError(
                "NPU requested but NPU runtime is not available. "
                "Check drivers and torch_npu installation, or set device to cpu/cuda."
            )
        return torch.device(s if ":" in s else "npu")

    raise ValueError(
        f"Unsupported device: {device_str}. Use auto, cpu, cuda, cuda:0, npu, npu:0."
    )
