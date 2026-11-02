import torch


def _peak_gpu_mem_increase(fn, *args, device=None):
    """Return the peak bytes the CUDA caching allocator hands out while ``fn`` runs.

    ``torch.cuda.mem_get_info`` reports device-wide usage, which on GPUs that share memory with
    the CPU (e.g. Jetson Thor) moves with every other allocation on the system. The caching
    allocator peak instead counts only the tensors allocated by ``fn``, including temporaries
    freed before it returns.

    ``device`` defaults to the current CUDA device; pass an explicit device when ``fn``
    allocates on a non-current device.
    """
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    fn(*args)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) - before


def is_sm10x():
    return torch.cuda.get_device_capability() >= (10, 0)


def is_hopper():
    return torch.cuda.get_device_capability() == (9, 0)
