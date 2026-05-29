import torch
import gc


def estimate_preprocessing_gpu_memory(point_count, has_reflectance=True, grid_sizes=[1.0, 2.0, 4.0]):
    """
    Estimate GPU memory needed for preprocessing based on empirical observations.

    Args:
        point_count (int): Number of points in the point cloud
        has_reflectance (bool): Whether reflectance data is present
        grid_sizes (list): List of grid sizes for voxelization

    Returns:
        float: Estimated GPU memory in GB
    """
    channels = 4 if has_reflectance else 3
    base_memory = point_count * channels * 4  # 4 bytes per float32

    # Empirical model based on actual usage patterns:
    # - Original tensor: 1x
    # - Working copies during processing: ~2x per grid size
    # - Intermediate tensors (cluster indices, scatter ops): ~1.5x
    # - GPU memory fragmentation: ~15%

    num_grids = len(grid_sizes)
    processing_multiplier = 1 + (num_grids * 2.0) + 1.5  # Empirically derived
    fragmentation_multiplier = 1.15

    estimated_bytes = base_memory * processing_multiplier * fragmentation_multiplier
    return estimated_bytes / (1024**3)  # Convert to GB


def get_available_gpu_memory():
    """
    Get available GPU memory in GB.

    Returns:
        float: Available GPU memory in GB, 0 if no GPU available
    """
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            return free / (1024**3)
        except Exception:
            return 0
    return 0


def get_current_gpu_memory():
    """
    Get currently allocated GPU memory in GB.

    Returns:
        float: Currently allocated GPU memory in GB
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0


def get_peak_gpu_memory():
    """
    Get peak GPU memory usage in GB.

    Returns:
        float: Peak GPU memory usage in GB
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def should_use_cpu_preprocessing(point_count, has_reflectance=True, grid_sizes=[1.0, 2.0, 4.0], threshold=0.8):
    """
    Determine whether to use CPU or GPU for preprocessing based on memory requirements.

    Args:
        point_count (int): Number of points in the point cloud
        has_reflectance (bool): Whether reflectance data is present
        grid_sizes (list): List of grid sizes for voxelization
        threshold (float): GPU memory usage threshold (0.8 = 80%)

    Returns:
        tuple: (use_cpu: bool, estimated_gb: float, available_gb: float)
    """
    estimated_gpu_mem = estimate_preprocessing_gpu_memory(point_count, has_reflectance, grid_sizes)
    available_gpu_mem = get_available_gpu_memory()

    # Use CPU if estimated memory > threshold of available GPU memory
    use_cpu = estimated_gpu_mem > (available_gpu_mem * threshold) if available_gpu_mem > 0 else True

    return use_cpu, estimated_gpu_mem, available_gpu_mem


def format_memory_info(point_count, has_reflectance, grid_sizes, resolution=None):
    """
    Format memory information for user display.

    Returns:
        str: Formatted memory information string
    """
    use_cpu, estimated_gb, available_gb = should_use_cpu_preprocessing(
        point_count, has_reflectance, grid_sizes
    )

    device_choice = "CPU" if use_cpu else "GPU"
    return (
        f"Point cloud: {point_count:,} points, estimated memory: {estimated_gb:.1f}GB, "
        f"available GPU: {available_gb:.1f}GB -> using {device_choice}"
    )