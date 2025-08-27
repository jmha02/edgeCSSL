"""
Memory monitoring utilities for tracking GPU memory usage during training
"""
import torch
import psutil
import os
from typing import Dict, Optional
import gc


class MemoryMonitor:
    """Monitor GPU and CPU memory usage during training."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize memory monitor.
        
        Args:
            device: CUDA device to monitor. If None, uses current device.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_stats()
        
    def reset_stats(self):
        """Reset all memory statistics."""
        self.peak_gpu_memory = 0.0
        self.current_gpu_memory = 0.0
        self.peak_cpu_memory = 0.0
        self.current_cpu_memory = 0.0
        
    def get_gpu_memory_mb(self) -> Dict[str, float]:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "cached": 0.0, "reserved": 0.0}
            
        # Force garbage collection to get accurate readings
        gc.collect()
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024  # MB
        cached = torch.cuda.memory_reserved(self.device) / 1024 / 1024  # MB
        
        return {
            "allocated": allocated,
            "cached": cached, 
            "reserved": cached,  # reserved is same as cached in newer PyTorch
        }
        
    def get_cpu_memory_mb(self) -> Dict[str, float]:
        """Get current CPU memory usage in MB."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        }
        
    def update_peak_memory(self):
        """Update peak memory usage statistics."""
        gpu_mem = self.get_gpu_memory_mb()
        cpu_mem = self.get_cpu_memory_mb()
        
        self.current_gpu_memory = gpu_mem["allocated"]
        self.current_cpu_memory = cpu_mem["rss"]
        
        if self.current_gpu_memory > self.peak_gpu_memory:
            self.peak_gpu_memory = self.current_gpu_memory
            
        if self.current_cpu_memory > self.peak_cpu_memory:
            self.peak_cpu_memory = self.current_cpu_memory
            
    def get_memory_summary(self) -> Dict[str, float]:
        """Get comprehensive memory usage summary."""
        gpu_mem = self.get_gpu_memory_mb()
        cpu_mem = self.get_cpu_memory_mb()
        
        return {
            "current_gpu_allocated_mb": gpu_mem["allocated"],
            "current_gpu_cached_mb": gpu_mem["cached"],
            "current_cpu_rss_mb": cpu_mem["rss"],
            "current_cpu_vms_mb": cpu_mem["vms"],
            "peak_gpu_allocated_mb": self.peak_gpu_memory,
            "peak_cpu_rss_mb": self.peak_cpu_memory,
        }
        
    def log_memory_usage(self, phase: str = "", step: Optional[int] = None) -> Dict[str, float]:
        """
        Log current memory usage and update peaks.
        
        Args:
            phase: Training phase (e.g., "forward", "backward", "epoch_end")
            step: Current training step
            
        Returns:
            Dictionary with current memory statistics
        """
        self.update_peak_memory()
        memory_stats = self.get_memory_summary()
        
        # Create log message
        phase_str = f"[{phase}]" if phase else ""
        step_str = f" Step {step}" if step is not None else ""
        
        print(f"Memory Usage{phase_str}{step_str}:")
        print(f"  GPU: {memory_stats['current_gpu_allocated_mb']:.1f}MB "
              f"(Peak: {memory_stats['peak_gpu_allocated_mb']:.1f}MB)")
        print(f"  CPU: {memory_stats['current_cpu_rss_mb']:.1f}MB "
              f"(Peak: {memory_stats['peak_cpu_rss_mb']:.1f}MB)")
              
        return memory_stats
        
    def compare_memory_usage(self, baseline_stats: Dict[str, float], 
                           current_stats: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compare current memory usage with baseline.
        
        Args:
            baseline_stats: Baseline memory statistics to compare against
            current_stats: Current stats (if None, will get current stats)
            
        Returns:
            Dictionary with memory differences
        """
        if current_stats is None:
            current_stats = self.get_memory_summary()
            
        comparison = {}
        for key in baseline_stats:
            if key in current_stats:
                diff = current_stats[key] - baseline_stats[key]
                comparison[f"{key}_diff"] = diff
                comparison[f"{key}_ratio"] = current_stats[key] / baseline_stats[key] if baseline_stats[key] > 0 else float('inf')
                
        return comparison
        
    def format_memory_report(self, title: str = "Memory Report") -> str:
        """Generate a formatted memory usage report."""
        stats = self.get_memory_summary()
        
        report = f"\n{'='*60}\n{title}\n{'='*60}\n"
        report += f"GPU Memory:\n"
        report += f"  Current Allocated: {stats['current_gpu_allocated_mb']:.1f} MB\n"
        report += f"  Current Cached:    {stats['current_gpu_cached_mb']:.1f} MB\n"
        report += f"  Peak Allocated:    {stats['peak_gpu_allocated_mb']:.1f} MB\n\n"
        
        report += f"CPU Memory:\n"
        report += f"  Current RSS:       {stats['current_cpu_rss_mb']:.1f} MB\n"
        report += f"  Current VMS:       {stats['current_cpu_vms_mb']:.1f} MB\n"
        report += f"  Peak RSS:          {stats['peak_cpu_rss_mb']:.1f} MB\n"
        report += f"{'='*60}\n"
        
        return report


def get_model_memory_footprint(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model memory footprint.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model memory statistics in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    total_size = param_size + buffer_size
    
    return {
        "parameters_mb": param_size / 1024 / 1024,
        "buffers_mb": buffer_size / 1024 / 1024,
        "total_model_mb": total_size / 1024 / 1024,
    }


def profile_memory_usage(func):
    """
    Decorator to profile memory usage of a function.
    
    Usage:
        @profile_memory_usage
        def training_step(self, batch, batch_idx):
            # training code here
            pass
    """
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        
        # Get memory before function call
        print(f"Memory before {func.__name__}:")
        before_stats = monitor.log_memory_usage()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Get memory after function call
        print(f"Memory after {func.__name__}:")
        after_stats = monitor.log_memory_usage()
        
        # Show the difference
        diff = monitor.compare_memory_usage(before_stats, after_stats)
        print(f"Memory difference for {func.__name__}:")
        for key, value in diff.items():
            if key.endswith('_diff'):
                print(f"  {key}: {value:.1f} MB")
                
        return result
        
    return wrapper