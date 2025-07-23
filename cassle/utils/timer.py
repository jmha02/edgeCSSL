import time
import torch
import json
import csv
import os
from datetime import datetime
from typing import Dict, Optional, Union, List
from contextlib import contextmanager


class Timer:
    """Timer utility for measuring execution time of code blocks and training components."""
    
    def __init__(self, sync_cuda: bool = True):
        """
        Initialize timer with optional CUDA synchronization.
        
        Args:
            sync_cuda (bool): Whether to synchronize CUDA before timing measurements.
        """
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.timers: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        
    def _get_time(self) -> float:
        """Get current time with optional CUDA synchronization."""
        if self.sync_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()
    
    def start(self, name: str) -> None:
        """Start timing for a named timer."""
        self.start_times[name] = self._get_time()
    
    def stop(self, name: str) -> float:
        """Stop timing for a named timer and return elapsed time."""
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = self._get_time() - self.start_times[name]
        
        if name not in self.timers:
            self.timers[name] = 0.0
            self.counts[name] = 0
            
        self.timers[name] += elapsed
        self.counts[name] += 1
        del self.start_times[name]
        
        return elapsed
    
    @contextmanager
    def time(self, name: str):
        """Context manager for timing code blocks."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def get_time(self, name: str) -> float:
        """Get total accumulated time for a timer."""
        return self.timers.get(name, 0.0)
    
    def get_average_time(self, name: str) -> float:
        """Get average time per call for a timer."""
        if name not in self.timers or self.counts[name] == 0:
            return 0.0
        return self.timers[name] / self.counts[name]
    
    def get_count(self, name: str) -> int:
        """Get number of times a timer was called."""
        return self.counts.get(name, 0)
    
    def reset(self, name: Optional[str] = None) -> None:
        """Reset timer(s). If name is None, reset all timers."""
        if name is None:
            self.timers.clear()
            self.start_times.clear()
            self.counts.clear()
        else:
            self.timers.pop(name, None)
            self.start_times.pop(name, None)
            self.counts.pop(name, None)
    
    def get_summary(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """Get a summary of all timers."""
        summary = {}
        for name in self.timers:
            summary[name] = {
                'total_time': self.timers[name],
                'average_time': self.get_average_time(name),
                'count': self.counts[name]
            }
        return summary
    
    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds >= 3600:
            return f"{seconds/3600:.2f}h"
        elif seconds >= 60:
            return f"{seconds/60:.2f}m"
        elif seconds >= 1:
            return f"{seconds:.2f}s"
        elif seconds >= 0.001:
            return f"{seconds*1000:.2f}ms"
        else:
            return f"{seconds*1000000:.2f}μs"
    
    def print_summary(self, sort_by: str = 'total_time') -> None:
        """Print a formatted summary of all timers."""
        summary = self.get_summary()
        if not summary:
            print("No timing data available")
            return
        
        # Sort by specified metric
        sorted_items = sorted(summary.items(), 
                            key=lambda x: x[1][sort_by], 
                            reverse=True)
        
        print("\n" + "="*80)
        print(f"{'Timer Name':<25} {'Total Time':<12} {'Avg Time':<12} {'Count':<8}")
        print("="*80)
        
        for name, data in sorted_items:
            total_str = self.format_time(data['total_time'])
            avg_str = self.format_time(data['average_time'])
            print(f"{name:<25} {total_str:<12} {avg_str:<12} {data['count']:<8}")
        
        print("="*80)


class TrainingTimer(Timer):
    """Specialized timer for tracking training-related metrics."""
    
    def __init__(self, sync_cuda: bool = True):
        super().__init__(sync_cuda)
        self.epoch_start_time: Optional[float] = None
        self.training_start_time: Optional[float] = None
        
    def start_training(self) -> None:
        """Mark the start of entire training process."""
        self.training_start_time = self._get_time()
        self.start('total_training_time')
    
    def start_epoch(self, epoch: int) -> None:
        """Mark the start of an epoch."""
        self.epoch_start_time = self._get_time()
        self.start(f'epoch_{epoch}')
    
    def end_epoch(self, epoch: int) -> float:
        """Mark the end of an epoch and return elapsed time."""
        return self.stop(f'epoch_{epoch}')
    
    def end_training(self) -> float:
        """Mark the end of training and return total training time."""
        return self.stop('total_training_time')
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get comprehensive training timing metrics."""
        metrics = {}
        
        # Total training time
        if 'total_training_time' in self.timers:
            metrics['total_training_time_seconds'] = self.get_time('total_training_time')
            metrics['total_training_time_formatted'] = self.format_time(metrics['total_training_time_seconds'])
        
        # Step timing statistics
        if self.get_count('training_step') > 0:
            metrics['avg_step_time_seconds'] = self.get_average_time('training_step')
            metrics['avg_step_time_formatted'] = self.format_time(metrics['avg_step_time_seconds'])
            metrics['total_steps'] = self.get_count('training_step')
        
        # Epoch timing statistics
        epoch_times = [self.get_time(name) for name in self.timers if name.startswith('epoch_')]
        if epoch_times:
            metrics['avg_epoch_time_seconds'] = sum(epoch_times) / len(epoch_times)
            metrics['avg_epoch_time_formatted'] = self.format_time(metrics['avg_epoch_time_seconds'])
            metrics['total_epochs'] = len(epoch_times)
        
        # Throughput metrics
        if 'total_training_time' in self.timers and self.get_count('training_step') > 0:
            total_time = self.get_time('total_training_time')
            total_steps = self.get_count('training_step')
            metrics['steps_per_second'] = total_steps / total_time
        
        return metrics
    
    def log_metrics_to_dict(self, prefix: str = 'timing/') -> Dict[str, float]:
        """Get timing metrics formatted for logging (e.g., to WandB)."""
        metrics = self.get_training_metrics()
        
        # Filter out formatted strings for logging (keep only numerical values)
        log_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_metrics[f"{prefix}{key}"] = value
        
        return log_metrics


class TimingLogger:
    """Logger for writing detailed timing analysis to files."""
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        """
        Initialize timing logger.
        
        Args:
            log_dir (str): Directory to save log files
            experiment_name (str): Name of the experiment (used in filenames)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"timing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log data storage
        self.epoch_logs: List[Dict] = []
        self.step_logs: List[Dict] = []
        self.overall_logs: Dict = {}
        self.task_logs: List[Dict] = []
        
        # File paths
        self.epoch_log_file = os.path.join(log_dir, f"{self.experiment_name}_epoch_timing.csv")
        self.step_log_file = os.path.join(log_dir, f"{self.experiment_name}_step_timing.csv")
        self.summary_log_file = os.path.join(log_dir, f"{self.experiment_name}_timing_summary.json")
        self.task_log_file = os.path.join(log_dir, f"{self.experiment_name}_task_timing.csv")
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
        print(f"Timing logger initialized. Logs will be saved to: {log_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with appropriate headers."""
        # Epoch timing CSV
        with open(self.epoch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'epoch', 'task_id', 'epoch_time_seconds', 
                'epoch_time_formatted', 'total_steps_in_epoch', 
                'avg_step_time_seconds', 'steps_per_second'
            ])
        
        # Step timing CSV
        with open(self.step_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'epoch', 'step', 'task_id', 
                'step_time_seconds', 'forward_time_seconds', 
                'online_eval_time_seconds'
            ])
        
        # Task timing CSV
        with open(self.task_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'task_id', 'task_time_seconds', 
                'task_time_formatted', 'total_epochs', 'total_steps',
                'avg_epoch_time_seconds', 'avg_step_time_seconds'
            ])
    
    def log_epoch(self, epoch: int, task_id: int, timer: 'TrainingTimer'):
        """Log epoch timing data."""
        timestamp = datetime.now().isoformat()
        metrics = timer.get_training_metrics()
        
        epoch_data = {
            'timestamp': timestamp,
            'epoch': epoch,
            'task_id': task_id,
            'epoch_time_seconds': timer.get_time(f'epoch_{epoch}'),
            'epoch_time_formatted': timer.format_time(timer.get_time(f'epoch_{epoch}')),
            'total_steps_in_epoch': timer.get_count('training_step'),
            'avg_step_time_seconds': metrics.get('avg_step_time_seconds', 0),
            'steps_per_second': metrics.get('steps_per_second', 0)
        }
        
        self.epoch_logs.append(epoch_data)
        
        # Write to CSV
        with open(self.epoch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch_data['timestamp'], epoch_data['epoch'], epoch_data['task_id'],
                epoch_data['epoch_time_seconds'], epoch_data['epoch_time_formatted'],
                epoch_data['total_steps_in_epoch'], epoch_data['avg_step_time_seconds'],
                epoch_data['steps_per_second']
            ])
    
    def log_step(self, epoch: int, step: int, task_id: int, timer: 'TrainingTimer'):
        """Log individual step timing data."""
        timestamp = datetime.now().isoformat()
        
        step_data = {
            'timestamp': timestamp,
            'epoch': epoch,
            'step': step,
            'task_id': task_id,
            'step_time_seconds': timer.get_average_time('training_step'),
            'forward_time_seconds': timer.get_average_time('forward_pass'),
            'online_eval_time_seconds': timer.get_average_time('online_eval')
        }
        
        self.step_logs.append(step_data)
        
        # Write to CSV (only every 10 steps to avoid excessive logging)
        if step % 10 == 0:
            with open(self.step_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step_data['timestamp'], step_data['epoch'], step_data['step'],
                    step_data['task_id'], step_data['step_time_seconds'],
                    step_data['forward_time_seconds'], step_data['online_eval_time_seconds']
                ])
    
    def log_task_completion(self, task_id: int, task_time: float, timer: 'TrainingTimer'):
        """Log task completion timing data."""
        timestamp = datetime.now().isoformat()
        metrics = timer.get_training_metrics()
        
        task_data = {
            'timestamp': timestamp,
            'task_id': task_id,
            'task_time_seconds': task_time,
            'task_time_formatted': timer.format_time(task_time),
            'total_epochs': metrics.get('total_epochs', 0),
            'total_steps': metrics.get('total_steps', 0),
            'avg_epoch_time_seconds': metrics.get('avg_epoch_time_seconds', 0),
            'avg_step_time_seconds': metrics.get('avg_step_time_seconds', 0)
        }
        
        self.task_logs.append(task_data)
        
        # Write to CSV
        with open(self.task_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                task_data['timestamp'], task_data['task_id'], task_data['task_time_seconds'],
                task_data['task_time_formatted'], task_data['total_epochs'], task_data['total_steps'],
                task_data['avg_epoch_time_seconds'], task_data['avg_step_time_seconds']
            ])
    
    def log_overall_summary(self, timer: 'TrainingTimer', additional_info: Dict = None):
        """Log overall training summary."""
        timestamp = datetime.now().isoformat()
        metrics = timer.get_training_metrics()
        summary = timer.get_summary()
        
        self.overall_logs = {
            'experiment_name': self.experiment_name,
            'timestamp': timestamp,
            'training_metrics': metrics,
            'timer_summary': summary,
            'additional_info': additional_info or {}
        }
        
        # Write to JSON
        with open(self.summary_log_file, 'w') as f:
            json.dump(self.overall_logs, f, indent=2, default=str)
    
    def generate_analysis_report(self, output_file: str = None) -> str:
        """Generate a comprehensive analysis report."""
        if output_file is None:
            output_file = os.path.join(self.log_dir, f"{self.experiment_name}_analysis_report.txt")
        
        with open(output_file, 'w') as f:
            f.write(f"Training Time Analysis Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Overall statistics
            if self.overall_logs:
                metrics = self.overall_logs.get('training_metrics', {})
                f.write(f"Overall Training Statistics:\n")
                f.write(f"  Total training time: {metrics.get('total_training_time_formatted', 'N/A')}\n")
                f.write(f"  Total epochs: {metrics.get('total_epochs', 'N/A')}\n")
                f.write(f"  Total steps: {metrics.get('total_steps', 'N/A')}\n")
                f.write(f"  Average epoch time: {metrics.get('avg_epoch_time_formatted', 'N/A')}\n")
                f.write(f"  Average step time: {metrics.get('avg_step_time_formatted', 'N/A')}\n")
                f.write(f"  Steps per second: {metrics.get('steps_per_second', 0):.2f}\n\n")
            
            # Task-by-task analysis
            if self.task_logs:
                f.write(f"Task-by-Task Analysis:\n")
                for task_data in self.task_logs:
                    f.write(f"  Task {task_data['task_id']}:\n")
                    f.write(f"    Duration: {task_data['task_time_formatted']}\n")
                    f.write(f"    Epochs: {task_data['total_epochs']}\n")
                    f.write(f"    Steps: {task_data['total_steps']}\n")
                    f.write(f"    Avg epoch time: {task_data['avg_epoch_time_seconds']:.2f}s\n")
                    f.write(f"    Avg step time: {task_data['avg_step_time_seconds']:.4f}s\n\n")
            
            # Performance trends
            if len(self.epoch_logs) > 1:
                f.write(f"Performance Trends:\n")
                first_epoch = self.epoch_logs[0]
                last_epoch = self.epoch_logs[-1]
                f.write(f"  First epoch time: {first_epoch['epoch_time_formatted']}\n")
                f.write(f"  Last epoch time: {last_epoch['epoch_time_formatted']}\n")
                
                if first_epoch['epoch_time_seconds'] > 0:
                    speedup = first_epoch['epoch_time_seconds'] / last_epoch['epoch_time_seconds']
                    f.write(f"  Speedup factor: {speedup:.2f}x\n")
            
            f.write(f"\nDetailed logs available in:\n")
            f.write(f"  Epoch timing: {self.epoch_log_file}\n")
            f.write(f"  Step timing: {self.step_log_file}\n")
            f.write(f"  Task timing: {self.task_log_file}\n")
            f.write(f"  Summary data: {self.summary_log_file}\n")
        
        print(f"Analysis report generated: {output_file}")
        return output_file