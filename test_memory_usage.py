#!/usr/bin/env python3
"""
Test script to compare memory usage between LoRA and full fine-tuning for ViT models.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def run_short_training(use_lora=False, max_epochs=2, batch_size=64):
    """
    Run a short training session and capture memory usage.
    
    Args:
        use_lora: Whether to use LoRA
        max_epochs: Number of epochs to train
        batch_size: Batch size for training
        
    Returns:
        Path to log file with memory usage information
    """
    
    # Activate conda environment first
    conda_setup = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate cassle"
    
    # Create temporary directory for this test
    test_dir = tempfile.mkdtemp(prefix=f"memory_test_{'lora' if use_lora else 'full'}_")
    log_file = os.path.join(test_dir, "training.log")
    
    print(f"\n{'='*60}")
    print(f"Running {'LoRA' if use_lora else 'Full'} training test...")
    print(f"Epochs: {max_epochs}, Batch Size: {batch_size}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}")
    
    # Build the command
    cmd_parts = [
        conda_setup,
        "&&",
        "cd /root/cassle",
        "&&", 
        "python3 main_pretrain.py",
        "--dataset cifar100",
        "--encoder vit_small", 
        "--data_dir ./data",
        "--train_dir train",
        "--val_dir val",
        f"--max_epochs {max_epochs}",
        "--devices 1",
        "--accelerator gpu",
        "--precision 16",
        "--optimizer sgd",
        "--lr 0.1",  # Lower LR for short test
        "--classifier_lr 0.05",
        "--weight_decay 1e-4",
        f"--batch_size {batch_size}",
        "--num_workers 4",
        "--brightness 0.4",
        "--contrast 0.4",
        "--saturation 0.2",
        "--hue 0.1",
        "--gaussian_prob 1.0 0.1",
        "--solarization_prob 0.0 0.2", 
        "--min_scale 0.08",
        "--size 32",
        "--num_crops 2",
        f"--name memory-test-{'lora' if use_lora else 'full'}",
        "--project cassle-memory-test",
        "--method dino",
        "--proj_hidden_dim 512",  # Smaller to fit in memory
        "--output_dim 128",
        "--num_prototypes 1024",  # Smaller to fit in memory
        "--student_temperature 0.1",
        "--teacher_temperature 0.07",
        "--warmup_teacher_temperature 0.04",
        "--warmup_teacher_temperature_epochs 1",  # Short warmup
        "--clip_grad 3.0",
        "--freeze_last_layer 1",
        "--base_tau_momentum 0.996",
        "--final_tau_momentum 1.0",
        "--scheduler cosine",
        "--task_idx 0",
        "--split_strategy data",
        "--num_tasks 1",
        "--offline",  # Don't log to wandb
        f"--default_root_dir {test_dir}",
    ]
    
    if use_lora:
        cmd_parts.extend([
            "--use_lora",
            "--lora_r 8",  # Smaller rank for test
            "--lora_alpha 16",
            "--lora_dropout 0.1"
        ])
        
    # Add logging redirection
    cmd_parts.extend([f"2>&1 | tee {log_file}"])
    
    # Join command
    cmd = " ".join(cmd_parts)
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(f"Training completed with return code: {result.returncode}")
        return log_file, test_dir
        
    except subprocess.TimeoutExpired:
        print("Training timed out after 5 minutes")
        return log_file, test_dir
    except Exception as e:
        print(f"Error during training: {e}")
        return log_file, test_dir


def extract_memory_stats(log_file):
    """Extract memory statistics from log file."""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return {}
        
    memory_stats = {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract key memory information
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract model memory footprint
            if "Total Model:" in line and "MB" in line:
                try:
                    value = float(line.split("Total Model:")[1].split("MB")[0].strip())
                    memory_stats['model_memory_mb'] = value
                except:
                    pass
                    
            # Extract peak GPU memory
            if "Peak GPU Memory:" in line and "MB" in line:
                try:
                    value = float(line.split("Peak GPU Memory:")[1].split("MB")[0].strip())
                    memory_stats['peak_gpu_memory_mb'] = value
                except:
                    pass
                    
            # Extract trainable parameters info
            if "Trainable params:" in line:
                try:
                    value_str = line.split("Trainable params:")[1].strip().replace(',', '')
                    memory_stats['trainable_params'] = int(value_str)
                except:
                    pass
                    
            # Extract total parameters
            if "All params:" in line:
                try:
                    value_str = line.split("All params:")[1].strip().replace(',', '')
                    memory_stats['total_params'] = int(value_str)
                except:
                    pass
                    
            # Extract trainable percentage
            if "Trainable:" in line and "%" in line:
                try:
                    value = float(line.split("Trainable:")[1].split("%")[0].strip())
                    memory_stats['trainable_percentage'] = value
                except:
                    pass
                    
        # Look for final memory report
        if "FINAL MEMORY USAGE REPORT" in content:
            report_start = content.find("FINAL MEMORY USAGE REPORT")
            report_section = content[report_start:report_start+1000]  # Get next 1000 chars
            
            for line in report_section.split('\n'):
                if "Peak Allocated:" in line and "MB" in line:
                    try:
                        value = float(line.split("Peak Allocated:")[1].split("MB")[0].strip())
                        memory_stats['final_peak_gpu_mb'] = value
                    except:
                        pass
                        
        print(f"Extracted memory stats: {memory_stats}")
        return memory_stats
        
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return {}


def compare_memory_usage():
    """Run both LoRA and full training tests and compare memory usage."""
    
    print("\n" + "="*80)
    print("MEMORY USAGE COMPARISON TEST")
    print("="*80)
    print("This test will run short training sessions with and without LoRA")
    print("to measure and compare peak memory usage.")
    print("="*80)
    
    # Test parameters
    max_epochs = 2
    batch_size = 32  # Small batch size to avoid OOM
    
    # Run tests
    tests = [
        {"name": "Full Training", "use_lora": False},
        {"name": "LoRA Training", "use_lora": True},
    ]
    
    results = {}
    
    for test in tests:
        print(f"\n🧪 Starting {test['name']} test...")
        log_file, test_dir = run_short_training(
            use_lora=test['use_lora'],
            max_epochs=max_epochs,
            batch_size=batch_size
        )
        
        # Extract memory statistics
        memory_stats = extract_memory_stats(log_file)
        results[test['name']] = {
            'memory_stats': memory_stats,
            'log_file': log_file,
            'test_dir': test_dir
        }
        
        print(f"✅ {test['name']} test completed")
        
        # Clean up test directory
        try:
            shutil.rmtree(test_dir)
            print(f"🗑️  Cleaned up {test_dir}")
        except Exception as e:
            print(f"⚠️  Could not clean up {test_dir}: {e}")
    
    # Print comparison
    print("\n" + "="*80)
    print("MEMORY USAGE COMPARISON RESULTS")
    print("="*80)
    
    for test_name, data in results.items():
        print(f"\n{test_name}:")
        memory_stats = data['memory_stats']
        
        if 'model_memory_mb' in memory_stats:
            print(f"  Model Memory: {memory_stats['model_memory_mb']:.1f} MB")
            
        if 'peak_gpu_memory_mb' in memory_stats:
            print(f"  Peak GPU Memory: {memory_stats['peak_gpu_memory_mb']:.1f} MB")
        elif 'final_peak_gpu_mb' in memory_stats:
            print(f"  Peak GPU Memory: {memory_stats['final_peak_gpu_mb']:.1f} MB")
            
        if 'trainable_params' in memory_stats and 'total_params' in memory_stats:
            print(f"  Trainable Params: {memory_stats['trainable_params']:,}")
            print(f"  Total Params: {memory_stats['total_params']:,}")
            
        if 'trainable_percentage' in memory_stats:
            print(f"  Trainable: {memory_stats['trainable_percentage']:.2f}%")
    
    # Calculate differences if both tests have data
    if len(results) == 2:
        full_stats = results['Full Training']['memory_stats']
        lora_stats = results['LoRA Training']['memory_stats']
        
        print(f"\n🔍 COMPARISON:")
        
        # Memory comparison
        if 'peak_gpu_memory_mb' in full_stats and 'peak_gpu_memory_mb' in lora_stats:
            full_gpu = full_stats['peak_gpu_memory_mb'] 
            lora_gpu = lora_stats['peak_gpu_memory_mb']
            savings = full_gpu - lora_gpu
            percentage_savings = (savings / full_gpu) * 100 if full_gpu > 0 else 0
            
            print(f"  GPU Memory Savings: {savings:.1f} MB ({percentage_savings:.1f}%)")
            
        # Parameter comparison  
        if 'trainable_params' in full_stats and 'trainable_params' in lora_stats:
            full_params = full_stats['trainable_params']
            lora_params = lora_stats['trainable_params']
            param_reduction = full_params - lora_params
            param_percentage = (param_reduction / full_params) * 100 if full_params > 0 else 0
            
            print(f"  Trainable Parameter Reduction: {param_reduction:,} ({param_percentage:.1f}%)")
    
    print("="*80)
    return results


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("main_pretrain.py"):
        print("❌ Error: main_pretrain.py not found. Please run from the cassle directory.")
        sys.exit(1)
        
    # Check if conda is available
    conda_check = subprocess.run(
        "source /root/miniconda3/etc/profile.d/conda.sh && conda --version", 
        shell=True, capture_output=True, text=True
    )
    
    if conda_check.returncode != 0:
        print("❌ Error: Conda not found. Please ensure conda is properly installed.")
        sys.exit(1)
        
    # Run the comparison
    results = compare_memory_usage()