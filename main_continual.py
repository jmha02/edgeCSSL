import copy
import itertools
import subprocess
import sys
import os
import json
import time
import csv
from datetime import datetime


def str_to_dict(command):
    d = {}
    for part, part_next in itertools.zip_longest(command[:-1], command[1:]):
        if part[:2] == "--":
            if part_next[:2] != "--":
                d[part] = part_next
            else:
                d[part] = part
        elif part[:2] != "--" and part_next[:2] != "--":
            part_prev = list(d.keys())[-1]
            if not isinstance(d[part_prev], list):
                d[part_prev] = [d[part_prev]]
            if not part_next[:2] == "--":
                d[part_prev].append(part_next)
    return d


def dict_to_list(command):
    s = []
    for k, v in command.items():
        s.append(k)
        if k != v and v[:2] != "--":
            s.append(v)
    return s


def run_bash_command(args):
    for i, a in enumerate(args):
        if isinstance(a, list):
            args[i] = " ".join(a)
    command = ("python3 main_pretrain.py", *args)
    command = " ".join(command)
    p = subprocess.Popen(command, shell=True)
    p.wait()


def save_continual_timing_logs(checkpoint_dir, task_times, total_time, start_task_idx, num_tasks):
    """Save continual learning timing information to log files."""
    timing_log_dir = os.path.join(checkpoint_dir, 'continual_timing_logs')
    os.makedirs(timing_log_dir, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # Save CSV log
    csv_file = os.path.join(timing_log_dir, 'continual_learning_timing.csv')
    write_header = not os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'timestamp', 'experiment_id', 'task_id', 'task_time_seconds', 
                'task_time_hours', 'cumulative_time_seconds', 'cumulative_time_hours'
            ])
        
        experiment_id = f"{checkpoint_dir.split('/')[-1]}_{timestamp.split('T')[0]}"
        cumulative_time = 0
        
        for i, task_time in enumerate(task_times, start=start_task_idx):
            cumulative_time += task_time
            writer.writerow([
                timestamp, experiment_id, i, task_time, task_time/3600,
                cumulative_time, cumulative_time/3600
            ])
    
    # Save JSON summary
    json_file = os.path.join(timing_log_dir, f'continual_summary_{timestamp.split("T")[0]}.json')
    summary_data = {
        'experiment_info': {
            'timestamp': timestamp,
            'checkpoint_dir': checkpoint_dir,
            'start_task_idx': start_task_idx,
            'num_tasks': num_tasks,
            'total_tasks_completed': len(task_times)
        },
        'timing_summary': {
            'total_continual_time_seconds': total_time,
            'total_continual_time_hours': total_time / 3600,
            'average_task_time_seconds': sum(task_times) / len(task_times) if task_times else 0,
            'average_task_time_hours': (sum(task_times) / len(task_times)) / 3600 if task_times else 0,
            'min_task_time_seconds': min(task_times) if task_times else 0,
            'max_task_time_seconds': max(task_times) if task_times else 0
        },
        'per_task_timing': [
            {
                'task_id': i,
                'time_seconds': task_time,
                'time_hours': task_time / 3600,
                'time_formatted': f"{task_time/3600:.2f}h ({task_time/60:.2f}m)"
            }
            for i, task_time in enumerate(task_times, start=start_task_idx)
        ]
    }
    
    with open(json_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Generate analysis report
    report_file = os.path.join(timing_log_dir, f'continual_analysis_report_{timestamp.split("T")[0]}.txt')
    with open(report_file, 'w') as f:
        f.write("Continual Learning Timing Analysis Report\n")
        f.write("="*50 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Experiment Directory: {checkpoint_dir}\n\n")
        
        f.write("Summary Statistics:\n")
        f.write(f"  Total continual learning time: {total_time/3600:.2f}h ({total_time/60:.2f}m)\n")
        f.write(f"  Number of tasks completed: {len(task_times)}\n")
        if task_times:
            avg_time = sum(task_times) / len(task_times)
            f.write(f"  Average time per task: {avg_time/3600:.2f}h ({avg_time/60:.2f}m)\n")
            f.write(f"  Fastest task: {min(task_times)/3600:.2f}h ({min(task_times)/60:.2f}m)\n")
            f.write(f"  Slowest task: {max(task_times)/3600:.2f}h ({max(task_times)/60:.2f}m)\n")
        
        f.write("\nPer-Task Breakdown:\n")
        for i, task_time in enumerate(task_times, start=start_task_idx):
            f.write(f"  Task {i}: {task_time/3600:.2f}h ({task_time/60:.2f}m)\n")
        
        f.write(f"\nDetailed logs available in:\n")
        f.write(f"  CSV data: {csv_file}\n")
        f.write(f"  JSON summary: {json_file}\n")
    
    print(f"\nContinual learning timing logs saved to: {timing_log_dir}")
    print(f"  - CSV data: {csv_file}")
    print(f"  - JSON summary: {json_file}")
    print(f"  - Analysis report: {report_file}")
    
    return timing_log_dir


if __name__ == "__main__":
    args = sys.argv[1:]
    args = str_to_dict(args)

    # parse args from the script
    num_tasks = int(args["--num_tasks"])
    start_task_idx = int(args.get("--task_idx", 0))
    distill_args = {k: v for k, v in args.items() if "distill" in k}

    # delete things that shouldn't be used for task_idx 0
    args.pop("--task_idx", None)
    for k in distill_args.keys():
        args.pop(k, None)

    # check if this experiment is being resumed
    # look for the file last_checkpoint.txt
    last_checkpoint_file = os.path.join(args["--checkpoint_dir"], "last_checkpoint.txt")
    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file) as f:
            ckpt_path, args_path = [line.rstrip() for line in f.readlines()]
            start_task_idx = json.load(open(args_path))["task_idx"]
            args["--resume_from_checkpoint"] = ckpt_path

    # main task loop - track overall timing
    continual_start_time = time.perf_counter()
    task_times = []
    
    for task_idx in range(start_task_idx, num_tasks):
        print(f"\n{'='*60}")
        print(f"Starting Task {task_idx} of {num_tasks-1}")
        print(f"{'='*60}")
        
        task_start_time = time.perf_counter()

        task_args = copy.deepcopy(args)

        # add pretrained model arg
        if task_idx != 0 and task_idx != start_task_idx:
            task_args.pop("--resume_from_checkpoint", None)
            task_args.pop("--pretrained_model", None)
            assert os.path.exists(last_checkpoint_file)
            ckpt_path = open(last_checkpoint_file).readlines()[0].rstrip()
            task_args["--pretrained_model"] = ckpt_path

        if task_idx != 0 and distill_args:
            task_args.update(distill_args)

        task_args["--task_idx"] = str(task_idx)
        task_args = dict_to_list(task_args)

        run_bash_command(task_args)
        
        task_end_time = time.perf_counter()
        task_duration = task_end_time - task_start_time
        task_times.append(task_duration)
        
        print(f"\n{'='*60}")
        print(f"Task {task_idx} completed in {task_duration/3600:.2f}h ({task_duration/60:.2f}m)")
        print(f"{'='*60}")
    
    # Final timing summary
    continual_end_time = time.perf_counter()
    total_continual_time = continual_end_time - continual_start_time
    
    # Save timing logs to files
    if task_times and args.get("--checkpoint_dir"):
        timing_log_dir = save_continual_timing_logs(
            args["--checkpoint_dir"], 
            task_times, 
            total_continual_time, 
            start_task_idx, 
            num_tasks
        )
    
    print(f"\n{'='*80}")
    print("CONTINUAL LEARNING COMPLETED - OVERALL TIMING SUMMARY")
    print(f"{'='*80}")
    print(f"Total continual learning time: {total_continual_time/3600:.2f}h ({total_continual_time/60:.2f}m)")
    print(f"Number of tasks completed: {len(task_times)}")
    if task_times:
        avg_task_time = sum(task_times) / len(task_times)
        print(f"Average time per task: {avg_task_time/3600:.2f}h ({avg_task_time/60:.2f}m)")
        
        print("\nPer-task timing breakdown:")
        for i, task_time in enumerate(task_times, start=start_task_idx):
            print(f"  Task {i}: {task_time/3600:.2f}h ({task_time/60:.2f}m)")
    print(f"{'='*80}")
