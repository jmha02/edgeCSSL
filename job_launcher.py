import sys
import os
import subprocess
import argparse
from datetime import datetime
import inspect
import itertools

def str_to_dict(command):
    """Fixed version of str_to_dict that handles empty commands properly"""
    d = {}
    if not command:
        return d
    
    for part, part_next in itertools.zip_longest(command[:-1], command[1:]):
        if part is None or part == "" or len(part) < 2:
            continue
            
        if part.startswith("--"):
            if part_next is not None and not part_next.startswith("--"):
                d[part] = part_next
            else:
                d[part] = part
        elif not part.startswith("--") and part_next is not None and not part_next.startswith("--"):
            # Only proceed if we have keys in the dictionary
            if len(d.keys()) > 0:
                part_prev = list(d.keys())[-1]
                if not isinstance(d[part_prev], list):
                    d[part_prev] = [d[part_prev]]
                if not part_next.startswith("--"):
                    d[part_prev].append(part_next)
    return d

parser = argparse.ArgumentParser()
parser.add_argument("--script", type=str, required=True)
parser.add_argument("--mode", type=str, default="normal")
parser.add_argument("--experiment_dir", type=str, default=None)
parser.add_argument("--base_experiment_dir", type=str, default="./experiments")
parser.add_argument("--gpu", type=str, default="v100-16g")
parser.add_argument("--num_gpus", type=int, default=2)
parser.add_argument("--hours", type=int, default=20)
parser.add_argument("--requeue", type=int, default=0)

args = parser.parse_args()

# load file
if os.path.exists(args.script):
    with open(args.script) as f:
        command_lines = [line.strip().strip("\\").strip() for line in f.readlines()]
        # Filter out empty lines and comments
        command_lines = [line for line in command_lines if line and not line.startswith("#")]
        command = command_lines
else:
    print(f"{args.script} does not exist.")
    exit()

# Join all lines and split into arguments
full_command = " ".join(command)
command_parts = full_command.split()

# Find python command and extract arguments
python_idx = -1
for i, part in enumerate(command_parts):
    if "python" in part or "main_continual.py" in part:
        python_idx = i
        break

if python_idx == -1:
    print("Could not find python command in script")
    exit()

# Extract arguments (everything after main_continual.py)
script_args = []
for i, part in enumerate(command_parts[python_idx:]):
    if "main_continual.py" in part:
        script_args = command_parts[python_idx + i + 1:]
        break

assert (
    "--checkpoint_dir" not in script_args
), "Please remove the --checkpoint_dir argument, it will be added automatically"

# collect args
command_args = str_to_dict(script_args)

# create experiment directory
if args.experiment_dir is None:
    args.experiment_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.experiment_dir += f"-{command_args['--name']}"
full_experiment_dir = os.path.join(args.base_experiment_dir, args.experiment_dir)
os.makedirs(full_experiment_dir, exist_ok=True)
print(f"Experiment directory: {full_experiment_dir}")

# add experiment directory to the command
command_with_checkpoint = full_command + f" --checkpoint_dir {full_experiment_dir}"
command = command_with_checkpoint

# run command
if args.mode == "normal":
    p = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stdout)
    p.wait()

elif args.mode == "slurm":
    # infer qos
    if 0 <= args.hours <= 2:
        qos = "qos_gpu-dev"
    elif args.hours <= 20:
        qos = "qos_gpu-t3"
    elif args.hours <= 100:
        qos = "qos_gpu-t4"

    # build slurm command
    command = inspect.cleandoc(
        f"""
        #!/bin/bash
        #SBATCH --job-name {command_args['--name']}
        #SBATCH -C {args.gpu}
        #SBATCH --qos {qos}
        #SBATCH --nodes=1
        #SBATCH --gres gpu:{args.num_gpus}
        #SBATCH --cpus-per-task {int(int(command_args['--num_workers']) * 2 * args.num_gpus)}
        #SBATCH --hint nomultithread
        #SBATCH --time {args.hours}:00:00
        #SBATCH --output outs/{command_args['--name']}.out
        #SBATCH --error outs/{command_args['--name']}.err
        #SBATCH -a 0-{args.requeue}%1

        # cleans out modules loaded in interactive and inherited by default
        module purge

        # loading conda env
        source ~/.bashrc
        conda activate cassle

        # echo of launched commands
        set -x

        cd $WORK/cassle

        # code execution
        {command}
        """
    )

    # write command
    command_path = os.path.join(full_experiment_dir, "command.sh")
    with open(command_path, "w") as f:
        f.write(command)

    # run command
    p = subprocess.Popen(f"sbatch {command_path}", shell=True, stdout=sys.stdout, stderr=sys.stdout)
    p.wait()
