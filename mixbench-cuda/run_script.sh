#!/bin/bash

# Function to handle the interrupt signal (Ctrl+C)
handle_sigint() {
  echo "Interrupt received, cleaning up..."
  # Kill all child processes of the current script
  kill -- -$$ 2>/dev/null
  exit 1
}

# Set the trap to call handle_sigint on SIGINT
trap handle_sigint SIGINT

unbuffer nvidia-smi --query-gpu=index,timestamp,power.draw,clocks.sm,clocks.mem,clocks.gr --format=csv -l 1 -i 0 | tee -a power_fp64_zero.txt & 
for idx in {1..1};

    do
        date
        OMP_NUM_THREADS=1 unbuffer srun -u --gpus-per-task=1 -N1 -n1 -c1 ./mixbench-cuda 0 0 1 | tee -a power_fp64_zero.txt &
        process_pid=$!
        sleep 1
    done
wait $process_pid
killall nvidia-smi
killall python3
exit 0
echo "Script running, press Ctrl+C to stop"
wait
echo "Script finished"

