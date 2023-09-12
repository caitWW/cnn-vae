#!/bin/bash
#SBATCH --job-name=vae    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=qw3971@princeton.edu

module load anaconda3/2022.5
conda activate torch-env
module load cudatoolkit/11.7

python3 train_vae.py -mn test_run --dataset_root /home/qw3971/clevr/image_generation/run2_transform/ --save_dir /home/qw3971/cnn-vae/run2_saccade/ --target_root /home/qw3971/clevr/image_generation/run2_transform_shifted/ --nepoch 50
