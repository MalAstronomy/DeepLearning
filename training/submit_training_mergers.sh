#!/bin/bash
#
#SBATCH --job-name=merger-test
#SBATCH --output=./alan-log/slurm-%j.out
#SBATCH --time="2-00:00:00" 
#SBATCH --gres=gpu:0
##SBATCH --exclude=alan-compute-[06-09]
##SBATCH --nodelist=alan-compute-02
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=6G

# print the name of the node on which the job is running:
echo "Job running on $SLURMD_NODENAME"

### Copy the data to the Alan machine: ###

HOME_DIR="/home/mquesnel/Courses/DeepLearning/datasets/density70"

NODE_DIR="/scratch/mquesnel/DeepLearningProject/"
mkdir -p $NODE_DIR # Create the directory if it doesn't already exist

FILE_NAME="merger_train_dens795_sqtscaled_no28_flip.h5"

FILE_LOC_NODE="$NODE_DIR$FILE_NAME"

if ! [ -f "$FILE_LOC_NODE" ]; then
    echo "File "$FILE_LOC_NODE" does not already exists"
    SECONDS=0
    cp -r "$HOME_DIR/$FILE_NAME" "$NODE_DIR"
    duration=$SECONDS
    echo "File copied in $(($duration / 60))m$(($duration % 60))s"
    else
        echo "File "$FILE_LOC_NODE" already exists"
fi 
      
### Activate the virtual environment: ###
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py3_env

### Go to the proper directory: ###
cd /home/mquesnel/Courses/DeepLearning/training/

### Run the python script: ###
echo "Running the training script"
python training_mergers.py -floc $FILE_LOC_NODE -dla "mlp" -dsiz 795 -opt "Adam" -bs 32 -lr '1e-4' -nep 100 -splt "90" -met 'mse'