#!/bin/bash
#SBATCH -J vsfe_gasProperties
#SBATCH --ntasks=1
#SBATCH -t 48:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-user=g.fred@columbia.edu
#SBATCH --mail-type=ALL

#. ~/.bashrc
. ~/.profile

# designed to be submitted/ ran in the script directory
export LANG=en_US
module purge
module load python
module load gcc
module load openmpi
# activate py environment
source ~/scratch/master/bin/activate
 
SCRIPT="./gas_pipeline.py"
DIR="/scratch/zt1/project/ricotti-prj/user/fgarcia4/sim_data/cluster_evolution/fs035_ms10/"
# also tried
# cd /scratch/zt1/project/ricotti-prj/user/fgarcia4/globclustevo/visuals/
# SCRIPT = "./low-sfe.py"
# number of cores specified above
tmpirun -n 1 python3 $SCRIPT $DIR 200 1606 1 2>&1 
