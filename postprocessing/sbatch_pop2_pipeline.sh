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
 
SCRIPT="./pop2_pipeline.py"
DIR="/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial"
# also tried
# cd /scratch/zt1/project/ricotti-prj/user/fgarcia4/globclustevo/visuals/
# SCRIPT = "./low-sfe.py"
# number of cores specified above
mpirun -n 1 python3 $SCRIPT $DIR 153 466 1 2>&1 
