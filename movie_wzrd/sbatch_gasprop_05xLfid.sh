#!/bin/bash
#SBATCH -J gasProp-0.5xLfid
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3999
#SBATCH -t 72:00:00

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
 
SCRIPT="/scratch/zt1/project/ricotti-prj/user/fgarcia4/tiramisu/movie_wzrd/gas_properties.py"
DIR="/scratch/zt1/project/ricotti-prj/user/ricotti/CosmicGems/CC_0.5x_Salp"
# also tried
# cd /scratch/zt1/project/ricotti-prj/user/fgarcia4/globclustevo/visuals/
# SCRIPT = "./low-sfe.py"
# number of cores specified above
mpirun python3 $SCRIPT $DIR 0 84 1 0.5LfidGasProperties 2>&1 
