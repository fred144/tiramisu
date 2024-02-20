#!/bin/bash
#SBATCH -J gasProperties
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=3999
#SBATCH -t 48:00:00

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
DIR="/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial"
# also tried
# cd /scratch/zt1/project/ricotti-prj/user/fgarcia4/globclustevo/visuals/
# SCRIPT = "./low-sfe.py"
# number of cores specified above
mpirun python3 $SCRIPT $DIR 304 466 1 VSFEGas 2>&1 
