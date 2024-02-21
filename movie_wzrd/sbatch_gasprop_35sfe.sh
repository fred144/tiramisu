#!/bin/bash
#SBATCH -J 70sfegasProperties
#SBATCH --ntasks=1
#SBATCH -t 72:00:00
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
 
SCRIPT="/scratch/zt1/project/ricotti-prj/user/fgarcia4/tiramisu/movie_wzrd/gas_properties.py"
DIR="/afs/shell.umd.edu/project/ricotti-prj/user/fgarcia4/dwarf/data/cluster_evolution/fs035_ms10"
# also tried
# cd /scratch/zt1/project/ricotti-prj/user/fgarcia4/globclustevo/visuals/
# SCRIPT = "./low-sfe.py"
# number of cores specified above
mpirun -n 1 python3 $SCRIPT $DIR 170 1607 1 35SFEGas 2>&1 
