#!/bin/bash
#SBATCH -J gas_renderer
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3999
#SBATCH -t 24:00:00

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
 
SCRIPT="/scratch/zt1/project/ricotti-prj/user/fgarcia4/tiramisu/movie_wzrd/density_temp_projection.py"
DIR="/scratch/zt1/project/ricotti-prj/user/fgarcia4/ramses/galaxies/data/cluster_evolution/haloD_varSFE_Lfid_Salp_ks20231024"
# also tried
# cd /scratch/zt1/project/ricotti-prj/user/fgarcia4/globclustevo/visuals/
# SCRIPT = "./low-sfe.py"
# number of cores specified above
mpirun python3 $SCRIPT $DIR 64 220 1 LfidSF 2>&1 
