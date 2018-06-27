#!/bin/bash
#SBATCH --account=def-msonde1
#SBATCH --ntasks=64             # number of MPI processes
#SBATCH --mem-per-cpu=64G      # memory; default unit is megabytes
#SBATCH --time=00-12:00          # time (DD-HH:MM)

cd amdtk;
python setup.py install;
cd ..;
profile=job_${SLURM_JOB_ID}_$(hostname)
echo "CREATING PROFILE ${profile}";
ipython profile create ${profile};


echo "STARTING ENGINES";
ipcontroller --ip='*' --profile=${profile} --log-to-file &
sleep 10;
srun ipengine --profile=${profile} --location=$(hostname) --log-to-file & 
sleep 45
echo "ENGINES STARTED";
python -u run_amdtk_nc.py --bottom_plu_count=50 --n_epochs=10 --audio_dir=../audio/TRAIN/ --eval_dir=../audio/TRAIN/ --output_dir=output --batch-size=64 --profile ${profile}
echo "DONE";
ipcluster stop;
