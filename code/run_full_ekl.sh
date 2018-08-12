cd amdtk;
python3 setup.py install;
cd ..;
#profile=job_${SLURM_JOB_ID}_$(hostname)
#echo "CREATING PROFILE ${profile}";
#ipython profile create ${profile};


echo "STARTING ENGINES";
#ipcontroller --ip='*' --profile=${profile} --log-to-file &
ipcluster start --profile default -n 2 --daemonize;
#srun ipengine --profile=${profile} --location=$(hostname) --log-to-file & 
sleep 10;
echo "ENGINES STARTED";
python3 -u run_amdtk_nc.py --bottom_plu_count=80 --max_slip_factor=0.2 --n_epochs=3 --audio_dir=../audio/short/ --eval_dir=../audio/short/ --output_dir=output --batch-size=2
echo "DONE";
ipcluster stop;
