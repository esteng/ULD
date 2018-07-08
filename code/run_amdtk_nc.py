import glob
import os
import numpy as np
import time as systime
import re 
import textgrid as tg
import argparse 

from multiprocessing import cpu_count
from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from ipyparallel import Client
import _pickle as pickle
import random

from amdtk.shared.stats import collect_data_stats, accumulate_stats

import sys
DEBUG = False
resume = None #'/home/esteng/project/esteng/ULD/code/output/models/epoch-0-batch-0'
train=True
import amdtk
import subprocess

# Set random seed
myseed = 4
np.random.seed(myseed)
random.seed(myseed)

# np.seterr(divide='warn', over='warn', under='warn', invalid='raise')

print("successfully completed imports")

def run_amdtk_nc(args):

	elbo = []
	time = []
	def callback(args):
			elbo.append(args['objective'])
			time.append(args['time'])
			print('elbo=' + str(elbo[-1]), 'time=' + str(time[-1]))
			
	rc = Client(profile=args.profile)
	rc.debug = DEBUG
	dview = rc[:]
	print('Connected to', len(dview), 'jobs.')


	print("done importing!")


	audio_dir = os.path.abspath(args.audio_dir)
	print('audio dir:', audio_dir)
	eval_dir = os.path.abspath(args.eval_dir)
	print('eval dir:', eval_dir)
	output_dir = os.path.abspath(args.output_dir)
	print('output dir:', output_dir)

	fea_paths = []
	top_paths = []

	for root, dirs, files in os.walk(audio_dir):
		for file in files:
			print('file:', file)
			if file.lower().endswith(".fea"): 
				fea_paths.append(os.path.join(root,file))
			if file.lower().endswith(".top"):
				top_paths.append(os.path.join(root, file))


	# fea_paths = [os.path.abspath(fname) for fname in glob.glob(fea_path_mask)]
	# top_path_mask = os.path.join(audio_dir,'*top')
	# top_paths = [os.path.abspath(fname) for fname in glob.glob(top_path_mask)]

	for path in fea_paths:
		assert(os.path.exists(path))

	print('fea_paths:', fea_paths)
	print('top_paths:', top_paths)

	zipped_paths = list(zip(sorted(fea_paths), sorted(top_paths)))

	print('zipped_paths:', zipped_paths)

	assert(len(zipped_paths)>0)

	for path_pair in zipped_paths:
		assert(re.sub("\.fea", "", path_pair[0]) == re.sub("\.top", "", path_pair[1]))

	print("There are {} files".format(len(fea_paths)))
	print("Getting mean and variance of input data...")
	data_stats = dview.map_sync(collect_data_stats, fea_paths)

	# Accumulate the statistics over all the utterances.
	final_data_stats = accumulate_stats(data_stats)

	tops = []
	# Read top PLU sequence from file
	for top_path in top_paths:
		with open(top_path, 'r') as f:
			topstring = f.read()
			top_list = topstring.strip().split(',')
			tops += [int(x) for x in top_list]


	num_tops = max(tops)+1

	elbo = []
	time = []
	
	print("Creating phone loop model...")
	conc = 0.1


	if resume == None:

		model = amdtk.PhoneLoopNoisyChannel.create(
			n_units=str(args.bottom_plu_count),  # number of acoustic units
			n_states=3,   # number of states per unit
			n_comp_per_state=args.n_comp,   # number of Gaussians per emission
			n_top_units=num_tops, # size of top PLU alphabet
			mean=np.zeros_like(final_data_stats['mean']), 
			var=np.ones_like(final_data_stats['var']/100),
			max_slip_factor=args.max_slip_factor,
			extra_cond=args.extra_cond
		)
		
	else:
		with open(resume, 'rb') as f1:
			model = pickle.load(f1)

		numgex = re.compile("[\d]+")
		res_epoch, res_batch = [int(x) for x in numgex.findall(str(resume))]
		model.start_epoch=res_epoch
		model.starting_batch=res_batch
	if train: 
		if not os.path.exists(os.path.join(output_dir, "models")):
			os.mkdir(os.path.join(output_dir, "models"))

		data_stats= final_data_stats
		print("Creating VB optimizer...")   
		optimizer = amdtk.NoisyChannelOptimizer(
			dview, 
			data_stats, 
			args= {'epochs': args.n_epochs,
			 'batch_size': args.batch_size,
			 'lrate': args.lrate,
			 'output_dir': output_dir,
			 'audio_dir': audio_dir,
			 'eval_audio_dir': audio_dir,
			 'audio_samples_per_sec': 100},
			model=model,
			pkl_path=os.path.join(output_dir, "models")
		)

		print("Running VB optimization...")
		begin = systime.time()
		print("running with {} paths".format(len(list(zipped_paths))))
		optimizer.run(zipped_paths, callback)
		end = systime.time()
		print("VB optimization took ",end-begin," seconds.")

	print("***ELBO***")
	for i, n in enumerate(elbo):
		print('Epoch '+str(i)+': ELBO='+str(n))

	if args.decode:
		for (fea_path, top_path) in zipped_paths:
			data = amdtk.read_htk(fea_path)

			# Normalize the data
			data_mean = np.mean(data)
			data_var = np.var(data)
			data = (data-data_mean)/np.sqrt(data_var)

			# Read top PLU sequence from file
			with open(top_path, 'r') as f:
				topstring = f.read()
				tops = topstring.strip().split(',')
				tops = [int(x) for x in tops]

			#result = model.decode(data, tops, state_path=False)
			#result_path = model.decode(data, tops, state_path=True)
			(result_intervals, edit_path, _) = model.decode(data, tops, phone_intervals=True, edit_ops=True)

			print("---")
			print("Phone sequence for file", fea_path, ":")
			print(result_intervals)
			print(edit_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("--bottom_plu_count",  help="number of bottom PLUs", required=True, type=int)
	parser.add_argument("--max_slip_factor",  help="what fraction to allow top & bottom PLU positions to differ by", required=True, type=float)
	parser.add_argument("--n_comp",  help="number of Gaussian components per state", required=True, type=int)
	parser.add_argument("--n_epochs",  help="number of epochs of training to run", default=1, type=int)
	parser.add_argument("--audio_dir",  help="location of audio files to train on", required=True)
	parser.add_argument("--eval_dir",  help="location of audio files to evaluate on", required=True)
	parser.add_argument("--output_dir",  help="where to put evaluations and checkpoints files", required=True)
	parser.add_argument("--batch-size", default=cpu_count(), type=int)
	parser.add_argument("--profile", help="ipyparallel profile to connect to", default="default")
	parser.add_argument("--decode", help="flag to decode or not", action="store_true", default=False)
	parser.add_argument("--lrate", help="set the learn rate", default=0.01, type=float)
	parser.add_argument("--extra_cond", help="set to true for bigram conditioning", action="store_true", default=True)
	args = parser.parse_args()

	run_amdtk_nc(args)
