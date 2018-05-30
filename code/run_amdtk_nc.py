import glob
import os
import numpy as np
import time as systime
import re 
import textgrid as tg
import argparse 


from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from ipyparallel import Client
import _pickle as pickle
import random

import sys
DEBUG = True
resume = "/Users/Elias/ULD/code/models/epoch-0-batch-0"
# resume = None
# train=True
train=False
# resume=None
import amdtk
import subprocess

# Set random seed
myseed = 4
np.random.seed(myseed)
random.seed(myseed)

# np.seterr(divide='warn', over='warn', under='warn', invalid='raise')

print("successfully completed imports")


def collect_data_stats(filename):
	"""Job to collect the statistics."""
	# We  re-import this module here because this code will run
	# remotely.
	import amdtk
	data = amdtk.read_htk(filename)
	stats_0 = data.shape[0]
	stats_1 = data.sum(axis=0)
	stats_2 = (data**2).sum(axis=0)
	retval = (
		stats_0,
		stats_1,
		stats_2
	)
	return retval


def accumulate_stats(data_stats):
	n_frames = data_stats[0][0]
	mean = data_stats[0][1]
	var = data_stats[0][2]
	for stats_0, stats_1, stats_2 in data_stats[1:]:
		n_frames += stats_0
		mean += stats_1
		var += stats_2
	mean /= n_frames
	var = (var / n_frames) - mean**2

	data_stats = {
		'count': n_frames,
		'mean': mean,
		'var': var/20
	}
	return data_stats

# njobs = 2
# print("starting ipyparallel cluster with "+str(njobs)+" engines")
# subprocess.Popen(['ipcluster', 'start',' --profile', 'default',' -n', str(njobs), '--daemonize'])
# subprocess.Popen(['sleep', '6']).communicate()
def callback(args):
		elbo.append(args['objective'])
		time.append(args['time'])
		print('elbo=' + str(elbo[-1]), 'time=' + str(time[-1]))
	 

def run_amdtk_nc(num_bottom_plus, num_epochs, audio_dir, eval_dir, output_dir):

	rc = Client(profile='default')
	rc.debug = DEBUG
	dview = rc[:]
	print('Connected to', len(dview), 'jobs.')


# print("done importing!")
# audio_dir = '../audio/icicles'
# # audio_dir = '../audio/FAEM0'


	print("done importing!")

	# audio_dir = '../audio/icicles'
	#audio_dir = '../audio/FAEM0'
	#audio_dir = '../audio/TIMIT/FAEM0'

	audio_dir = os.path.abspath(audio_dir)
	print('audio dir:', audio_dir)
	eval_dir = os.path.abspath(eval_dir)
	print('eval dir:', eval_dir)
	output_dir = os.path.abspath(output_dir)
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
			n_units=str(num_bottom_plus),  # number of acoustic units
			n_states=3,   # number of states per unit
			n_comp_per_state=4,   # number of Gaussians per emission
			n_top_units=str(num_tops), # size of top PLU alphabet
			mean=np.zeros_like(final_data_stats['mean']), 
			var=np.ones_like(final_data_stats['var']/100),
			max_slip_factor=.1
		)
		 
	else:
		with open(resume, 'rb') as f1:
			model = pickle.load(f1)

	if train: 
		data_stats= final_data_stats
		print("Creating VB optimizer...")   
		optimizer = amdtk.NoisyChannelOptimizer(
			dview, 
			data_stats, 
			args= {'epochs': num_epochs,
			 'batch_size': 4,
			 'lrate': 0.01,
			 'output_dir': output_dir,
			 'audio_dir': audio_dir,
			 'eval_audio_dir': audio_dir,
			 'audio_samples_per_sec': 100},
			model=model,
			pkl_path="models"

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



# if __name__ == '__main__':
	# parser = argparse.ArgumentParser()

	# parser.add_argument("bottom_plu_count",  help="number of bottom PLUs")
	# parser.add_argument("n_epochs",  help="number of epochs of training to run")
	# parser.add_argument("audio_dir",  help="location of audio files to train on")
	# parser.add_argument("eval_dir",  help="location of audio files to evaluate on")
	# parser.add_argument("output_dir",  help="where to put evaluation files")
	# args = parser.parse_args()
	# print(args.bottom_plu_count, args.audio_dir, args.eval_dir, args.output_dir)


# run_amdtk_nc(int(args.bottom_plu_count), int(args.n_epochs), args.audio_dir, args.eval_dir, args.output_dir)
num_bottom_plus = 20
num_epochs = 2
audio_dir = "../audio/icicles"
eval_dir= "../audio/icicles"
output_dir = "output/icicles"

rc = Client(profile='default')
rc.debug = DEBUG
dview = rc[:]
print('Connected to', len(dview), 'jobs.')


print("done importing!")

# audio_dir = '../audio/icicles'
#audio_dir = '../audio/FAEM0'
#audio_dir = '../audio/TIMIT/FAEM0'

audio_dir = os.path.abspath(audio_dir)
print('audio dir:', audio_dir)

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
		n_units=num_bottom_plus,  # number of acoustic units
		n_states=3,   # number of states per unit
		n_comp_per_state=4,   # number of Gaussians per emission
		n_top_units=num_tops, # size of top PLU alphabet
		mean=np.zeros_like(final_data_stats['mean']), 
		var=np.ones_like(final_data_stats['var']),
		max_slip_factor=.1
	)
	 
else:
	with open(resume, 'rb') as f1:
		model = pickle.load(f1)

if train: 
	data_stats= final_data_stats
	print("Creating VB optimizer...")   
	optimizer = amdtk.NoisyChannelOptimizer(
		dview, 
		data_stats, 
		args= {'epochs': num_epochs,
		 'batch_size': 4,
		 'lrate': 0.01,
		 'output_dir': 'output',
		 'audio_dir': audio_dir,
		 'eval_audio_dir': audio_dir,
		 'audio_samples_per_sec': 100},
		model=model,
		pkl_path="models"

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

