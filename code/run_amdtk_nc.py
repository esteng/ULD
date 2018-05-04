import glob
import os
import numpy as np
import time as systime
import re 
import textgrid as tg

from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from ipyparallel import Client
import _pickle as pickle

import sys
# sys.path.insert(0, './amdtk')
# sys.path.append("/Users/Elias/ULD/code/amdtk")
# DEBUG = True
DEBUG = True
# resume = "/Users/Elias/ULD/code/models/epoch-2-batch-4"
resume = None
train=True
# resume=None
import amdtk
import subprocess


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
        'var': var/100
    }
    return data_stats

# print("starting engines")
# njobs = 2
# subprocess.Popen(['ipcluster', 'start',' --profile', 'default',' -n', str(njobs), '--daemonize'])
# subprocess.Popen(['sleep', '10']).communicate()


rc = Client(profile='default')
rc.debug = DEBUG
dview = rc[:]
print('Connected to', len(dview), 'jobs.')


print("done importing!")
# audio_dir = '../audio/icicles'
audio_dir = '../audio/FAEM0'

audio_dir = os.path.abspath(audio_dir)

# fea_path_mask = os.path.join(audio_dir, '*fea')

fea_paths = []
top_paths = []

for root, dirs, files in os.walk(audio_dir):
	for file in files:
		if file.lower().endswith(".fea"): 
			fea_paths.append(os.path.join(root,file))
		if file.lower().endswith(".top"):
			top_paths.append(os.path.join(root, file))


# fea_paths = [os.path.abspath(fname) for fname in glob.glob(fea_path_mask)]
# top_path_mask = os.path.join(audio_dir,'*top')
# top_paths = [os.path.abspath(fname) for fname in glob.glob(top_path_mask)]

for path in fea_paths:
    assert(os.path.exists(path))

zipped_paths = list(zip(sorted(fea_paths), sorted(top_paths)))

assert(len(zipped_paths)>0)

for path_pair in zipped_paths:
    print(path_pair)
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
def callback(args):
    elbo.append(args['objective'])
    time.append(args['time'])
    print('elbo=' + str(elbo[-1]), 'time=' + str(time[-1]))
 

print("Creating phone loop model...")
conc = 0.1


if resume == None:

    model = amdtk.PhoneLoopNoisyChannel.create(
        n_units=20,  # number of acoustic units
        n_states=3,   # number of states per unit
        n_comp_per_state=4,   # number of Gaussians per emission
        n_top_units=num_tops, # size of top PLU alphabet
        mean=np.zeros_like(final_data_stats['mean']), 
        var=np.ones_like(final_data_stats['var']),
        max_slip_factor=.05
        #concentration=conc
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
	    args= {'epochs': 3,
	     'batch_size': 4,
	     'lrate': 0.01,
	     'pkl_path': "models/",
	     'log_dir': 'logs'},
	    model=model,

	)

	print("Running VB optimization...")
	begin = systime.time()
	print("running with {} paths".format(len(list(zipped_paths))))
	optimizer.run(zipped_paths, callback)
	end = systime.time()
	print("VB optimization took ",end-begin," seconds.")

# fig1 = figure(
#     x_axis_label='time (s)', 
#     y_axis_label='ELBO',
#     width=400, 
#     height=400
# )
# x = np.arange(0, len(elbo), 1)
# fig1.line(x, elbo)
#show(fig1)

print("\nDECODING\n")

date_string = systime.strftime("textgrids_%Y-%m-%d_%H:%M")

# Need to change this according to 
samples_per_sec = 100


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
    # (result_intervals, edit_path, hmm_i) = model.decode(data, tops, phone_intervals=True, edit_ops=True)
    (result_intervals, edit_path, _) = model.decode(data, tops, phone_intervals=True, edit_ops=True)
    print("---")
    print("Phone sequence for file", fea_path, ":")
    print(result_intervals)
    print(edit_path)

    #print(result_intervals)
    
    # counts_by_number = {}
    # counts_by_duration = {}
    # for item in result_intervals:
    #     phone = item[0]
    #     dur = item[2]-item[1]
    #     counts_by_number[phone] = counts_by_number.get(phone, 0) + 1
    #     counts_by_duration[phone] = counts_by_duration.get(phone, 0) + dur/samples_per_sec

    # print("\nTotal number of phones: ",len(counts_by_number.keys()))
    # print("Concentration multiplier: ", conc)

    # print("\nPhone counts by number: ")
    # print_bar_graph(counts_by_number)
    # print("\nPhone counts by duration: ")
    # print_bar_graph(counts_by_duration)


    write_textgrids = True

    if write_textgrids:
        output_dir = os.path.join(audio_dir, date_string)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        amdtk.utils.write_textgrid(result_intervals, 
                                    samples_per_sec, 
                                    os.path.join(output_dir, os.path.split(fea_path)[1][:-4]+'.TextGrid'))

        print("Wrote textgrids to", output_dir)

print("success")
subprocess.Popen(['ipcluster' ,'stop', '--profile', 'default'])


