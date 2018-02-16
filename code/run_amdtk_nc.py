import glob
import os
import numpy as np
import time as systime

from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from ipyparallel import Client


import sys
sys.path.insert(0, 'amdtk')

import amdtk

print("successfully completed imports")
amdtk.utils.test_import()

# I assume the reason this is parallelized is because
# for large amounts of data it could be very slow?
def collect_data_stats(filename):
    print("filename:", filename)
    """Job to collect the statistics."""
    # We  re-import this module here because this code will run
    # remotely.

    import sys 
    sys.path.append("./amdtk")
    import amdtk
    data = amdtk.read_htk(filename)

    stat_length = data.shape[0]
    stat_sum = data.sum(axis=0)
    stat_squared_sum = (data**2).sum(axis=0)
    return (
        stat_length,
        stat_sum,
        stat_squared_sum
    )

def accumulate_stats(list_of_stats):
    n_frames = 0
    mean = 0
    var = 0
    for stat_length, stat_sum, stat_squared_sum in list_of_stats:
        n_frames += stat_length
        mean += stat_sum
        var += stat_squared_sum
    mean /= n_frames
    var = (var / n_frames) - mean**2

    return {
        'count': n_frames,
        'mean': mean,
        'var': var
    }

# fname = '/path/to/features.fea'
# fname_vad = fname + '[10:100]'
# data = amdtk.read_htk(fname_vad)

# path = '../audio_test/falr0_sx425.fea'
# data = amdtk.read_htk(path)


rc = Client(profile='default')
dview = rc[:]
print('Connected to', len(dview), 'jobs.')

with dview.sync_imports():
    import sys
    sys.path.append("./amdtk")
    import amdtk


fea_path_mask = '../audio/icicles/*fea'
fea_paths = [os.path.abspath(fname) for fname in glob.glob(fea_path_mask)]
top_path_mask = '../audio/icicles/*top'
top_paths = [os.path.abspath(fname) for fname in glob.glob(top_path_mask)]

for path in fea_paths:
    assert(os.path.exists(path))

zipped_paths = list(zip(fea_paths, top_paths))

assert(len(zipped_paths)>0)

for path_pair in zipped_paths:
    print(path_pair)

print("Getting mean and variance of input data...")
data_stats = dview.map_sync(collect_data_stats, fea_paths)

# Accumulate the statistics over all the utterances.
final_data_stats = accumulate_stats(data_stats)


# Read top PLU sequence from file
with open(top_paths[0], 'r') as f:
    topstring = f.read()
    tops = topstring.strip().split(',')
    tops = [int(x) for x in tops]

num_tops = max(tops)+1




elbo = []
time = []
def callback(args):
    elbo.append(args['objective'])
    time.append(args['time'])
    print('elbo=' + str(elbo[-1]), 'time=' + str(time[-1]))
 

print("Creating phone loop model...")
conc = 0.1
model = amdtk.PhoneLoopNoisyChannel.create(
    n_units=20,  # number of acoustic units
    n_states=3,   # number of states per unit
    n_comp_per_state=4,   # number of Gaussians per emission
    n_top_units=num_tops, # size of top PLU alphabet
    mean=np.zeros_like(final_data_stats['mean']), 
    var=np.ones_like(final_data_stats['var']) #,
    #concentration=conc
)
 
print("Creating VB optimizer...")   
optimizer = amdtk.NoisyChannelOptimizer(
    dview, 
    final_data_stats, 
    args= {'epochs': 3,
     'batch_size': 400,
     'lrate': 0.01},
    model=model,

)

print("Running VB optimization...")
begin = systime.time()
optimizer.run(zipped_paths, callback)
end = systime.time()
print("VB optimization took ",end-begin," seconds.")

fig1 = figure(
    x_axis_label='time (s)', 
    y_axis_label='ELBO',
    width=400, 
    height=400
)
x = np.arange(0, len(elbo), 1)
fig1.line(x, elbo)
#show(fig1)

print("\nDECODING\n")

date_string = systime.strftime("textgrids_%Y-%m-%d_%H:%M")

# Need to change this according to 
samples_per_sec = 100

def print_bar_graph(dictionary, max_x=20):
    maximum = max(dictionary.values())
    num_per_x = max(round(maximum/20),1)
    graph_items = []
    for key,value in sorted(dictionary.items()):
        how_many_xs = round(value/num_per_x)
        xs = 'x' * how_many_xs
        if xs=='' and value > 0:
            xs = '.'
        num_string = '('+str(key)+', {0:.2f}) '.format(value)
        graph_items.append((num_string, xs))
    max_len = max([len(x[0]) for x in graph_items])
    for item in graph_items:
        space_padding = ' ' * (max_len-len(item[0]))
        print(item[0]+space_padding+item[1])


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
    # result_intervals = model.decode(data, tops, phone_intervals=True)
    result_intervals, groups = model.decode(data, tops, phone_intervals=True, context=True)

    with open("../experiments/groups/g1.txt", "w") as f1:
        for key,group in groups:
            for top in group:
                f1.write("{},{}\n".format(key,top))
    print("---")
    print("Phone sequence for file", fea_path, ":")
    print(result_intervals)

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
        output_dir = os.path.join('..', 'audio', date_string)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        amdtk.utils.write_textgrid(result_intervals, 
                                    samples_per_sec, 
                                    os.path.join(output_dir, os.path.split(fea_path)[1][:-4]+'.TextGrid'))

        print("Wrote textgrids to", output_dir)

print("success")

'''
'''

