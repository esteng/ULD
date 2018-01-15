import glob
import os
import numpy as np
import time as systime

from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from ipyparallel import Client
import amdtk

print("successfully completed imports")

# I assume the reason this is parallelized is because
# for large amounts of data it could be very slow?
def collect_data_stats(filename):
    print("filename:", filename)
    """Job to collect the statistics."""
    # We  re-import this module here because this code will run
    # remotely.
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

#path = ['../audio_test/falr0_sx425.fea']
paths = ['../audio/abonza_lininisa.fea'] #, '../audio_test/falr0_sx425.fea']

#path_mask = '../audio_test/test/*fea'
#paths = [fname for fname in glob.glob(path_mask)]
#data = amdtk.read_htk(paths[0])

#path_mask = '../audio_test/buckeye/*fea'
# path_mask = '../audio_test/lectures/*fea'
# paths = [fname for fname in glob.glob(path_mask)]

print("Input files:")
for path in paths:
    print(path)

#path = ['../audio_test/abonza_lininisa.htk']
print("Getting mean and variance of input data...")
data_stats = dview.map_sync(collect_data_stats, paths)

# Accumulate the statistics over all the utterances.
final_data_stats = accumulate_stats(data_stats)





elbo = []
time = []
def callback(args):
    elbo.append(args['objective'])
    time.append(args['time'])
    print('elbo=' + str(elbo[-1]), 'time=' + str(time[-1]))
 

print("Creating phone loop model...")
conc = 0.1
model = amdtk.PhoneLoop.create(
    n_units=50,  # number of acoustic units
    n_states=3,   # number of states per unit
    n_comp_per_state=4,   # number of Gaussians per emission
    mean=np.zeros_like(final_data_stats['mean']), 
    var=np.ones_like(final_data_stats['var']) #,
    #concentration=conc
)
 
print("Creating VB optimizer...")   
optimizer = amdtk.StochasticVBOptimizer(
    dview, 
    final_data_stats, 
    args= {'epochs': 2,
     'batch_size': 400,
     'lrate': 0.01},
    model=model,

)

print("Running VB optimization...")
begin = systime.time()
optimizer.run(paths, callback)
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


for path in paths:
    data = amdtk.read_htk(path)

    # Normalize the data
    data_mean = np.mean(data)
    data_var = np.var(data)
    data = (data-data_mean)/np.sqrt(data_var)

    print("type of model is ", type(model))

    result = model.decode(data, state_path=False)
    result_path = model.decode(data, state_path=True)
    # result_intervals = model.decode(data)
    print("---")
    print("Phone sequence for file ", path)
    print(result)
    
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
        if not os.path.isdir(date_string):
            os.mkdir(date_string)
        amdtk.utils.write_textgrid(result_intervals, 
                                    samples_per_sec, 
                                    os.path.join(date_string, os.path.split(path)[1][:-4]+'.TextGrid'))

print("success")

'''
'''

