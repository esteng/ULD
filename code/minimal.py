import glob
import os
import numpy as np
import time as systime
import re 
import textgrid as tg
import matplotlib.pyplot as plt
import matplotlib as matplotlib

from scipy.stats import multivariate_normal

from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from ipyparallel import Client
import _pickle as pickle
import random 
import sys

myseed = 6
np.random.seed(myseed)
np.seterr(divide='raise', over='raise', under='raise', invalid='raise')



import amdtk


top_alphabet = [0,1,2,3]



bot_alphabet = [0,1,2,3,4]





top_strings = np.random.choice(top_alphabet, (50, 10))


# Dirichlet parameters for distribution over edit ops for each of the top PLUs 
# (plus distribution over final insert bottoms)
# (insert_top prob) (insert_bottom probs) (sub probs)
# ops_priors = [
#     [1,1,2,1,1,1,3,2,1,1,1],
#     [2,1,1,2,1,1,1,3,2,1,1],
#     [3,1,1,1,2,1,1,1,3,1,1],
#     [3,2,1,1,1,1,1,2,1,3,1],

# ]

ops_priors = np.ones((len(top_alphabet), 2*len(bot_alphabet)+1))
# Draw a sample from each Dirichlet to get the distribution
ops_dists = [np.random.dirichlet(i) for i in ops_priors]



bottom_strings = []
i = 0
while i < len(top_strings):
    ts = top_strings[i]
    j = 0
    bs = []
    while j < len(ts):
#         sample some edit operation
#         eo = np.argmax(np.random.multinomial(1, full_dir))
        eo = np.argmax(np.random.multinomial(1, ops_dists[ts[j]]))
#         print(ops_dists[ts[j]])
#         print(eo)
        if eo == 0:
#             insert top, do nothing
            j+=1
        elif eo >= 1 and eo <= len(bot_alphabet):
            bc = bot_alphabet[eo-1]
            bs.append(bc)
        else:
#             sub
#             bc = bot_alphabet[(eo-5)%4]
            bc = bot_alphabet[(eo- (len(bot_alphabet) + 1))] 
            bs.append(bc)
            j+=1
    i+=1
    print("num of top string PLUs: ", len(ts))
    print("num of bottom string PLUs: ", len(bs))
    print("\n")
    bottom_strings.append(bs)

print(bottom_strings)
print(len(bottom_strings[0]))

components = []
# Distribution over components for each bottom PLU state
components.append([[.7,.3],
             [.9,.1],
             [.3,.7]])
components.append([[.3,.7],
             [.5,.5],
             [.8,.2]])
components.append([[.4,.6],
             [.2,.8],
             [.9,.1]])
components.append([[.5,.5],
             [.3,.7],
             [.6,.4]])
components.append([[.1,.9],
             [.6,.4],
             [.7,.3]])

x_vals = list(range(-30,30,2))
random.shuffle(x_vals)
y_vals = list(range(-30,30,2))
random.shuffle(y_vals)

random_means = [[x,y] for x,y in zip(x_vals,y_vals)]

component_parameters = [
    [
        [
            [random_means[0], [[1,0],[0,1]]], # bottom PLU 0, HMM state 0, component 0
            [random_means[1], [[1,0],[0,1]]]  # bottom PLU 0, HMM state 0, component 1
        ],
        [
             [random_means[2], [[1,0],[0,1]]], # bottom PLU 0, HMM state 1, component 0
             [random_means[3], [[1,0],[0,1]]]  # bottom PLU 0, HMM state 1, component 1
        ],
        [
            [random_means[4], [[1,0],[0,1]]], # bottom PLU 0, HMM state 2, component 0
            [random_means[5], [[1,0],[0,1]]]  # bottom PLU 0, HMM state 2, component 1
        ]
    ],
    [
        [
            [random_means[6], [[1,0],[0,1]]], # bottom PLU 1, HMM state 0, component 0
            [random_means[7], [[1,0],[0,1]]]  # bottom PLU 1, HMM state 0, component 1
        ],
        [
             [random_means[8], [[1,0],[0,1]]], # bottom PLU 1, HMM state 1, component 0
             [random_means[9], [[1,0],[0,1]]]  # bottom PLU 1, HMM state 1, component 1
        ],
        [
            [random_means[10], [[1,0],[0,1]]], # bottom PLU 1, HMM state 2, component 0
            [random_means[11], [[1,0],[0,1]]]  # bottom PLU 1, HMM state 2, component 1
        ]
    ],
    [        
        [
            [random_means[12], [[1,0],[0,1]]], # bottom PLU 2, HMM state 0, component 0
            [random_means[13], [[1,0],[0,1]]]  # bottom PLU 2, HMM state 0, component 1
        ],
        [
             [random_means[14], [[1,0],[0,1]]], # bottom PLU 2, HMM state 1, component 0
             [random_means[15], [[1,0],[0,1]]]  # bottom PLU 2, HMM state 1, component 1
        ],
        [
            [random_means[16], [[1,0],[0,1]]], # bottom PLU 2, HMM state 2, component 0
            [random_means[17], [[1,0],[0,1]]]  # bottom PLU 2, HMM state 2, component 1
        ]
    ],
    [
        [
            [random_means[18], [[1,0],[0,1]]], # bottom PLU 3, HMM state 0, component 0
            [random_means[19], [[1,0],[0,1]]]  # bottom PLU 3, HMM state 0, component 1
        ],
        [
             [random_means[20], [[1,0],[0,1]]], # bottom PLU 3, HMM state 1, component 0
             [random_means[21], [[1,0],[0,1]]]  # bottom PLU 3, HMM state 1, component 1
        ],
        [
            [random_means[22], [[1,0],[0,1]]], # bottom PLU 3, HMM state 2, component 0
            [random_means[23], [[1,0],[0,1]]]  # bottom PLU 3, HMM state 2, component 1
        ]
    ],
    [
        [
            [random_means[24], [[1,0],[0,1]]], # bottom PLU 4, HMM state 0, component 0
            [random_means[25], [[1,0],[0,1]]]  # bottom PLU 4, HMM state 0, component 1
        ],
        [
             [random_means[26], [[1,0],[0,1]]], # bottom PLU 4, HMM state 1, component 0
             [random_means[27], [[1,0],[0,1]]]  # bottom PLU 4, HMM state 1, component 1
        ],
        [
            [random_means[28], [[1,0],[0,1]]], # bottom PLU 4, HMM state 2, component 0
            [random_means[29], [[1,0],[0,1]]]  # bottom PLU 4, HMM state 2, component 1
        ]
    ]


]



all_data = []



# all_data.append(broken)

def sample_HMMGMM(bs):
    mfccs = []
    for i, char in enumerate(bs):
        vec = None
#         if i%10 == 0:
#             print("done with ", i)
#         start_state = np.random.choice([0,1,2], p=init)
        curr_state = 0
#       get the mfcc vector and transition
        while curr_state < 3:
            gmm_component_choice = np.random.choice(2,p=components[char][curr_state])
            vec = np.random.multivariate_normal(*component_parameters[char][curr_state][gmm_component_choice])
#             vec = components[curr_state][0]*np.random.multivariate_normal(*n1) + \
#                     components[curr_state][1]*np.random.multivariate_normal(*n2)
            
            mfccs.append(vec)
#             new_curr_state = np.random.choice([0,1,2], p=transition[curr_state])
            # Transition to next state with 50% probability
            curr_state = np.random.choice([curr_state,curr_state+1],p=[0.5,0.5])
#             if curr_state == 2 and new_curr_state == 0:
    #         get the mfcc vector and transition
#             curr_state = new_curr_state
    return np.array(mfccs)


for i, bs in enumerate(bottom_strings):
    print("num of bottom PLUs: ", len(bs))
    mfccs = sample_HMMGMM(bs)
    print("num of frames: ", mfccs.shape[0])
    print("frames per PLU: ", mfccs.shape[0]/float(len(bs)))
    
    all_data.append(mfccs)

def collect_data_stats(data):
    """Job to collect the statistics."""
    # We  re-import this module here because this code will run
    # remotely.
    
    stats_0 = data.shape[0]
    stats_1 = data.sum(axis=0)
    stats_2 = (data**2).sum(axis=0)
    retval = (
        stats_0,
        stats_1,
        stats_2
    )
    return retval

data_stats = list(map(collect_data_stats, all_data))



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

final_data_stats = accumulate_stats(data_stats)


import subprocess
import amdtk
num_tops = 4

all_data_and_tops = list(zip(all_data, top_strings))


elbo = []
time = []
def callback(args):
    elbo.append(args['lower_bound'])
    time.append(args['time'])
    print('elbo=' + str(elbo[-1]), 'time=' + str(time[-1]))
 

print("Creating phone loop model...")
conc = 0.1

n_units=5  # number of acoustic units
n_states=3   # number of states per unit
n_comp_per_state=3   # number of Gaussians per emission


model = amdtk.PhoneLoopNoisyChannel.create(
    n_units=n_units,  # number of acoustic units
    n_states=n_states,   # number of states per unit
    n_comp_per_state=n_comp_per_state,   # number of Gaussians per emission
    n_top_units=num_tops, # size of top PLU alphabet
    mean=np.zeros_like(final_data_stats['mean']), 
    var=np.ones_like(final_data_stats['var']),
    max_slip_factor=.05
    #concentration=conc
)

optimizer = amdtk.ToyNoisyChannelOptimizer(
    None, 
    final_data_stats, 
    args= {'epochs': 1,
     'batch_size': 1,
     'lrate': 0.01,
     'pkl_path': "example_test",
     'log_dir': 'logs'},
    model=model,
    dir_path="dir"
    
)

print("Running VB optimization...")
begin = systime.time()
optimizer.run(all_data_and_tops, callback)
end = systime.time()
print("VB optimization took ",end-begin," seconds.")

print("\nDECODING\n")

# Need to change this according to 
samples_per_sec = 100

all_data_and_tops = list(zip(all_data, top_strings))

for data, tops in all_data_and_tops:

    # Normalize the data
    data_mean = np.mean(data)
    data_var = np.var(data)
    data = (data-data_mean)/np.sqrt(data_var)

    #result = model.decode(data, tops, state_path=False)
    #result_path = model.decode(data, tops, state_path=True)
    (result_intervals, edit_path, plu_path) = model.decode(data, tops, phone_intervals=True, edit_ops=True)

    print("---")
    print("Phone sequence:")
    print(result_intervals)
    print(edit_path)
