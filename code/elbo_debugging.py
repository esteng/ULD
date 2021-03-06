
import glob
import os
import numpy as np
import time as systime
import re 
import textgrid as tg
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from ipyparallel import Client
import _pickle as pickle

import sys
# 
np.random.seed(1234)
# np.random.seed(438)



top_alphabet = [0,1,2,3]
bot_alphabet = [0,1,2,3,4]
# top_strings = np.random.choice(top_alphabet, (40, 10))

# just broken example
top_strings = [[3 ,3, 2, 3, 1, 2, 3,2, 2, 0]]

ops_priors = [
    [1,1,2,1,1,1,3,2,1,1,1],
    [2,1,1,2,1,1,1,3,2,1,1],
    [3,1,1,1,2,1,1,1,3,1,1],
    [3,2,1,1,1,1,1,2,1,3,1],

]

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
        if eo == 0:
#             insert top, do nothing
            j+=1
        elif eo >= 1 and eo <= 5:
            bc = bot_alphabet[eo-1]
            bs.append(bc)
        else:
            bc = bot_alphabet[(eo-6)] 
            bs.append(bc)
            j+=1
    i+=1

    bottom_strings.append(bs)


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


# We need a set of parameters for each component
# for each HMM state for each bottom PLU
# So in this case, we need 2 * 3 * 5 = 30 sets of parameters
component_parameters = [
    [
        [
            [[0,0], [[1,0],[0,1]]], # bottom PLU 0, HMM state 0, component 0
            [[0.1,0.1], [[1,0],[0,1]]]  # bottom PLU 0, HMM state 0, component 1
        ],
        [
             [[0.2,0.2], [[1,0],[0,1]]], # bottom PLU 0, HMM state 1, component 0
             [[0.3,0.03], [[1,0],[0,1]]]  # bottom PLU 0, HMM state 1, component 1
        ],
        [
            [[0.4,0.4], [[1,0],[0,1]]], # bottom PLU 0, HMM state 2, component 0
            [[0.5,0.5], [[1,0],[0,1]]]  # bottom PLU 0, HMM state 2, component 1
        ]
    ],
    [
        [
            [[1.0,1.0], [[1,0],[0,1]]], # bottom PLU 1, HMM state 0, component 0
            [[1.1,1.1], [[1,0],[0,1]]]  # bottom PLU 1, HMM state 0, component 1
        ],
        [
             [[1.2,1.2], [[1,0],[0,1]]], # bottom PLU 1, HMM state 1, component 0
             [[1.3,1.3], [[1,0],[0,1]]]  # bottom PLU 1, HMM state 1, component 1
        ],
        [
            [[1.4,1.4], [[1,0],[0,1]]], # bottom PLU 1, HMM state 2, component 0
            [[1.5,1.5], [[1,0],[0,1]]]  # bottom PLU 1, HMM state 2, component 1
        ]
    ],
    [        
        [
            [[2.0,2.0], [[1,0],[0,1]]], # bottom PLU 2, HMM state 0, component 0
            [[2.1,2.1], [[1,0],[0,1]]]  # bottom PLU 2, HMM state 0, component 1
        ],
        [
             [[2.2,2.2], [[1,0],[0,1]]], # bottom PLU 2, HMM state 1, component 0
             [[2.3,2.3], [[1,0],[0,1]]]  # bottom PLU 2, HMM state 1, component 1
        ],
        [
            [[2.4,2.4], [[1,0],[0,1]]], # bottom PLU 2, HMM state 2, component 0
            [[2.5,2.5], [[1,0],[0,1]]]  # bottom PLU 2, HMM state 2, component 1
        ]
    ],
    [
        [
            [[3.0,3.0], [[1,0],[0,1]]], # bottom PLU 3, HMM state 0, component 0
            [[3.1,3.1], [[1,0],[0,1]]]  # bottom PLU 3, HMM state 0, component 1
        ],
        [
             [[3.2,3.2], [[1,0],[0,1]]], # bottom PLU 3, HMM state 1, component 0
             [[3.3,3.3], [[1,0],[0,1]]]  # bottom PLU 3, HMM state 1, component 1
        ],
        [
            [[3.4,3.4], [[1,0],[0,1]]], # bottom PLU 3, HMM state 2, component 0
            [[3.5,3.5], [[1,0],[0,1]]]  # bottom PLU 3, HMM state 2, component 1
        ]
    ],
    [
        [
            [[4.0,4.0], [[1,0],[0,1]]], # bottom PLU 4, HMM state 0, component 0
            [[4.1,4.1], [[1,0],[0,1]]]  # bottom PLU 4, HMM state 0, component 1
        ],
        [
             [[4.2,4.2], [[1,0],[0,1]]], # bottom PLU 4, HMM state 1, component 0
             [[4.3,4.3], [[1,0],[0,1]]]  # bottom PLU 4, HMM state 1, component 1
        ],
        [
            [[4.4,4.4], [[1,0],[0,1]]], # bottom PLU 4, HMM state 2, component 0
            [[4.5,4.5], [[1,0],[0,1]]]  # bottom PLU 4, HMM state 2, component 1
        ]
    ]
]


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

all_data = []

# test broken
# for i, bs in enumerate(bottom_strings):
#     mfccs = sample_HMMGMM(bs)
#     all_data.append(mfccs)

broken= np.array([[3.6940856145206573, 4.2145844941542911], [4.9519037656236629, 3.3738197399146892], [2.879431727306732, 4.8065272541729112], [2.3370229111652407, 4.7993773961262693], [3.212840577431435, 3.645435024464442], [4.2663255701035352, 4.7516586542953858], [5.96875470770512, 4.115505597211472], [0.93082600958804962, 0.21343186204453946], [1.716017298763892, 0.98815048015749474], [1.7768393714901807, 0.96761273548975457], [1.1585292194500256, 1.8279667672261921], [0.79726355937057458, 3.6948927325339946], [2.289084870762069, 1.5877617634548575], [1.415612090613257, 3.4920535479558912], [2.0681299003444638, 3.194713581182774], [0.1367139332991516, 2.7750914193212175], [1.7067263087430105, 1.2162584527239384], [0.092819668662665444, 1.4053744245487607], [2.2092441700356584, -0.27491613400573867], [1.3769304337397723, 1.3623271062334319], [-2.0080487915088909, -0.93732001179232804], [0.1313425741849113, -0.46397547211393525], [-0.010738241484489708, 1.3829500028873207], [1.6918940625141223, 3.2225916674582198], [2.797902595183213, 2.605013317465386], [2.0448141784651748, 2.2074609006877259], [4.1920837587792734, 2.9112206641703251], [1.7060222221460759, 4.3009725487832418], [0.98089195700399201, 2.0292749978017031]])
all_data.append(broken)


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
        'var': var
    }
    return data_stats

final_data_stats = accumulate_stats(data_stats)



print("Running...")

import subprocess
import amdtk
num_tops = 4

print("Finished imports in ipython notebook.")

all_data_and_tops = list(zip(all_data, list(top_strings)))

#print(all_data_and_tops[0])

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
    max_slip_factor=.12
    #concentration=conc
)


print("Creating VB optimizer...")   
optimizer = amdtk.ToyNoisyChannelOptimizer(
    None, 
    final_data_stats, 
    args= {'epochs': 3,
     'batch_size': 4,
     'lrate': 0.01,
     'pkl_path': "example_test",
     'log_dir': 'logs'},
    model=model,
    dir_path='dir'
)

print("Running VB optimization...")
begin = systime.time()
print("running with toy data")
print('len(all_data_and_tops[0])')

optimizer.run(all_data_and_tops, callback)
end = systime.time()
print("VB optimization took ",end-begin," seconds.")
print(elbo)
plt.plot(time,elbo)
plt.show()