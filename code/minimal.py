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

# Generates toy data from the exact prior distributions used to initialize the model.
# 
# The model should be able to learn this pretty much perfectly; if not, something is very wrong.

# In[3]:


top_alphabet = [0,1,2,3]


#  let's have 1 more bottom level PLU
# 
# 

# In[4]:


bot_alphabet = [0,1,2,3,4]


# Now we get some random sequences of top letters that represent a top alphabet.
# 
# Since the top level strings are a given input into the model, it doesn't actually matter what we generate here. So we're just 

# In[6]:



top_strings = np.random.choice(top_alphabet, (100, 10))
# top_strings = [[2, 1, 3, 0, 2 ,1 ,3 ,2 ,0, 2]]



# Now we define prob distributions over ins, sub, del, and sample edit operations for each top-level PLU

# In[5]:


# Dirichlet parameters for distribution over edit ops for each of the top PLUs 
# (plus distribution over final insert bottoms)
# (insert_top prob) (insert_bottom probs) (sub probs)
ops_priors = [
    [1,1,2,1,1,1,3,2,1,1,1],
    [2,1,1,2,1,1,1,3,2,1,1],
    [3,1,1,1,2,1,1,1,3,1,1],
    [3,2,1,1,1,1,1,2,1,3,1],

]

# ops_priors = [
#     [1,.1,.1,.1,.1,.1,3,2,1,1,1],
#     [2,.1,.1,.1,.1,.1,1,3,2,1,1],
#     [3,.1,.1,.1,.1,.1,1,1,3,1,1],
#     [3,.1,.1,.1,.1,.1,1,2,1,3,1],
# ]

# Draw a sample from each Dirichlet to get the distribution
ops_dists = [np.random.dirichlet(i) for i in ops_priors]

# ins_top = [2]
# ins_bot = [.5,1,.5,4,2]
# sub = [[100, 3, 4, 1],
#       [.5, 100, 2, 2],
#       [2, .75, 100, 3],
#       [3,2,1,100]]


# full_dir = np.array(ins_top + ins_bot + sub[0] + sub[1] + sub[2] + sub[3], dtype=np.float64)
# # normalize 
# full_dir = full_dir/np.sum(full_dir)
# print(full_dir)



# In[6]:


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
        elif eo >= 1 and eo <= 5:
            bc = bot_alphabet[eo-1]
            bs.append(bc)
        else:
#             sub
#             bc = bot_alphabet[(eo-5)%4]
            bc = bot_alphabet[(eo-6)] 
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

# all_data = [np.array([[  2.05155094,  -0.21958725],
#        [-21.18441686,  13.47182682],
#        [-12.66353389, -14.43373648],
#        [ -2.44524407, -30.14195923],
#        [ -1.90920304, -29.72973341],
#        [ -1.64462944, -31.69891959],
#        [ -5.69023566, -29.46615842],
#        [ 20.79864861,   2.89165273],
#        [ 27.01527207,  17.78937303],
#        [ 21.00458018,   1.88587284],
#        [ 23.37018152,   3.09965334],
#        [ 18.48828411,   9.07030973],
#        [ 18.78263039,   7.50658146],
#        [  5.92027435,  12.87266549],
#        [-13.7214832 ,   5.57245987],
#        [-14.32365897,   7.26610337],
#        [  3.67983371, -28.59319584],
#        [-13.38810558,   5.73015594],
#        [-27.68491436, -19.0146655 ],
#        [ -9.22257572, -21.1771874 ],
#        [ -8.94542881, -20.90871353],
#        [-29.48310529,  26.92885736],
#        [ -4.08417101,  -4.4485657 ],
#        [ -2.88430622,  -3.20589674],
#        [-23.46705339, -22.42589584],
#        [  0.3087891 ,   1.42780744],
#        [-12.86878205, -13.64516234],
#        [-13.37105209, -15.06862477],
#        [ -2.53070072, -28.86305773],
#        [ 22.49028137,   0.55446982],
#        [  9.25689919,  26.84595558],
#        [  5.82101927,  11.56937009],
#        [  4.96671585, -28.9352741 ],
#        [-13.25692409,   6.19329784],
#        [-28.54330395, -18.12379468],
#        [-28.03557673, -17.94364959],
#        [ -8.53279034, -22.92530243],
#        [ -6.90226241, -21.54658041],
#        [-25.65417522,  -8.31538398],
#        [  3.75080818,   0.6689234 ],
#        [  1.44877553,  -0.91101976],
#        [-11.37860274, -12.66746968],
#        [ -2.42180208, -30.33001182],
#        [-29.48271375,  26.55873158],
#        [ 13.32428763,  -0.21532429],
#        [ 11.90868398, -25.93882685],
#        [ 23.8264038 ,   4.41090073],
#        [ 22.18004166,   5.31659165],
#        [ 24.30258949,   4.71339106],
#        [ 20.20984291,  -7.69392935],
#        [-11.15590566, -12.68065002],
#        [-10.11487073, -11.67312762],
#        [-10.16840314, -10.60686741],
#        [-19.75637473,  19.60022329],
#        [-17.79754711,  19.62239235],
#        [-18.8349971 ,  19.95086764],
#        [-18.04104671,  20.96738962],
#        [ 20.6046289 ,  -5.88050852],
#        [ -9.84550273, -11.19441872],
#        [-11.18751869, -12.27531394],
#        [-11.14250084, -11.17202901],
#        [ -8.34103464, -12.31465698],
#        [-17.53458765,  20.14990859]])]



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
    args= {'epochs': 10,
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
