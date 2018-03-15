import glob
import os
import numpy as np

from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot

from ipyparallel import Client
import sys

import amdtk
print(amdtk.__file__)
import subprocess


train_fea = []
train_fea_mask = '/Users/esteng/ULD/audio/TIMIT/FAKS0/*fea'
train_fea = [fname for fname in glob.glob(train_fea_mask)]
print(len(train_fea))


subprocess.Popen(['ipcluster', 'start',' --profile', 'default',' -n', '4', '--daemonize'])
subprocess.Popen(['sleep', '10']).communicate()



profile = 'default'
rc = Client(profile=profile)
rc.debug=True
dview = rc[:]
print('Connected to', len(dview), 'jobs.')


# Estimate the mean and the variance (per dimension) of the database. We need this statistics to perform mean/variance normalization during the training.

# In[41]:


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

data_stats = dview.map_sync(collect_data_stats, train_fea)

# Accumulate the statistics over all the utterances.
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


# ## Training
# 
# Now everything is ready for the training. First we need to create the phone-loop model. Currently, AMDTK also support Bayesian GMM though this model is usually less accurate.

# In[34]:


model = amdtk.PhoneLoop.create(
    50,  # number of acoustic units
    3,   # number of state per units
    4,   # number of Gaussian per emission
    np.zeros_like(data_stats['mean']), 
    np.ones_like(data_stats['var'])
)

#model = amdtk.Mixture.create(
#    200, # Number of Gaussian in the mixture.
#    np.zeros_like(data_stats['mean']), 
#    np.ones_like(data_stats['var'])
#)


# For the phone-loop and the GMM model optimization is done with the natural gradient descent. 

# In[35]:


elbo = []
time = []
def callback(args):
    elbo.append(args['objective'])
    time.append(args['time'])
    print('elbo=' + str(elbo[-1]), 'time=' + str(time[-1]))
    
optimizer = amdtk.StochasticVBOptimizer(
    dview, 
    data_stats, 
    {'epochs': 2,
     'batch_size': 400,
     'lrate': 0.01},
    model
)
optimizer.run(train_fea, callback)



data = amdtk.read_htk(train_fea[6])
print(model.decode(data))




print(model.decode(data, state_path=True))




subprocess.Popen(['ipcluster' ,'stop', '--profile', 'default'])

