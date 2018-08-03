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

from amdtk.shared.stats import collect_data_stats_by_speaker, accumulate_stats_by_speaker

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

rc = Client(profile='default')
dview = rc[:]
print('Connected to', len(dview), 'jobs.')

amdtk.evals.evaluate_model(dview, "/Users/Elias/ULD/code/epoch-0-batch-0", "/Users/Elias/ULD/audio/icicles", "/Users/Elias/ULD/code/eval_test",100, True)