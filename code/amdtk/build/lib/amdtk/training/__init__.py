
"""Acoustic Model Discovery Toolkit (AMDTK) module.
Set of tools to do Bayesian clustering of raw acoustic
features to automatically discover phone-like units.

"""
print("INSIDE INIT")
from .inference import StochasticVBOptimizer, NoisyChannelOptimizer, ToyNoisyChannelOptimizer
print("SUCCESSFUL IMPORT IN INIT")
