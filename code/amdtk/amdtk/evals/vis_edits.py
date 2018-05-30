from matplotlib import pyplot as plt  
import numpy as np 



def visualize_edits(edit_path):
	ops, tops, bottoms = zip(*edit_path)
	for i in range(len(ops)):
		
