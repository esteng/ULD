import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt  
import numpy as np 
import os


def visualize_edits(edit_path, save_path):
	ops, tops, bottoms = zip(*edit_path)
	top_y = 2
	bot_y = 0
	xs = [i for i in range(len(ops))]
	subs = [idx for idx in xs if ops[idx] == "SUB"]
	its = [idx for idx in xs if ops[idx] == "IT"]
	ibs = [idx for idx in xs if ops[idx] == "IB"]



	plt.scatter(subs, [top_y]*len(subs))
	plt.scatter(subs, [bot_y]*len(subs))

	for idx, i in enumerate(subs):
		plt.annotate(tops[i], xy=(i,top_y))
		plt.annotate(bottoms[i], xy=(i, bot_y))

	plt.scatter(its, [top_y]*len(its))
	for idx, i in enumerate(its):
		plt.annotate(tops[i], xy=(i, top_y))

	plt.scatter(ibs, [bot_y]*len(ibs))
	for idx, i in enumerate(ibs):
		plt.annotate(bottoms[i], xy=(i, bot_y))

	plt.savefig(save_path)

		
def heatmap(all_paths, save_path):
	all_tops = sorted(set([t[1] for path in all_paths for t in path]))
	all_bots = sorted(set([t[2] for path in all_paths for t in path]))

	# linear interpolation max to min
	all_tops = np.arange(0, all_tops[-1]+1, 1)
	all_bots = np.arange(0, all_bots[-1]+1, 1)
	print(all_tops)
	print(all_bots)

	heat_dict = {x: np.zeros((len(all_tops), len(all_bots))) for x in ["SUB", "IB", "IT"]}
	for path in all_paths:
		for op, top, bot in path:
			heat_dict[op][top, bot] += 1


	sub_arr = heat_dict["SUB"]
	fig, ax = plt.subplots()
	im = ax.imshow(sub_arr)
	ax.set_yticks(np.arange(len(all_tops)))
	ax.set_xticks(np.arange(len(all_bots)))
	ax.set_yticklabels(all_tops)
	ax.set_xticklabels(all_bots)

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	for i in range(len(all_tops)):
	    for j in range(len(all_bots)):
	        text = ax.text(j, i, sub_arr[i, j],
	                       ha="center", va="center", color="w")

	ax.set_title("Heatmap of substitutions")
	fig.tight_layout()
	plt.savefig(save_path)



