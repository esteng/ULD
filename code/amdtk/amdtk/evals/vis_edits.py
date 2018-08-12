import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt  
import numpy as np 
import os


def visualize_edits(edit_path, save_path):
	plt.figure(figsize=(12,3))
	plt.axis("off")
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
		# plt.plot((i,i), (top_y, bot_y), 'o-')

	plt.scatter(its, [top_y]*len(its))
	for idx, i in enumerate(its):
		plt.annotate(tops[i], xy=(i, top_y))

		# plt.plot((i,i-1), (top_y, bot_y), 'o-')

	plt.scatter(ibs, [bot_y]*len(ibs))
	for idx, i in enumerate(ibs):
		plt.annotate(bottoms[i], xy=(i, bot_y))

	# draw lines
	top_ptr = -1
	bottom_ptr = -1
	for op, top, bottom in edit_path:
		if op == "SUB":
			# draw straight line
			top_ptr += 1 
			bottom_ptr += 1
			max_ptr = max(top_ptr, bottom_ptr)
			top_ptr, bottom_ptr = max_ptr,max_ptr
			plt.plot((top_ptr, bottom_ptr), (top_y, bot_y), 'bo-')
			
		if op == "IT":
			top_ptr += 1 
			plt.plot((top_ptr, bottom_ptr), (top_y, bot_y), 'ro-')
			
		if op == "IB":
			bottom_ptr += 1
			plt.plot((top_ptr, bottom_ptr), (top_y, bot_y), 'go-')
			
	plt.savefig(save_path)

		
def heatmap(all_paths, save_path):
	plt.figure()
	if len(all_paths) == 0:
		# No paths, so nothing to do
		return

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
	        text = ax.text(j, i, int(sub_arr[i, j]),
	                       ha="center", va="center", color="w")

	ax.set_title("Heatmap of substitutions")
	# fig.tight_layout()
	plt.savefig(save_path)


visualize_edits([('SUB', 11, 0), ('SUB', 24, 2), ('IT', 15, 2), ('SUB', 12, 3), ('IT', 6, 3), ('SUB', 4, 1), ('IT', 16, 1), ('SUB', 2, 3), ('SUB', 4, 1), ('IT', 3, 1), ('SUB', 0, 3), ('IT', 22, 3), ('SUB', 18, 0), ('SUB', 17, 1), ('IT', 23, 1), ('SUB', 25, 0), ('SUB', 21, 2), ('SUB', 14, 3), ('SUB', 20, 0), ('SUB', 10, 3), ('SUB', 9, 1), ('SUB', 22, 0), ('SUB', 15, 1), ('SUB', 23, 3), ('SUB', 15, 2), ('SUB', 26, 3), ('SUB', 1, 1), ('SUB', 24, 3), ('SUB', 7, 1), ('SUB', 26, 2), ('SUB', 1, 1), ('SUB', 5, 3), ('SUB', 8, 1), ('SUB', 21, 0), ('SUB', 1, 1), ('SUB', 19, 3), ('SUB', 27, 0), ('SUB', 13, 3), ('IT', 8, 3), ('IT', 11, 3)]
, "/Users/Elias/Desktop/test.png")
