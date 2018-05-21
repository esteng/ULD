from cluster import read_tg
import re


from itertools import groupby

# we want to read top tg, then bottom tg
# for a given top phone, we want to see what it was replaced by in bottom
# e.g. t: bu1er, ban2er, wan2, ...
# so we need: 
	# the top, bottom tgs
	# the operations done and where they were done


def get_context(group_file, top_string, top_plu):
	with open(group_file) as f1:
		group_lines = [x.split(",") for x in f1.readlines()]
		bottom_plus = [x[0] for x in group_lines]
		top_idxs = [int(x[2][:-2]) for x in group_lines]
		print(top_idxs)
		top_to_bottom = {top_string[top_idx]:[] for top_idx in top_idxs}
		for i,top_idx in enumerate(top_idxs):
			top_to_bottom[top_string[top_idx]].append(bottom_plus[i])

		print(top_to_bottom.keys())

		interest_idxs = [i for i,x in enumerate(top_string) if x == top_plu]
		context_window = 3
		interest_strings = []
		for idx in interest_idxs:
			start = idx-3 if idx-3>=0 else 0			
			end=idx+3 if idx+3<len(top_string) else 0
			string = top_string[start:end]
			for bottom_map in top_to_bottom[string[3]]:
				string[3] = bottom_map
				interest_strings.append(string)

		print(set([tuple(x) for x in interest_strings]))








top = [x for x in re.split("(.)", "wi sa EIt tAIni AIsIk@l@z b@loU AUR Ruf") if x not in ['', ' ']]
get_context("../groups/filename", top, 't')
	# file format: bottom,(bottom, top)

