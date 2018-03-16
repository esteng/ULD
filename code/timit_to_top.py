import re
import os
import sys

def convert_dir(path, dest):
	all_sents = []
	all_paths = []
	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith(".PHN"):
				discourse = file.split(".")[0]
				speaker = os.path.split(root)[1]
				path_to_file = os.path.join(root,file)

				with open(path_to_file, "r") as f1:
					lines = f1.readlines()
				labels = []
				for line in lines:
					splitline = re.split("\s", line)
					label = splitline[2]
					labels.append(label.strip())
				all_sents.append(labels)
				all_paths.append(os.path.join(dest, speaker, discourse + ".top"))
	# get set of all chars
	all_chars = set([x for y in all_sents for x in y])
	mapping = {c:str(i) for i,c in enumerate(sorted(all_chars))}
	all_sents = [[mapping[c] for c in sent] for sent in all_sents]
	assert(len(all_sents) == len(all_paths))
	for i, s in enumerate(all_sents):
		with open(all_paths[i], "w") as f1:
			f1.write(",".join(s))


convert_dir("/Volumes/data/corpora/TIMIT_fixed/TRAIN", "/Users/esteng/ULD/audio/TIMIT")

