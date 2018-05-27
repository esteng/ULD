import re
import os
import argparse 

def pre_to_top(pre_file):

	top_file = pre_file[:-4]+".top"

	print("Converting "+pre_file+" to "+top_file)

	with open(pre_file, 'r') as f:
		pre_string = f.read().strip()
		phones = pre_string.split(" ")
		mappings = {}
		nums = []
		for phone in phones:
			if phone not in mappings:
				mappings[phone] = len(mappings)
			nums.append(mappings[phone])  

		top_string = ",".join([str(num) for num in nums])

		print(str(len(phones))+" phones in sequence")
		print(str(len(mappings))+" unique phones")

		with open(top_file, 'w') as g:
			g.write(top_string)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("pre",  help="path to file of pre-tops")
	args = parser.parse_args()
	
	pre_to_top(args.pre)