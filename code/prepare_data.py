import re
import os
import subprocess
import argparse 
import textgrid as tg

from amdtk.shared.phones import TIMIT_phones



def convert_dir(src_path,  output_path, conf):

	for root, dirs, files in os.walk(src_path):
		for filename in files:
			if filename.lower().endswith(".wav"):
				src_file = os.path.join(root, filename)
				output_path_full = os.path.join(output_path, os.path.relpath(root, src_path))
				wav_to_fea(src_file, output_path_full, conf)
			if filename.lower().endswith(".textgrid"):
				src_file = os.path.join(root, filename)
				output_path_full = os.path.join(output_path, os.path.relpath(src_path, root))
				textgrid_to_top(src_file, output_path_full)

def wav_to_fea(src, dest, conf):
	src_filename = re.sub("\.WAV", "", os.path.split(src)[-1])
	if not os.path.exists(dest):
		os.makedirs(dest)
	dest_path_full = os.path.join(dest, src_filename + ".fea")
	subprocess.Popen(['hcopy', '-C', conf, src, dest_path_full])

def textgrid_to_top(file, dest):
	src_filename = re.sub('\.TextGrid' , '', file)
	print('trying to convert file', file)
	t = tg.TextGrid()
	t.read(file)
	phones = [x.mark.lower() for x in t.tiers[0]]

	dest_path_full = os.path.join(dest, src_filename + '.top')

	with open(dest_path_full, "w") as f1:
		ints = []
		for phone in phones:
			assert(phone in TIMIT_phones.phone_to_int)
			ints.append(str(TIMIT_phones.phone_to_int[phone]))
		f1.write(",".join(ints))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("src",  help="source of wav files")
	parser.add_argument("dest",  help="desired location for output files")
	parser.add_argument("conf",  help="location of HCopy configuration file")
	args = parser.parse_args()
	print(args.src, args.dest)
	convert_dir(args.src, args.dest, args.conf)
