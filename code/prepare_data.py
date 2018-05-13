import re
import os
import subprocess
import argparse 
import textgrid as tg


timit_phone_symbols = [
'b',
'd',
'g',
'p',
't',
'k',
'bcl',
'dcl',
'gcl',
'pcl',
'tcl',
'kcl',
'dx',
'q',
'jh',
'ch',
's',
'sh',
'z',
'zh',
'f',
'th',
'v',
'dh',
'm',
'n',
'ng',
'em',
'en',
'eng',
'nx',
'',
'l',
'r',
'w',
'y',
'hh',
'hv',
'el',
'iy',
'ih',
'eh',
'ey',
'ae',
'aa',
'aw',
'ay',
'ah',
'ao',
'oy',
'ow',
'uh',
'uw',
'ux',
'er',
'ax',
'ix',
'axr',
'ax-h' ]

timit_silence_symbols = [
'',
'sil',
'pau',
'epi',
'h#',
'sp'
]

phone_to_int = {}

for i, phone in enumerate(timit_phone_symbols):
	phone_to_int[phone] = i

n_items = len(phone_to_int.items())
for i, silence in enumerate(timit_silence_symbols):
	phone_to_int[silence] = n_items




def convert_dir(src_path,  output_path, conf):

	phone_to_int = {}

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

	print('\n'.join([ str(x) for x in phone_to_int.items() ]))

def wav_to_fea(src, dest, conf):
	src_filename = re.sub("\.WAV", "", os.path.split(src)[-1])
	if not os.path.exists(dest):
		os.makedirs(dest)
	dest_path_full = os.path.join(dest, src_filename + ".fea")
	subprocess.Popen(['hcopy', '-C', conf, src, dest_path_full])

def textgrid_to_top(file, dest):
	src_filename = re.sub('\.TextGrid' , '', file)
	t = tg.TextGrid()
	t.read(file)
	phones = [x.mark.lower() for x in t.tiers[0]]
	next_phone_num = len(phone_to_int.items())

	dest_path_full = os.path.join(dest, src_filename + '.top')

	with open(dest_path_full, "w") as f1:
		ints = []
		for phone in phones:
			if phone in phone_to_int:
				ints.append(str(phone_to_int[phone]))
			else:
				print(phone + " not in dict")
		f1.write(",".join(ints))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("src",  help="source of wav files")
	parser.add_argument("dest",  help="desired location for output files")
	parser.add_argument("conf",  help="location of HCopy configuration file")
	args = parser.parse_args()
	print(args.src, args.dest)
	convert_dir(args.src, args.dest, args.conf)
