import re
import os
import subprocess
import argparse 
import textgrid as tg

from amdtk.shared.phones import TIMIT_phones


special_cases = {

	# Dictionary entry has period
	'houses etc []': 'etc.',
	'and mr henry': 'mr.',
	'and mrs lincoln': 'mrs.',

	# Dictionary entry has leading/trailing hyphen
	'the anti infective': 'anti-',
	'the anti slavery': 'anti-',
	'make criss cross': 'criss-',
	'knick knacks within': '-knacks',
	'all knick knacks': 'knick-',
	'a non contributory': 'non-',
	'or non contributory': 'non-',
	'rated non supervisory': 'non-',
	'partly non nonsense': 'non-',
	'and semi heights': 'semi-',
	'this uni directional': 'uni-',
	'one upmanship is': '-upmanship',
	'push ups push': '-ups',
	'push ups are': '-ups',
	'moth zig zagged': 'zig-',
	'zig zagged along': '-zagged',

	# Homographs

	'arms close to': 'close~adj',
	'you close it': 'close~v',
	'particularly close to': 'close~adj',
	'snarled close overhead': 'close~adj',
	'was close herdin\'': 'close~adj',
	'her close []': 'close~adj',
	'they close sometime': 'close~v',

	'may live longer': 'live~v',
	'of live pigeons': 'live~adj',
	'penguins live near': 'live~v',
	'to live []': 'live~v',
	'should live in': 'live~v',
	'never live forever': 'live~v',

	'[] object a': 'object~n',
	'you object to': 'object~v',

	'graduation present []': 'present~n~adj',
	'interpretations present the': 'present~v',
	'seldom present anecdotal': 'present~v',
	'the present book': 'present~n~adj',
	'example present significant': 'present~v',
	'have present today': 'present~n~adj',
	'his present crew': 'present~n~adj',

	'[] project development': 'project~n',
	'the project with': 'project~n',
	'big project not': 'project~n',
	'can project long': 'project~v',

	'had read as': 'read~v_past',
	'to read are': 'read~v_pres',

	'to separate electron': 'separate~v',
	'of separate system': 'separate~adj',

	'fails use force': 'use~v',
	'didn\'t use white': 'use~v',
	'to use that': 'use~v',
	'cooking use curry': 'use~v',
	'biologists use radioactive': 'use~v',
	'would use up': 'use~v',
	'you use to': 'use~n',
	'always use mine': 'use~v',
	'purists use canned': 'use~v',
	'the use of': 'use~n',
	'to use these': 'use~v',
	'recreation use []': 'use~n',
	'you use parking': 'use~v',
	'[] use deductible': 'use~n',

	'a wound []': 'wound~n',
	'lagoon wound around': 'wound~v',
	'open wound was': 'wound~n',

}



def convert_dir(src_path, output_path, conf, use_wrd, timit_dict_path=None):

	counter = 0

	if use_wrd:
		timit_dict = read_dict(timit_dict_path)

	for root, dirs, files in os.walk(src_path):
		for filename in files:
			counter += 1
			if counter % 5000 == 0:
				print('Processed {} TIMIT files...'.format(counter))
			if False:#filename.lower().endswith(".wav"):
				src_file = os.path.join(root, filename)
				output_path_full = os.path.join(output_path, os.path.relpath(root, src_path))
				wav_to_fea(src_file, output_path_full, conf)
			if use_wrd:
				if filename.lower().endswith(".wrd"):
					src_file = os.path.join(root, filename)
					output_path_full = os.path.join(output_path, os.path.relpath(root, src_path))
					wrd_to_top(src_file, output_path_full, timit_dict)
			else:
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
	src_filename = re.sub('\.TextGrid' , '', os.path.split(file)[1])
	print('Converting {} to .top file'.format(file))
	t = tg.TextGrid()
	t.read(file)
	phones = [x.mark.lower() for x in t.tiers[0]]
	ints = [ TIMIT_phones.phone_to_int[p] for p in phones]
	
	dest_path_full = os.path.join(dest, src_filename + '.top')

	with open(dest_path_full, "w") as f1:
		f1.write(",".join(ints))

def wrd_to_top(file, dest, timit_dict):
	src_filename = re.sub('\.WRD', '', os.path.split(file)[1])

	# Open .wrd file (which lists all words in this timit recording)
	with open(file, 'r') as f:
		all_phones = []
		# Get word
		lines = f.read().strip().split('\n')
		line_parts = [ line.strip().split(' ') for line in lines ]
		for i, parts in enumerate(line_parts):
			assert(len(parts)==3)
			word = parts[2]

			# Get list of phones for word
			if word in timit_dict:
				phones = timit_dict[word]
			else:
				# Handle weird cases
				before = line_parts[i-1][2]+' ' if i > 0 else '[] '
				after = ' '+line_parts[i+1][2] if i < len(line_parts)-1 else ' []'
				context = before+word+after
				word_fixed = special_cases[context]

				if word_fixed in timit_dict:
					phones = timit_dict[word_fixed]

				else:
					raise KeyError('KeyError: {}'.format(word_fixed))
			all_phones.extend(phones)

		# Convert phones to ints
		all_ints = [ TIMIT_phones.phone_to_int[p] for p in all_phones]
	
	# Write to .top file
	dest_path_full = os.path.join(dest, src_filename + '.top')
	with open(dest_path_full, "w") as f1:
		# f1.write('\n'.join([ '{}\t{}'.format(z[0], z[1]) for z in zipped ]))
		f1.write(",".join([ str(i) for i in all_ints ]))


def handle_homographs(context):
	pass



def read_dict(dict_path):

	timit_dict = {}

	with open(dict_path, 'r') as f:

		for line in f:
			line = line.strip()
			if line[0] == ';':
				continue
			word, phones = line.split('  ')
			phones = phones.replace('/', '').replace('1','').replace('2','')
			phone_list = phones.split(' ')
			timit_dict[word] = phone_list
	print('Loaded TIMIT dict...')
	return timit_dict





if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("src",  help="source of wav files")
	parser.add_argument("dest",  help="desired location for output files")
	parser.add_argument("conf",  help="location of HCopy configuration file")
	parser.add_argument("dict",  help="location of TIMIT pronunciation dictionary (only needed if --wrd option is set)")
	parser.add_argument("--wrd", help="whether to generate tops from .WRD files (rather than TextGrids)", action="store_true", default=False)
	args = parser.parse_args()
	print('\n*************************************')
	print('Source: {}\nDestination: {}\nTIMIT dict location: {}\nGenerate tops from transcriptions: {}'.format(args.src, args.dest, args.dict, args.wrd))
	print('*************************************\n')
	convert_dir(args.src, args.dest, args.conf, args.wrd, args.dict)
	print('Done.\n')
