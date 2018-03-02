import sys
import os
import time
import textgrid as tg
import argparse
import time
import re
import subprocess

polyglotdb_path = "/Users/esteng/PolyglotDB"
sys.path.insert(0, polyglotdb_path)

import polyglotdb.io as pgio
from polyglotdb import CorpusContext
from polyglotdb.utils import ensure_local_database_running
from polyglotdb.config import CorpusConfig

# path_to_timit = r'/Volumes/data/corpora/TIMIT_fixed/TEST/DR1/FAKS0'

graph_db = {'host':'localhost', 'port': 7474,
            'user': 'neo4j', 'password': 'test'}

SAM_RATE = 1

def loading(config, path_to_timit):
	# Initial import of the corpus to PGDB
	# only needs to be done once. resets the corpus if it was loaded previously.
	with CorpusContext(config) as c:
		c.reset()
		print('reset')
		parser = pgio.inspect_timit(path_to_timit)
		parser.call_back = call_back
		beg = time.time()
		c.load(parser, path_to_timit)
		end = time.time()
		print('Loading took: {}'.format(end - beg))



def call_back(*args):
	args = [x for x in args if isinstance(x, str)]
	if args:
		print(' '.join(args))


def export_textgrid(config, path, wav_path=None):
	with CorpusContext(config) as c:
		discourses = c.discourses
		levels = c.hierarchy.annotation_types
		for d in discourses:
			grid = tg.TextGrid()
			tier = tg.IntervalTier()

			q=c.query_graph(c.phone)
			q = q.filter(c.phone.discourse.name == d)
			q = q.order_by(c.phone.begin)
			res = q.all()

			for phone in res:
				tier.add(phone.begin/SAM_RATE, phone.end/SAM_RATE, phone.label)
			grid.append(tier)
			speaker = d.split("_")[0]
			just_file = d.split("_")[1]
			filename = just_file+".TextGrid"
			per_speaker_path = os.path.join(path, speaker)
			if not os.path.exists(per_speaker_path):
				os.mkdir(per_speaker_path)
			grid.write(os.path.join(per_speaker_path, filename))
			top_filename = just_file + ".top"
			export_tops(res, os.path.join(per_speaker_path, top_filename))
			if wav_path is not None:
				try:
					print("running ", just_file)
					path_to_wav = wav_path[just_file]
					mfcc(path_to_wav, os.path.join(per_speaker_path), "mfcc_16khz.conf")
				except KeyError:
					pass

def export_tops(phones, path):
	with open(path, "w") as f1:
		just_phones = [x.label for x in phones]
		phone_set = set(just_phones)
		mapping = {phone:str(i) for i, phone in enumerate(list(sorted(phone_set)))}
		int_phones = [mapping[x] for x in just_phones]
		f1.write(",".join(int_phones) + "\n")


def mfcc(src, dest, conf):
	print("running ", src)
	src_filename = re.sub("\.wav", "", os.path.split(src)[-1])
	dest_filename = os.path.join(dest, src_filename + ".fea")
	subprocess.Popen(['hcopy', '-C', conf, src, dest_filename])
	print("wrote to ", dest)
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("timit_path", help="path to timit corpus")	
	parser.add_argument("dest_path", help="path to destination")
	parser.add_argument("--reset", help="set to true to reset corpus", default=False)
	parser.add_argument("--convert", help="set to true if converting mfccs", default=False)
	args = parser.parse_args()


	corpus_name = "TIMIT"
	with ensure_local_database_running('database') as config:
		conf = CorpusConfig(corpus_name, **config)
		if args.reset:
			loading(conf, args.timit_path)
		if args.convert:
			filename_to_path = {}
			for root, dirs, files in os.walk(args.timit_path):
				for file in files:
					if re.match(".*\.[Ww][Aa][Vv]", file) is not None:
						src_filename = re.sub("\.[Ww][Aa][Vv]", "", file)
						path = os.path.join(root, file)
						filename_to_path[src_filename] = path



		export_textgrid(conf, args.dest_path, filename_to_path)



