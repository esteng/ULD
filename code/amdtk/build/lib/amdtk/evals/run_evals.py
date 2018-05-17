import argparse 
import pickle
import os
import re 
import numpy as np
import textgrid as tg
import amdtk

from .cluster import segmentation_performance, pred_to_true_clustering, seq_to_bound
from .nmi import tot_nmi, tot_ami
from ..io import read_htk


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
'ax-h',
'oo' ]

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

int_to_phone = {v: k for k, v in phone_to_int.items()}



def frame_labeling_accuracy(model_frame_labels, gold_standard_frame_labels, model_to_gold):
	num_frames = sum([len(x) for x in model_frame_labels])
	num_correct = 0
	for model_utterance, gold_utterance in zip(model_frame_labels, gold_standard_frame_labels):
		for model_frame, gold_frame in zip(model_utterance, gold_utterance):
			if model_to_gold[model_frame] == gold_frame:
				num_correct += 1
	return float(num_correct)/num_frames

def boundary_precision(model_frame_labels, gold_standard_frame_labels):

	model_boundaries = seq_to_bound(model_frame_labels)
	gold_boundaries = seq_to_bound(gold_standard_frame_labels)

	total = 0
	correct = 0

	for i, bound in enumerate(model_boundaries):
		for j in range(len(bound)):
			if bound[j] == 1:
				gold_window = gold_boundaries[i][max(0, j-1):min(len(bound), j+2)]
				if any(gold_window):
					correct += 1
				total +=1

	return correct/total



def evaluate_model(model_dir, audio_dir, output_dir, samples_per_sec, one_model=False, write_textgrids=False):
	if not one_model:
		# Load model(s)
		models = []
		for dirpath, dirnames, filenames in os.walk(model_dir):
			for filename in filenames:
				if filename.lower().endswith(".pkl"):
					model_path = os.path.join(dirpath, filename)
					model = pickle.load(open(model_path,"rb"))
					models.append((model, model_path))
	# only load one model
	else:
		model = pickle.load(open(model_dir, 'rb'))
		models = [(model, model_dir)]

	# Compile audio file paths
	audio_dir = os.path.abspath(audio_dir)
	fea_paths = []
	top_paths = []
	textgrid_paths = []
	for root, dirs, files in os.walk(audio_dir):
		for file in files:
			if file.lower().endswith(".fea"): 
				fea_paths.append(os.path.join(root,file))
			if file.lower().endswith(".top"):
				top_paths.append(os.path.join(root, file))
			if file.lower().endswith(".textgrid"):
				textgrid_paths.append(os.path.join(root, file))

	print('fea_paths:', fea_paths)
	print('top_paths:', top_paths)
	print('textgrid_paths:', textgrid_paths)

	zipped_paths = list(zip(sorted(fea_paths), sorted(top_paths), sorted(textgrid_paths)))

	for fea_file, top_file, textgrid_file in zipped_paths:
		assert(re.sub("\.fea", "", fea_file) == re.sub("\.top", "", top_file))
		assert(re.sub("\.fea", "", fea_file) == re.sub("\.TextGrid", "", textgrid_file))


	for model, model_path in models:

		model_output_dir = os.path.join(output_dir, 'model_{}'.format(os.path.basename(model_path)))
		if not os.path.exists(model_output_dir):
			os.mkdir(model_output_dir)

		pred_frame_labels_all = []
		true_frame_labels_all = []
		edit_ops_all = []

		print("Evaluating model",model_path)

		# Decode the data using the model
		for fea_path, top_path, textgrid_path in zipped_paths:


			data = read_htk(fea_path)

			# Normalize the data
			data_mean = np.mean(data)
			data_var = np.var(data)
			data = (data-data_mean)/np.sqrt(data_var)

			# Read top PLU sequence from file
			with open(top_path, 'r') as f:
				topstring = f.read()
				tops = topstring.strip().split(',')
				tops = [int(x) for x in tops]

			#result = model.decode(data, tops, state_path=False)
			#result_path = model.decode(data, tops, state_path=True)
			phone_intervals, edit_ops, hmm_states, plus = \
					model.decode(data, tops, phone_intervals=True, 
													edit_ops=True, hmm_states=True, plus=True)
			pred_frame_labels = np.array(plus)


			# print("---")
			# print("Predicted labels for file", fea_path, ":")
			# print(pred_frame_labels)
			# print("pred shape: ", pred_frame_labels.shape)

			# Get the PLU labels from the textgrid
			true_frame_labels = np.array(read_tg(textgrid_path, len(pred_frame_labels)))

			# print("---")
			# print("True labels from textgrid", textgrid_path, ":")
			# print(true_frame_labels)
			# print("true shape:", true_frame_labels.shape)

			assert(len(pred_frame_labels)==len(true_frame_labels))

			pred_frame_labels_all.append(pred_frame_labels)
			true_frame_labels_all.append(true_frame_labels)
			edit_ops_all.append(edit_ops)


		pred_frame_labels_all = np.array(pred_frame_labels_all)
		true_frame_labels_all = np.array(true_frame_labels_all)

		if write_textgrids:
			tg_dir = os.path.join(model_output_dir, 'textgrids')
			if not os.path.exists(tg_dir):
				os.mkdir(tg_dir)
			amdtk.utils.write_textgrid(phone_intervals, 
									samples_per_sec, 
									os.path.join(tg_dir, os.path.split(fea_path)[1][:-4]+'.TextGrid'))

		# Perform analyses

		# print(segmentation_performance(true_frame_labels_all, pred_frame_labels_all))

		correspond_output_file = os.path.join(model_output_dir, 'correspond_{}.txt'.format(os.path.basename(model_path)))
		correspondance_analysis(true_frame_labels_all, pred_frame_labels_all, correspond_output_file)

		edit_op_output_file = os.path.join(model_output_dir, 'edit_ops_{}.txt'.format(os.path.basename(model_path)))
		edit_op_analysis(edit_ops_all, max(phone_to_int.values()), edit_op_output_file)

		# print(pred_to_true_clustering(true_frame_labels_all, pred_frame_labels_all))

		# print(avg_nmi(true_frame_labels_all, pred_frame_labels_all))
		# return(segmentation_performance(true_frame_labels_all, pred_frame_labels_all),\
		#  pred_to_true_clustering(true_frame_labels_all, pred_frame_labels_all),
		#  avg_nmi(true_frame_labels_all, pred_frame_labels_all))
# <<<<<<< HEAD
# 		return(segmentation_performance(true_frame_labels_all, pred_frame_labels_all), 
# 			pred_to_true_clustering(true_frame_labels_all, pred_frame_labels_all), 
# 			avg_nmi(true_frame_labels_all, pred_frame_labels_all))


		# Write clustering accuracy and nmi to file
		# pred_true = pred_to_true_clustering(true_frame_labels_all, pred_frame_labels_all) # This is the same as v-measure lol
		nmi = tot_nmi(true_frame_labels_all, pred_frame_labels_all)
		ami = tot_ami(true_frame_labels_all, pred_frame_labels_all)
		bound_precision = boundary_precision(true_frame_labels_all, pred_frame_labels_all)
		bound_recall = boundary_precision(pred_frame_labels_all, true_frame_labels_all)
		bound_f1 = 2 * bound_precision * bound_recall / (bound_precision + bound_recall)
		# with open(os.path.join(model_output_dir, "eval-stats-{}.txt".format(os.path.basename(model_path))), "w") as f1:
		# 	f1.write("pred_true, nmi, bound_acc\n")
		# 	f1.write("{}, {}, {}\n".format(pred_true, nmi, bound_acc))

		return(nmi, ami, bound_precision, bound_recall, bound_f1)


def edit_op_analysis(edit_ops_all, n_true_types, output_file):

	edit_op_counts = dict()
	for utt in edit_ops_all:
		for op in utt:
			edit_op_counts[op] = edit_op_counts.get(op, 0) + 1

	sub_counts = [ (v,k) for k,v in edit_op_counts.items() if k[0]=='SUB' and v > 0]
	ib_counts  = [ (v,k) for k,v in edit_op_counts.items() if k[0]=='IB' and v > 0]
	it_counts  = [ (v,k) for k,v in edit_op_counts.items() if k[0]=='IT' and v > 0]

	sub_sum = sum([ x[0] for x in sub_counts ])
	ib_sum = sum([ x[0] for x in ib_counts ])
	it_sum = sum([ x[0] for x in it_counts ])

	sub_counts.sort(reverse=True)
	ib_counts.sort(reverse=True)
	it_counts.sort(reverse=True)

	with open(output_file, 'w') as f1:
		f1.write("{} substitutes ({}%)\n".format(sub_sum, sub_sum/(sub_sum+ib_sum+it_sum)))
		f1.write("{} insert bottoms ({}%)\n".format(ib_sum, ib_sum/(sub_sum+ib_sum+it_sum)))
		f1.write("{} insert tops ({}%)\n".format(it_sum, it_sum/(sub_sum+ib_sum+it_sum)))
		f1.write("\n\n")

		for i in range(n_true_types):

			relevant_subs = [ (v,k) for v,k in sub_counts if k[1]==i ]
			relevant_ibs = [ (v,k) for v,k in ib_counts if k[1]==i ]
			relevant_its = [ (v,k) for v,k in it_counts if k[1]==i ]

			relevant_sum = sum([ x[0] for x in relevant_subs ]) + \
				sum([ x[0] for x in relevant_ibs ]) + \
				sum([ x[0] for x in relevant_its ])

			f1.write('\n\nMost common operations for phone "{}":\n'.format(int_to_phone[i]))
			for v,k in relevant_subs[:10]:
				f1.write('    SUB {} ({}%)\n'.format(k[2], (v/relevant_sum)*100))
			for v,k in relevant_ibs[:10]:
				f1.write('    IB {} ({}%)\n'.format(k[2], (v/relevant_sum)*100))
			for v,k in relevant_its[:10]:
				f1.write('    IT {} ({}%)\n'.format(k[2], (v/relevant_sum)*100))


def correspondance_analysis(true_frame_labels_all, pred_frame_labels_all, output_file):

	n_true_types = max([ max(x) for x in true_frame_labels_all ]) + 1
	n_pred_types = max([ max(x) for x in pred_frame_labels_all ]) + 1

	correspondances = np.zeros((n_true_types, n_pred_types))

	for utt_index, utt in enumerate(true_frame_labels_all):
		for frame_index, frame in enumerate(utt):
			correspondances[frame, pred_frame_labels_all[utt_index][frame_index]] += 1

	with open(output_file, 'w') as f1:
		for true_type in range(correspondances.shape[0]):
			corresponds_to = list(correspondances[true_type,:])
			total = np.sum(corresponds_to)
			corresponds_to_pairs = [ (x,i) for i, x in enumerate(corresponds_to) if x > 0 ]
			corresponds_to_pairs.sort(reverse=True)

			f1.write('Phone "{}" (n={}) corresponds to discovered PLU labels...\n'.format(int_to_phone[true_type], total))
			[ f1.write('    {}, n={}, {}%\n'.format(pred_type, count, (count/total)*100)) for count, pred_type in corresponds_to_pairs[:10] ]
			f1.write('\n')

		f1.write('\n\n====================================================================\n\n')

		for pred_type in range(correspondances.shape[1]):
			corresponds_to = list(correspondances[:, pred_type])
			total = np.sum(corresponds_to)
			corresponds_to_pairs = [ (x,i) for i, x in enumerate(corresponds_to) ]
			corresponds_to_pairs.sort(reverse=True)

			f1.write('Discovered PLU {} (n={}) corresponds to phones...\n'.format(pred_type, total))
			[ f1.write('    "{}", n={}, {}%\n'.format(int_to_phone[true_type], count, (count/total)*100)) for count, true_type in corresponds_to_pairs[:10] ]
			f1.write('\n')


def read_tg(path, n_frames):

	tg_frame_labels = [None] * n_frames

	t = tg.TextGrid()
	t.read(path)
	# get phone tier
	try:
		phone_tier = t.getList("phones")[0]
	except IndexError:
		try:
			phone_tier = t.getList("Phones")[0]
		except IndexError:
			phone_tier = t.getList("None")[0]
	#print(t.__dict__)
	audio_len = float(t.maxTime) - float(t.minTime)
	frame_length = audio_len/n_frames
	
	for frame_index in range(n_frames):
		time = frame_index * frame_length
		interval = phone_tier.intervals[phone_tier.indexContaining(time)]
		if interval is None:
			interval_label = None
		else:
			interval_label = interval.mark
		tg_frame_labels[frame_index] = interval_label
		time += frame_length

	tg_frame_labels = [phone_to_int[x] for x in tg_frame_labels]

	return tg_frame_labels







if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("model_path",  help="path to directory containing pickled model files")
	parser.add_argument("audio_dir",  help="path to audio directory to evaluate on")
	args = parser.parse_args()
	
	evaluate_model(args.model_path, args.audio_dir)