import argparse 
import pickle
import os
import numpy as np
import textgrid as tg
import amdtk

from .cluster import segmentation_performance, pred_to_true_clustering
from .nmi import avg_nmi
from ..io import read_htk

def frame_labeling_accuracy(model_frame_labels, gold_standard_frame_labels, model_to_gold):
	num_frames = sum([len(x) for x in model_frame_labels])
	num_correct = 0
	for model_utterance, gold_utterance in zip(model_frame_labels, gold_standard_frame_labels):
		for model_frame, gold_frame in zip(model_utterance, gold_utterance):
			if model_to_gold[model_frame] == gold_frame:
				num_correct += 1
	return float(num_correct)/num_frames

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

	zipped_paths = zip(sorted(fea_paths), sorted(top_paths), sorted(textgrid_paths))

	zipped_paths_list = list(zipped_paths)

	for model, model_path in models:

		model_output_dir = os.path.join(output_dir, 'model_{}'.format(os.path.basename(model_path)))
		os.mkdir(model_output_dir)

		pred_frame_labels_all = []
		true_frame_labels_all = []

		print("Evaluating model",model_path)

		# Mapping from text labels in textgrid to numeric phone identifiers
		mapping = {}

		# Decode the data using the model
		for (fea_path, top_path, textgrid_path) in zipped_paths_list:

			print("HELLO")

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
			true_frame_labels, mapping = np.array(read_tg(textgrid_path, len(pred_frame_labels), mapping))

			# print("---")
			# print("True labels from textgrid", textgrid_path, ":")
			# print(true_frame_labels)
			# print("true shape:", true_frame_labels.shape)

			assert(len(pred_frame_labels)==len(true_frame_labels))

			pred_frame_labels_all.append(pred_frame_labels)
			true_frame_labels_all.append(true_frame_labels)


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

		correspondance_analysis(true_frame_labels_all, pred_frame_labels_all, mapping, correspond_output_file)

		# print(pred_to_true_clustering(true_frame_labels_all, pred_frame_labels_all))

		# print(avg_nmi(true_frame_labels_all, pred_frame_labels_all))
		# return(segmentation_performance(true_frame_labels_all, pred_frame_labels_all),\
		#  pred_to_true_clustering(true_frame_labels_all, pred_frame_labels_all),
		#  avg_nmi(true_frame_labels_all, pred_frame_labels_all))

		# Write clustering accuracy and nmi to file
		pred_true = pred_to_true_clustering(true_frame_labels_all, pred_frame_labels_all)
		nmi = avg_nmi(true_frame_labels_all, pred_frame_labels_all)
		with open(os.path.join(model_output_dir, "eval-stats-{}.txt".format(os.path.basename(model_path))), "w") as f1:
			f1.write("pred_true, nmi\n")
			f1.write("{},{}\n".format(pred_true, nmi))

		return(pred_true, nmi)



def correspondance_analysis(true_frame_labels_all, pred_frame_labels_all, mapping, output_file):

	print('mapping: ', mapping)
	reverse_mapping = {v: k for k, v in mapping.items()}

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
			corresponds_to_pairs = [ (x,i) for i, x in enumerate(corresponds_to) ]
			corresponds_to_pairs.sort(reverse=True)

			f1.write('Phone "{}" (n={}) corresponds to discovered PLU labels...\n'.format(reverse_mapping[true_type], total))
			[ f1.write('    {}, n={}, {}%\n'.format(pred_type, count, count/total)) for count, pred_type in corresponds_to_pairs ]
			f1.write('\n')


def read_tg(path, n_frames, mapping):

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

	types = set(tg_frame_labels)
	if len(mapping.values()) == 0:
		curr_index = 0
	else:
		curr_index = max(mapping.values())+1
	for phone_type in types:
		if phone_type not in mapping:
			mapping[phone_type] = curr_index
			print('inserting {}:{} into mapping'.format(phone_type, curr_index))
			curr_index += 1

	tg_frame_labels = [mapping[x] for x in tg_frame_labels]

	return tg_frame_labels, mapping







if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument("model_path",  help="path to directory containing pickled model files")
	parser.add_argument("audio_dir",  help="path to audio directory to evaluate on")
	args = parser.parse_args()
	
	evaluate_model(args.model_path, args.audio_dir)