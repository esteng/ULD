
# Shared functions for accumulating mean and variance statistics of the data

def collect_data_stats(filename):
	"""Job to collect the statistics."""
	# We  re-import this module here because this code will run
	# remotely.
	import amdtk
	data = amdtk.read_htk(filename)
	stats_0 = data.shape[0]
	stats_1 = data.sum(axis=0)
	stats_2 = (data**2).sum(axis=0)
	retval = (
		stats_0,
		stats_1,
		stats_2
	)
	return retval


def collect_data_stats_by_speaker(filename):

	import os

	stats = collect_data_stats(filename)
	speaker = os.path.split(os.path.split(filename)[0])[1]

	return (speaker, stats)


def accumulate_stats(data_stats):
	n_frames = data_stats[0][0]
	mean = data_stats[0][1]
	var = data_stats[0][2]
	for stats_0, stats_1, stats_2 in data_stats[1:]:
		n_frames += stats_0
		mean += stats_1
		var += stats_2
	mean /= n_frames
	var = (var / n_frames) - mean**2

	data_stats = {
		'count': n_frames,
		'mean': mean,
		'var': var/20
	}
	return data_stats

def accumulate_stats_by_speaker(data_stats):

	data_stats_dict = {}

	for speaker, stats in data_stats:
		if speaker in data_stats_dict:
			data_stats_dict[speaker].append(stats)
		else:
			data_stats_dict[speaker] = [stats]


	data_stats_accumulated_by_speaker = {}

	for speaker in data_stats_dict.keys():
		data_stats_accumulated_by_speaker[speaker] = accumulate_stats(data_stats_dict[speaker])
	
	return data_stats_accumulated_by_speaker
