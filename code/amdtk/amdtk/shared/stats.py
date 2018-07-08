
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
