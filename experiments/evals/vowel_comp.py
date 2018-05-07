from matplotlib import pyplot as plt 
import numpy as np 
import textgrid as tg 
import re
import os
import sys
from conch import analyze_segments
from conch.analysis.segments import SegmentMapping
from conch.analysis.formants import PraatSegmentFormantTrackFunction, FormantTrackFunction, \
    PraatSegmentFormantPointFunction
from conch.analysis.praat import PraatAnalysisFunction
from pyraat.parse_outputs import parse_point_script_output



# want f1, f2 for each discovered vowel 
# plan: get a list of vowel intervals from the gold-standard data
	# get the gold-standard formant values
# for each vowel in the gold standard, get corresponding time slice from the aligned data
	# get formant values for that vowel
# average formants over vowels, plot on vowel chart 


# problem: how to determine which segments in the aligned are vowels
# solution 1: get formants for all segments, based on formants decide which should be considered vowels


vowel_gex = re.compile(r"[aeiou].*")

def get_vowel_formants(tg_path, audio_path, reference=True):
	"""
	read a gold-standard textgrid, return all vowel intervals 
	"""
	vowels = SegmentMapping()
	# channel = TODO: find out channel
	padding = .25

	ref_tg = tg.TextGrid()

	ref_tg.read(tg_path)

	for tier in ref_tg.tiers:
		if re.match(r"[pP]hones?", tier.name) is not None:
			for i, interval in enumerate(tier.intervals):
				if vowel_gex.match(interval.mark) is not None:
					# vowels.append(interval)
					begin = interval.minTime
					end = interval.maxTime
					label = interval.mark

					vowels.add_file_segment(audio_path, begin, end, label=label, channel=0,  padding=padding)

	max_freq = 5500
	formant_function = PraatSegmentFormantPointFunction(praat_path="/Applications/Praat.app/Contents/MacOS/Praat",
														max_frequency=max_freq, num_formants=5, window_length=0.025,
														time_step=0.01)
	output = analyze_segments(vowels, formant_function) 
	return output

def plot(ref_vowels, found_vowels):
	small_d = dict()
	# d = list(ref_vowels.items())[0][0]
	for key, value in ref_vowels.items():
		label = key.__dict__["properties"]['label']
		f1 = value["F1"]
		f2 = value["F2"]
		try: 
			small_d[label].append([f1, f2])
		except KeyError:
			small_d[label] = [[f1, f2]]

	# average
	for key,value in small_d.items():
		value = np.array(value)
		avg_value = np.sum(value, axis=0)/value.shape[0]
		small_d[key] = avg_value

	sorted_items = sorted(small_d.items(), key=lambda x: x[0])
	labels = [x[0] for x in sorted_items]
	formants = [x[1] for x in sorted_items]
	f1 = [x[0] for x in formants]
	f2 = [x[1] for x in formants]
	# plot formants
	plt.scatter(f1, f2)

	for item in sorted_items:
		label = item[0]
		f1 = item[1][0]
		f2 = item[1][1]
		plt.annotate(label, xy=(f1, f2))

	plt.show()




if __name__ == '__main__':
	ref_vowels = get_vowel_formants("/Users/Elias/ULD/audio/icicles/icicles.TextGrid", "/Users/Elias/ULD/audio/icicles/icicles.wav")
	prod_vowels = get_vowel_formants("/Users/Elias/ULD/audio/icicles/")

