import re
import numpy as np
import sys
import os
import textgrid as tg


def frame_scores(y_true_path, y_pred_path):
	length, y_true_phones = read_tg(y_true_path)
	_, y_pred_phones = read_tg(y_pred_path)
	y_true = seg_to_one_hot(y_true_phones,length)
	y_pred = seg_to_one_hot(y_pred_phones,length)

	true_pos = np.dot(y_pred, y_true)
	true_neg = np.dot(np.ones_like(y_pred)-y_pred, np.ones_like(y_true)-y_true)
	false_pos = np.dot(np.ones_like(y_pred)-y_pred, y_true)
	false_neg = np.dot(y_pred, np.ones_like(y_true)-y_true)
    
	accuracy = (true_pos+true_neg)/len(y_pred)
	precision = true_pos/(true_pos+false_pos)
	recall = true_pos/(true_pos+false_neg)
    
	f1 = 2*(precision*recall)/(precision+recall)
    
	return (accuracy, recall, precision, f1)


def seg_to_one_hot(tier, l):
	"""
	take textgrid tier, convert to 1-hot vector by 10ms frames, 
	where there's a 1 if there is a boundary in that 
	10ms frame, 0 otherwise
	"""
	vec = np.zeros(int(np.rint(l*100))+1)
	
	for i,interval in enumerate(tier):
		print(interval.mark)
		ms_min, ms_max =  float(interval.minTime)*100, float(interval.maxTime)*100
		print(float(interval.minTime),float(interval.maxTime) )
		print(ms_min, ms_max)
		start_idx = int(np.rint(ms_min))
		end_idx = int(np.rint(ms_max))
		print(start_idx, end_idx)
		vec[start_idx] =1
		vec[end_idx] = 1
	return vec


def read_tg(path):
	t = tg.TextGrid()
	t.read(path)
	# get phone tier
	phones = t.getList("phones")[0]
	print(t.__dict__)
	audio_len = float(t.maxTime) - float(t.minTime)
	return audio_len, phones





if __name__ == '__main__':
	tg_path = "/Volumes/data/corpora/Librispeech_360/30/30-4445-0000.TextGrid"
	length, phones = read_tg(tg_path)
	print(seg_to_one_hot(phones, length))

	sanity_check = frame_scores("/Volumes/data/corpora/Librispeech_360/30/30-4445-0000.TextGrid", "/Volumes/data/corpora/Librispeech_360/30/30-4445-0000.TextGrid")
	print("sanity_check: ", sanity_check)

	# data_check = frame_scores("../audio/")




