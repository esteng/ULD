from sklearn.metrics.cluster import normalized_mutual_info_score
import textgrid as tg 
import numpy as np 
import os
import sys
import re


def tg_to_cluster(tg_path, mapping=None):
	grid = tg.TextGrid().read(tg_path)
	phone_tier = None
	for tier_name in ["phones", "Phones", "None"]:
		try:
			phone_tier = grid.getList(tier_name)[0]
			break
		except IndexError:
			continue
	if phone_tier == None:
		print("Error: textgrid {} does not have a valid tier".format(tg_path))
		sys.exit(1)
	phone_list = []
	for interval in phone_tier.intervals:
		if mapping == None:
			phone_type = interval.mark
		else:
			phone_type = mapping[interval.mark]
		phone_duration = interval.end-interval.begin
		n_phones = int(phone_duration/10)
		phone_list.extend([phone_type]*n_phones)
	return phone_list

def nmi(true_list, pred_list):
	"""
	get the normalized mutual information between the predicted and the true alignment 
	"""
	true_list = [int(x) for x in true_list]
	pred_list = [int(x) for x in pred_list]
	return normalized_mutual_info_score(true_list, pred_list)


def avg_nmi(all_true, all_pred):
	all_phones = set()
	for true_clusters in all_true:
		all_phones |= set(true_clusters)
	mapping = {x:i for i, x in enumerate(all_phones)}
	all_true = [[mapping[x] for x in true_clusters] for true_clusters in all_true]

	nmis = []
	for true_clusters, pred_clusters in zip(all_true, all_pred):	
		nmis.append(nmi(true_clusters, pred_clusters))

	return sum(nmis)/len(nmis)



