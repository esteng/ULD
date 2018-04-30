
"""
Training algorithms vor various models.

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

"""

import abc
import time
import numpy as np
import os
from ipyparallel.util import interactive
import _pickle as pickle
from amdtk import read_htk


class Optimizer(metaclass=abc.ABCMeta):

	def __init__(self, dview, data_stats, args, model):
		self.dview = dview
		self.epochs = int(args.get('epochs', 1))
		self.batch_size = int(args.get('batch_size', 2))
		self.pkl_path = args.get("pkl_path", None)
		self.log_dir = args.get("log_dir", None)
		if self.log_dir is not None:
			self.log_file = self.log_dir  + "/" + time.strftime("opt_%Y-%m-%d_%H:%M.log")

		# print("log dir: ", self.log_dir, "log file", self.log_file)
		self.model = model
		self.time_step = 0
		self.data_stats = data_stats

		# with self.dview.sync_imports():
		# 	import numpy
		# 	from amdtk import read_htk
		# 	import _pickle as pickle
		# 	import os

		# self.dview.push({
		# 	'data_stats': data_stats
		# })

	def run(self, data, callback):
		import _pickle as pickle
		start_time = time.time()

		for epoch in range(self.epochs):

			# Shuffle the data to avoid cycle in the training.
			np_data = np.array(data, dtype=object)
			idxs = np.arange(0, len(data))
			np.random.shuffle(idxs)
			shuffled_data = np_data[idxs]
			if self.batch_size < 0:
				batch_size = len(data)
			else:
				batch_size = self.batch_size

			for mini_batch in range(0, len(data), batch_size):
				self.time_step += 1

				# Index of the data mini-batch.
				start = mini_batch
				end = mini_batch + batch_size

				# Reshaped the list of features.
				fea_list = shuffled_data[start:end]
				# print("batch_size: ", batch_size)
				# print("len(self.dview): ", len(self.dview))
				n_utts = batch_size // len(self.dview)
				# print("n_utts: ", n_utts)
				new_fea_list = [fea_list[i:i + n_utts]  for i in
								range(0, len(fea_list), n_utts)]
				# Update the model.
				objective = \
					self.train(new_fea_list, epoch + 1, self.time_step)

				# pickle model
				if self.pkl_path is not None:
					if not os.path.exists(self.pkl_path):
						os.makedirs(self.pkl_path)
					path_to_file = os.path.join(self.pkl_path,"epoch-{}-batch-{}".format(epoch, mini_batch))
					with open(path_to_file, "wb") as f1:
						pickle.dump(self.model, f1)


				# write to log
				if self.log_dir is not None:

					# print("writing to log file")
					with open(self.log_file, "a") as f1:
						f1.write(",".join([str(x) for x in [epoch+1, int(mini_batch / batch_size) + 1, objective]]))
						f1.write("\n")
				# Monitor the convergence.
				callback({
					'epoch': epoch + 1,
					'batch': int(mini_batch / batch_size) + 1,
					'n_batch': int(np.ceil(len(data) / batch_size)),
					'time': time.time() - start_time,
					'objective': objective,
				})

	@abc.abstractmethod
	def train(self, data, epoch, time_step):
		pass


class StochasticVBOptimizer(Optimizer):

	@staticmethod
	@interactive
	def e_step(args_list):
		exp_llh = 0.
		acc_stats = None
		n_frames = 0

		for arg in args_list:
			fea_file = arg

			# Mean / Variance normalization.
			data = read_htk(fea_file)
			data -= data_stats['mean']
			data /= numpy.sqrt(data_stats['var'])

			# Get the accumulated sufficient statistics for the
			# given set of features.
			s_stats = model.get_sufficient_stats(data)
			posts, llh, new_acc_stats = model.get_posteriors(s_stats,
															 accumulate=True)

			exp_llh += numpy.sum(llh)
			n_frames += len(data)
			if acc_stats is None:
				acc_stats = new_acc_stats
			else:
				acc_stats += new_acc_stats

		return (exp_llh, acc_stats, n_frames)


	def __init__(self, dview, data_stats, args, model):
		Optimizer.__init__(self, dview, data_stats, args, model)
		self.lrate = float(args.get('lrate', 1))

	def train(self, fea_list, epoch, time_step):
		# Propagate the model to all the remote clients.
		# print("TRAINING")
		self.dview.push({
			'model': self.model,
		})

		# Parallel accumulation of the sufficient statistics.
		stats_list = self.dview.map_sync(StochasticVBOptimizer.e_step,
										 fea_list)

		# Accumulate the results from all the jobs.
		exp_llh = stats_list[0][0]
		acc_stats = stats_list[0][1]
		n_frames = stats_list[0][2]
		for val1, val2, val3 in stats_list[1:]:
			exp_llh += val1
			acc_stats += val2
			n_frames += val3

		# print("getting kl")
		kl_div = self.model.kl_div_posterior_prior()
		# print("kl div:")
		# print(kl_div)
		# Scale the statistics.
		scale = self.data_stats['count'] / n_frames
		# print('acc_stats[2] before scaling: ', acc_stats[2])
		# print("scaling by {}".format(scale))
		acc_stats *= scale
		# print('acc_stats[2] after scaling: ', acc_stats[2])
		# print("updating model with ")
		# print(acc_stats)
		self.model.natural_grad_update(acc_stats, self.lrate)
		# print("returning")
		# print("scale")
		# print(scale)
		# print("exp_llh")
		# print(exp_llh)
		# print("kl_div")
		# print(kl_div)
		# print("self.data_stats[count]")
		# print(self.data_stats['count'])
		return (scale * exp_llh - kl_div) / self.data_stats['count']


class NoisyChannelOptimizer(Optimizer):

	def __init__(self, dview, data_stats, args, model):
		Optimizer.__init__(self, dview, data_stats, args, model)
		self.lrate = float(args.get('lrate', 1))


	# I think we can just use the superclass implementation Optimizer.run(),
	# because the real difference comes in train()
	# def run(self, data, callback):

	def e_step_nonstatic(self, args_list):

		# print(type(self))

		model = self.model
		data_stats = self.data_stats

		exp_llh = 0.
		acc_stats = None
		n_frames = 0

		for arg in args_list:
			(fea_file, top_file) = arg

			# Mean / Variance normalization.
			data = read_htk(fea_file)
			data -= data_stats['mean']
			data /= np.sqrt(data_stats['var'])

			# Read top PLU sequence from file
			with open(top_file, 'r') as f:
				topstring = f.read()
				tops = topstring.strip().split(',')
				tops = [int(x) for x in tops]

			# Get the accumulated sufficient statistics for the
			# given set of features.
			s_stats = model.get_sufficient_stats(data)
			posts, llh, new_acc_stats = model.get_posteriors(s_stats, tops,accumulate=True, filename=fea_file)

			exp_llh += np.sum(llh)
			n_frames += len(data)
			if acc_stats is None:
				acc_stats = new_acc_stats
			else:
				acc_stats += new_acc_stats

		return (exp_llh, acc_stats, n_frames)    

	def train(self, fea_list, epoch, time_step):

		# Propagate the model to all the remote clients.
		# self.dview.push({
		# 	'model': self.model,
		# })


		# Parallel accumulation of the sufficient statistics.
		# stats_list = self.dview.map_sync(NoisyChannelOptimizer.e_step,
		#                                 fea_list)

		# Serial version
		stats_list = []
		for pair in fea_list:
			stats_list.append(self.e_step_nonstatic(pair))

		import time
		# Accumulate the results from all the jobs.
		exp_llh = stats_list[0][0]
		acc_stats = stats_list[0][1]


			

		# print("accumulating states from a batch:")
		# print("exp_llh is ")
		# print(exp_llh)
		# print("acc-stats is ")
		# print(acc_stats)

		n_frames = stats_list[0][2]
		for val1, val2, val3 in stats_list[1:]:
			exp_llh += val1
			acc_stats += val2
			n_frames += val3

		kl_div = self.model.kl_div_posterior_prior()


		# Scale the statistics.
		scale = self.data_stats['count'] / n_frames
		acc_stats *= scale
		self.model.natural_grad_update(acc_stats, self.lrate)

		return (scale * exp_llh - kl_div) / self.data_stats['count']


	@staticmethod
	@interactive
	def e_step(args_list):
		import os
		from amdtk import read_htk

		exp_llh = 0.
		acc_stats = None
		n_frames = 0

		for arg in args_list:
			(fea_file, top_file) = arg
			# Mean / Variance normalization.
			data = read_htk(fea_file)
			data -= data_stats['mean']

			data /= numpy.sqrt(data_stats['var'])
			# Read top PLU sequence from file
			with open(top_file, 'r') as f:
				data_str = f.read()
				tops = data_str.strip().split(',')
				tops = [int(x) for x in tops]
			# Get the accumulated sufficient statistics for the
			# given set of features.
			s_stats = model.get_sufficient_stats(data)
			posts, llh, new_acc_stats = model.get_posteriors(s_stats, tops,
															 accumulate=True, filename=fea_file)

			exp_llh += numpy.sum(llh)

			n_frames += len(data)
			if acc_stats is None:
				acc_stats = new_acc_stats
			else:
				acc_stats += new_acc_stats

		return (exp_llh, acc_stats, n_frames)


class ToyNoisyChannelOptimizer(Optimizer):

	def __init__(self, dview, data_stats, args, model, dir_path):
		Optimizer.__init__(self, dview, data_stats, args, model)
		self.lrate = float(args.get('lrate', 1))
		self.dir_path = dir_path

	def e_step_nonstatic(self, args_list):


		model = self.model
		data_stats = self.data_stats

		exp_llh = 0.
		acc_stats = None
		n_frames = 0

		for arg in args_list:

			(data, tops) = arg

			print('----- top sequence', tops, '. data len =', data.shape[0], '-----')

			# Mean / Variance normalization.
			data -= data_stats['mean']
			data /= np.sqrt(data_stats['var'])


			# Get the accumulated sufficient statistics for the
			# given set of features.
			s_stats = model.get_sufficient_stats(data)
			posts, llh, new_acc_stats = model.get_posteriors(s_stats, tops,accumulate=True, filename="test")

			exp_llh += np.sum(llh)
			n_frames += len(data)
			if acc_stats is None:
				acc_stats = new_acc_stats
			else:
				acc_stats += new_acc_stats

		return (exp_llh, acc_stats, n_frames) 


	def run(self, data, callback):
		"""Run the Standard Variational Bayes training.

		Parameters
		----------
		model : :class:`PhoneLoop`
			Phone Loop model to train.

		"""
		start_time = time.time()
		for epoch in range(self.epochs):

			print('========== EPOCH', epoch, '==========')

			# Create a temporary directory.
			self.temp_dir = os.path.join(self.dir_path, 'epoch' +
										 str(epoch + 1))
			os.makedirs(self.temp_dir, exist_ok=True)

			# # Set the pruning thresold.
			# if epoch == 0:
			#     model.pruning_threshold = self.initial_pruning
			# else:
			#     model.pruning_threshold = self.pruning

			# Perform one epoch of the training.
			lower_bound = self.train(data, epoch+1, epoch+1)

			# Monitor the convergence after each epoch.
			args = {
				'model': self.model,
				'lower_bound': lower_bound,
				'epoch': epoch + 1,
				'tmpdir': self.dir_path,
				'time': time.time() - start_time
			}
			callback(args)


	def train(self, data_list, epoch, time_step):

		# Propagate the model to all the remote clients.
		# self.dview.push({
		# 	'model': self.model,
		# })

		# Parallel accumulation of the sufficient statistics.
		# stats_list = self.dview.map_sync(ToyNoisyChannelOptimizer.e_step,
		# 								data_list)


		# Serial version
		stats_list = []
		# for pair in data_list:
		stats_list.append(self.e_step_nonstatic(data_list))

		# Accumulate the results from all the jobs.
		exp_llh = stats_list[0][0]
		acc_stats = stats_list[0][1]

		# for s in acc_stats._stats:
		# 	print(s)

		n_frames = stats_list[0][2]
		for val1, val2, val3 in stats_list[1:]:
			exp_llh += val1
			acc_stats += val2
			n_frames += val3

		kl_div = self.model.kl_div_posterior_prior()

		# Scale the statistics.
		scale = self.data_stats['count'] / n_frames
		acc_stats *= scale

		# print("exp_llh: ", str(exp_llh))
		# print("kl_div: ", str(kl_div))
		# print("datastats[count]: ", str(self.data_stats["count"]))

		self.model.natural_grad_update(acc_stats, self.lrate)

		return (scale * exp_llh - kl_div) / self.data_stats['count']

	@staticmethod
	@interactive
	def e_step(args_list):
		import os
		from amdtk import read_htk



		exp_llh = 0.
		acc_stats = None
		n_frames = 0

		for data, tops in [args_list]:
			
			# Get the accumulated sufficient statistics for the
			# given set of features.
			s_stats = model.get_sufficient_stats(data)
			posts, llh, new_acc_stats = model.get_posteriors(s_stats, tops,
															 accumulate=True, filename="test")

			exp_llh += numpy.sum(llh)
			n_frames += len(data)
			if acc_stats is None:
				acc_stats = new_acc_stats
			else:
				acc_stats += new_acc_stats

		return (exp_llh, acc_stats, n_frames)




