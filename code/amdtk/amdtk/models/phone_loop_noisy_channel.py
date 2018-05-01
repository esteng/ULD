"""
Main class of the phone loop model.

Copyright (C) 2017, Lucas Ondel

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
import sys
from bisect import bisect
from itertools import groupby
from scipy.special import logsumexp
import _pickle as pickle

from profilehooks import profile
import math

from .hmm_utils import create_phone_loop_transition_matrix
from .hmm_utils import forward_backward
from .hmm_utils import viterbi
from .model import EFDStats, DiscreteLatentModel
from ..densities import Dirichlet, NormalGamma, NormalDiag

from collections import defaultdict

class Ops(object):
	IB = 0
	IT = 1
	SUB = 2
	NONE = 3
	CODES = [IB, IT, SUB, NONE]

	def to_string(op):
		if op==Ops.IB:
			return "IB"
		if op==Ops.IT:
			return "IT"
		if op==Ops.SUB:
			return "SUB"
		if op==Ops.NONE:
			return "NONE"
		return None

class PhoneLoopNoisyChannel(DiscreteLatentModel):
	"""Bayesian Phone Loop model with noisy channel addition.

	Bayesian Phone Loop with a sequence of top-level PLUs
	and a Dirichlet prior over edit operations.

	"""


	def create(n_units, n_states, n_comp_per_state, n_top_units, max_slip_factor, mean, var):
		"""Create and initialize a Bayesian Phone Loop Model.

		Parameters
		----------
		n_units : int
			Number of acoustic units i.e. phones.
		n_states : int
			Number of states for each acoustic unit.
		n_comp_per_state : int
			Number of compent per emission.
		n_top_units : int
			Size of top-level PLU alphabet
		max_slip_factor : float
			Max fraction of the top PLU sequence length by which 
			the bottom PLU index is allowed to differ from the top PLU index
		mean : numpy.ndarray
			Mean of the data set to train on.
		var : numpy.ndarray
			Variance of the data set to train on.

		Returns
		-------
		model : :class:`PhoneLoop`
			A new phone-loop model.

		"""
		tot_n_states = n_units * n_states
		tot_comp = tot_n_states * n_comp_per_state

		# Initialize the Dirichlets over operations for each unit
		num_ops = 1 + ( 2 * n_units )
		op_latent_priors = [Dirichlet(np.ones(num_ops)) for _ in range(n_top_units)]
		op_latent_posteriors = [Dirichlet(np.ones(num_ops)) for _ in range(n_top_units)]

		# Initialize the priors over Gaussian component choice within HMM states
		state_priors = [Dirichlet(np.ones(n_comp_per_state))
						for _ in range(tot_n_states)]
		state_posteriors = [Dirichlet(np.ones(n_comp_per_state))
							for _ in range(tot_n_states)]

		# Initialize the priors over Gaussian parameters 
		priors = []
		prior_mean = mean.copy()
		prior_var = var.copy()
		for i in range(tot_comp):

			prior = NormalGamma(
				prior_mean,
				np.ones_like(mean),
				np.ones_like(var),
				prior_var,
			)
			priors.append(prior)

		components = []
		cov = np.diag(prior_var)
		for i in range(tot_comp):
			s_mean = np.random.multivariate_normal(mean, cov)
			posterior = NormalGamma(
				s_mean,
				np.ones_like(mean),
				np.ones_like(var),
				prior_var
			)
			components.append(NormalDiag(priors[i], posterior))

		return PhoneLoopNoisyChannel(op_latent_priors, op_latent_posteriors,
						 state_priors, state_posteriors, components, max_slip_factor)

	def __init__(self, op_latent_priors, op_latent_posteriors,
						 state_priors, state_posteriors, components, max_slip_factor):

		# Ok I think we're not gonna do this here, we're just gonna implement
		# our own version of the DiscreteLatentModel functions (because we have
		# many more distributions over which we are doing inference)
		#DiscreteLatentModel.__init__(self, latent_prior, latent_posterior, components)
		self._components = components
		self._exp_np_matrix = self._get_components_params_matrix()

		self.n_units = int((len(op_latent_priors[0].natural_params)-1)/2)
		self.n_states = len(state_priors) // self.n_units
		self.n_comp_per_states = len(state_priors[0].natural_params)
		self.n_top_units = len(op_latent_priors)
		self.max_slip_factor = max_slip_factor

		self.op_latent_priors = op_latent_priors
		self.op_latent_posteriors = op_latent_posteriors

		self.state_priors = state_priors
		self.state_posteriors = state_posteriors

		self.p_threshold = float("-inf")

		self.post_update()

		self.update_renorms()

	def update_renorms(self):
		# Calculate all the renormalized operation distributions (by previous bottom PLU)
		# Don't allow same 2 bottom PLUs in a row
		self.renorms = [[None for _ in range(self.n_units)] for __ in range(self.n_top_units)]

		for i in range(self.n_top_units):
			dist = self.op_latent_posteriors[i].grad_log_partition
			# print(dist)
			for j in range(self.n_units):
				renorm_dist = np.copy(dist)
				# Set insert-bottom log-probability of previous phone to -inf (exp(-inf)=0)
				renorm_dist[j+1] = float('-inf')
				# Set substitute log-probability of previous phone to -inf (exp(-inf)=0)
				renorm_dist[j+1+self.n_units] = float('-inf')
				# Renormalize
				renorm_dist = renorm_dist - logsumexp(renorm_dist)
				self.renorms[i][j] = renorm_dist

		#print("self.renorms: ",self.renorms)

	def post_update(self):
		DiscreteLatentModel.post_update(self)

		# Update the states' weights.
		self.state_log_weights = np.zeros((self.n_units * self.n_states,
										   self.n_comp_per_states))
		for idx in range(self.n_units * self.n_states):
				self.state_log_weights[idx, :] = \
					self.state_posteriors[idx].grad_log_partition

	 
	def _get_state_llh(self, s_stats):
		# Evaluate the Gaussian log-likelihoods.
		exp_llh = self.components_exp_llh(s_stats)

		# Reshape the log-likelihood to get the per-state and per
		# component log-likelihood.
		r_exp_llh = exp_llh.reshape(self.n_units * self.n_states,
									self.n_comp_per_states, -1)

		# Emission log-likelihood.
		c_given_s_llh = r_exp_llh + self.state_log_weights[:, :, np.newaxis]
		state_llh = logsumexp(c_given_s_llh, axis=1).T
		c_given_s_resps = np.exp(c_given_s_llh - \
			state_llh.T[:, np.newaxis, :])

		return state_llh, c_given_s_resps


	def decode(self, data, plu_tops, state_path=False, phone_intervals=False, edit_ops=False, hmm_states=False, plus=True):
		s_stats = self.get_sufficient_stats(data)

		state_llh, c_given_s_resps = self._get_state_llh(s_stats)

		n_frames = state_llh.shape[0]
		max_slip = math.ceil(len(plu_tops)*self.max_slip_factor)
		max_plu_bottom_index = len(plu_tops) + max_slip

		n_frames = state_llh.shape[0]

		log05 = math.log(0.5)

		frames_per_top = math.ceil(float(n_frames)/len(plu_tops))

		# Calculate forward probabilities WITH BACKPOINTERS
		forward_probs = {}
		# Insert starting items
		for (item, prob) in self.generate_start_items(plu_tops, state_llh):
			forward_probs[item] = (prob, None)

		for plu_bottom_index in range(-1,max_plu_bottom_index):
			# print("forward ** plu_bottom_index = "+str(plu_bottom_index))
			# print("len(forward_probs) = "+str(len(forward_probs)))
			pt_lower_limit = max(-1,plu_bottom_index-max_slip)
			pt_upper_limit = min(len(plu_tops), plu_bottom_index+max_slip)
			for plu_top_index in range(pt_lower_limit, pt_upper_limit):
				for plu_bottom_type in range(self.n_units):
					frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
					frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
					for frame_index in range(frame_lower_limit, frame_upper_limit):
						for edit_op in Ops.CODES:
							if edit_op == Ops.IT:
								hmm_range = [self.n_states-1]
							elif edit_op == Ops.IB or edit_op == Ops.SUB:
								hmm_range = [0]
							else:
								hmm_range = range(self.n_states)
							for hmm_state in hmm_range:
								curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)
								if curr_state in forward_probs:
									nexts = self.next_states((curr_state, forward_probs[curr_state][0]), plu_tops, state_llh, max_slip, frames_per_top, log05, logging=False)

									for next_state_and_prob in nexts:
										(next_state, prob) = next_state_and_prob
										if next_state in forward_probs:
											(curr_prob, (back_pointer, back_prob)) = forward_probs[next_state]
											new_prob = np.logaddexp(forward_probs[next_state][0], prob)
											if prob > back_prob:
												forward_probs[next_state] = (new_prob, (curr_state, prob))
											else:
												forward_probs[next_state] = (new_prob, (back_pointer, back_prob))
										else:
											forward_probs[next_state] = (prob, (curr_state, prob))

		# Backtrace
		# Figure out which item to start from
		end_items = self.generate_end_items(plu_tops, state_llh, max_plu_bottom_index)
		#print("End items:",end_items)
		best_item = None
		best_prob = float("-inf")
		for (item, _) in end_items:
			value = forward_probs.get(item, (float("-inf"), None))
			if value[0] > best_prob:
				best_prob = forward_probs[item][0]
				best_item = item

		# print("BEST END ITEM:")
		# print(best_item)

		back_path = [best_item]

		(prob, back_pointer_tuple) = forward_probs[best_item]
		# print("APPENDING")
		# print(prob, " ", back_pointer_tuple)

		# Do backtrace
		while back_pointer_tuple is not None:
			back_path.append(back_pointer_tuple[0])
			(prob, back_pointer_tuple) = forward_probs[back_path[-1]]
			# print("APPENDING")
			# print(prob, " ", back_pointer_tuple)

		back_path.reverse()

		#(frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

		# Get PLU list
		plu_path = [state[2] for state in back_path]
		# Get HMM state list
		hmm_state_path = [state[2]*self.n_states+state[1] for state in back_path]

		return_values = []

		if phone_intervals:
			groups = groupby(plu_path)

			interval_path = []
			begin_index = 0
			for group in groups:
				end_index = begin_index + sum(1 for item in group[1])
				interval_path.append((group[0], begin_index, end_index))
				begin_index = end_index

			return_values.append(interval_path)

		if edit_ops:
			edit_path = [ (Ops.to_string(state[4]), plu_tops[state[5]], state[2]) for state in back_path if state[4] != Ops.NONE ]
			return_values.append(edit_path)

		if hmm_states:
			return_values.append(hmm_state_path)

		if plus:
			return_values.append(plu_path)
		

		if len(return_values) == 0:
			return plu_path
		if len(return_values) == 1:
			return return_values[0]
		else:
			return return_values


	# DiscreteLatentModel interface.
	# -----------------------------------------------------------------
	
	def kl_div_posterior_prior(self):
		"""Kullback-Leibler divergence between prior /posterior.

		Returns
		-------
		kl_div : float
			Kullback-Leibler divergence.

		"""
		#retval = DiscreteLatentModel.kl_div_posterior_prior(self)

		retval = 0.
		for i in range(len(self.op_latent_posteriors)):
			retval += self.op_latent_posteriors[i].kl_div(self.op_latent_priors[i])
		# print("after adding op kls retval is ", retval)

		for comp in self.components:
			retval += comp.posterior.kl_div(comp.prior)


		for idx, post in enumerate(self.state_posteriors):
			retval += post.kl_div(self.state_priors[idx])


		return retval

	# @profile(immediate=True)
	def get_posteriors(self, s_stats, top_seq, accumulate=False,filename=None):
		import time
		# print("max s_stats:")
		# print(np.max(s_stats))
		state_llh, c_given_s_resps = self._get_state_llh(s_stats)

		# print("max state_llh at {}:".format(time.clock()))
		# print(np.max(state_llh))
		# print(logsumexp(state_llh, axis=1))
		# print(state_llh)
		# The workhorse
		
		log_op_counts_normalized, log_state_counts = self.forward_backward_noisy_channel(top_seq, state_llh, filename)
		# # get last column of log state counts, which is the normalizer (i.e. P(X_{1:T})), prob of whole frame sequence
		# log_prob_observations = logsumexp(log_state_counts[:,-1])

		# print("got posteriors for file {} ".format(filename.strip()))

		# Normalize by frame
	
		# used to be axis=0
		state_norm = logsumexp(log_state_counts, axis=0)
		log_state_counts_perframe_normalized = log_state_counts - state_norm
		state_counts_perframe_normalized = np.exp(log_state_counts_perframe_normalized)
		# ^ this is equivalent to state_resps in the normal phone loop

		# # Normalize the log op counts based on the log state counts
		# state_llh_weighted = state_llh + log_state_counts_perframe_normalized.T
		# perframe_llh = logsumexp(state_llh_weighted, axis=1)
		# #perframe_llh = logsumexp(log_state_counts.T, axis=1)
		# data_llh = sum(perframe_llh)
		# log_op_counts_normalized = [i - data_llh for i in log_op_counts]
		op_counts_normalized = [np.exp(i) for i in log_op_counts_normalized]

		if accumulate:
			tot_resps = state_counts_perframe_normalized[:, np.newaxis, :] * c_given_s_resps
			gauss_resps = tot_resps.reshape(-1, tot_resps.shape[-1])
			
			# We don't need units_stats because we have the op counts

			state_stats = tot_resps.sum(axis=2)
			gauss_stats = gauss_resps.dot(s_stats)
			efdstats = [state_stats, gauss_stats]

			efdstats.extend(op_counts_normalized)

			acc_stats = EFDStats(efdstats)

			# print("acc op stats: ", acc_stats._stats[-1])

			return state_counts_perframe_normalized, state_norm[-1], acc_stats

		return state_counts_perframe_normalized, state_norm[-1]

	def natural_grad_update(self, acc_stats, lrate):

		"""Natural gradient update."""
		state_stats = acc_stats[0]
		gauss_stats = acc_stats[1]
		op_counts = acc_stats[2:]

		# Update edit op counts for each top PLU
		for i in range(len(op_counts)):
			op_count_i = op_counts[i]
			try:
				assert(np.all(np.isfinite(op_count_i)))
			except AssertionError:
				# print("failure at ", i)
				# print(op_count_i)
				pass
				# sys.exit()
			if i == 1:
				print("op count i:")
				# print(op_count_i)
				print("log op count:")
				# print(np.log(op_count_i))
				# print('======'+str(i)+'======\n')
				# print('before update natural_params: '+str(self.op_latent_posteriors[i].natural_params))
				# print('before update grad_log_partition: '+str(self.op_latent_posteriors[i].grad_log_partition))
				# print("normalized grad log partition in linear space before")
				# normed = np.exp(self.op_latent_posteriors[i].grad_log_partition - logsumexp(self.op_latent_posteriors[i].grad_log_partition))
				# print(normed)

			op_grad = self.op_latent_priors[i].natural_params + op_count_i
			op_grad = op_grad - self.op_latent_posteriors[i].natural_params
			self.op_latent_posteriors[i].natural_params += lrate * op_grad
			if i == 1:
				print("normalized grad log partition in linear space after")
				normed = np.exp(self.op_latent_posteriors[i].grad_log_partition - logsumexp(self.op_latent_posteriors[i].grad_log_partition))
				# print(normed)
			# print('after update natural_params: '+str(self.op_latent_posteriors[i].natural_params))
			# print('after update grad_log_partition: '+str(self.op_latent_posteriors[i].grad_log_partition))

		# Update the states' weights.
		for idx, post in enumerate(self.state_posteriors):
			grad = self.state_priors[idx].natural_params + state_stats[idx]
			grad -= post.natural_params
			post.natural_params = post.natural_params + lrate * grad

		# Update Gaussian components.
		for idx, stats in enumerate(gauss_stats):
			comp = self.components[idx]
			grad = comp.prior.natural_params + stats
			grad -= comp.posterior.natural_params
			comp.posterior.natural_params = \
				comp.posterior.natural_params + lrate * grad

		self.post_update()
		self.update_renorms()


	def forward_backward_noisy_channel(self, plu_tops, state_llh, file):

		max_slip = math.ceil(len(plu_tops)*self.max_slip_factor)

		n_frames = state_llh.shape[0]

		frames_per_top = math.ceil(float(n_frames)/len(plu_tops))

		# Calculate forward probabilities
		forward_probs = {}

		
		# Insert starting items
		for (item, prob) in self.generate_start_items(plu_tops, state_llh):
			forward_probs[item] = prob

		fw_ibs = 0

		neg_inf = float('-inf')

		log05 = math.log(0.5)

		pb_upper_limit = len(plu_tops)-1 + max_slip

		rfile_lines = []

		logging = False

		pb_lower_limit = -1

		# what if we do it as an np array? 
		forward_arr = np.full((pb_upper_limit+1,len(plu_tops), self.n_units,n_frames, len(Ops.CODES), self.n_states), float('-inf'))
		backward_arr = np.full((pb_upper_limit+1, len(plu_tops), self.n_units,n_frames, len(Ops.CODES), self.n_states), float('-inf'))
		fw_pb_idxs = set()
		fw_pt_idxs = set()
		fw_pb_types = set()
		fw_frame_idxs = set()
		fw_edit_ops = set()
		fw_hmm_states = set()

		# try no pruning

		for plu_bottom_index in range(pb_lower_limit,pb_upper_limit+1):
			fw_pb_idxs |= {plu_bottom_index}
			# print("forward ** plu_bottom_index = "+str(plu_bottom_index))
			# print("len(forward_probs) = "+str(len(forward_probs)))
			pt_lower_limit = max(-1,plu_bottom_index-max_slip)
			# pt_upper_limit = min(len(plu_tops), plu_bottom_index+max_slip)
			# pt_lower_limit = -1
			pt_upper_limit = len(plu_tops)
			for plu_top_index in range(pt_lower_limit,pt_upper_limit):
				fw_pt_idxs |= {plu_top_index}
				for plu_bottom_type in range(self.n_units):
					fw_pb_types |= {plu_bottom_type}
					frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
					frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
					# frame_lower_limit = -1
					# frame_upper_limit = n_frames-1
					for frame_index in range(frame_lower_limit,frame_upper_limit):
						fw_frame_idxs |= {frame_index}
						for edit_op in Ops.CODES:
							fw_edit_ops |= {edit_op}
							if edit_op == Ops.IT:
								hmm_range = [self.n_states-1]
							elif edit_op == Ops.IB or edit_op == Ops.SUB:
								hmm_range = [0]
							else:
								hmm_range = range(self.n_states)
							for hmm_state in hmm_range:
								fw_hmm_states |= {hmm_state}
								curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)
								

								# curr_forward =  forward_arr[np.array(curr_state)]
								# print("curr shape")
								# print(curr_forward.shape)
								# if curr_forward > neg_inf:
								# 	nexts = self.next_states((curr_state, curr_forward), plu_tops, state_llh, max_slip, frames_per_top, log05, logging)
								# 	for (next_state, prob) in nexts:
								# 		if forward_arr[list(next_state)] > neg_inf:
								# 			forward_arr[list(next_state)] = np.logaddexp(forward_arr[list(next_state)], prob)
								# 		else:
								# 			forward_arr[list(next_state)] = prob

								if curr_state in forward_probs and forward_probs[curr_state] > neg_inf:

									logging = (np.random.ranf() < 0)

									if False:
										rfile_lines.append(str(frame_index)+','+str(forward_probs[curr_state])+','+str(edit_op)+'\n')

									nexts = self.next_states((curr_state, forward_probs[curr_state]), plu_tops, state_llh, max_slip, frames_per_top, log05, logging)
									for (next_state, prob) in nexts:
										if next_state in forward_probs:
											forward_probs[next_state] = np.logaddexp(forward_probs[next_state], prob)
										else:
											forward_probs[next_state] = prob

		# print("Number of frames:",n_frames)

		if logging:
			with open('rfile', 'a') as f:
				f.write(''.join(rfile_lines))


		# Calculate backward probabilities, and also sum forwards+backwards probabilities in the same pass
		backward_probs = {}
		fw_bw_probs = {}

		excessive_fw = []
		excessive_bw = []
		excessive_fw_bw = []

		bw_pb_idxs = set()
		bw_pt_idxs = set()
		bw_pb_types = set()
		bw_frame_idxs = set()
		bw_edit_ops = set()
		bw_hmm_states = set()

		# Insert ending items
		end_items = self.generate_end_items(plu_tops, state_llh, max_slip)
		for (item, prob) in end_items:
			backward_probs[item] = prob

		# Initialize data structures for expected counts of HMM states for each frame, and also expected counts of edit operations
		log_op_counts = [np.full((1 + 2*self.n_units), float("-inf")) for _ in range(len(self.op_latent_posteriors))]
		log_state_counts = np.full((self.n_units*self.n_states, n_frames), float("-inf"))

		tot_ibs = 0
		tot_subs = 0
		tot_its = 0

		pb_upper_limit = len(plu_tops)-1 + max_slip
		pb_lower_limit = -1
		for plu_bottom_index in range(pb_upper_limit, pb_lower_limit-1, -1):
			bw_pb_idxs |= {plu_bottom_index}
			# print("backward ** plu_bottom_index = "+str(plu_bottom_index))
			# print("len(backward_probs) = "+str(len(backward_probs)))
			pt_lower_limit = max(-1,plu_bottom_index-max_slip)
			# pt_upper_limit = min(len(plu_tops)-1, plu_bottom_index+max_slip)
			# pt_lower_limit = -1
			# used to be pt_upper_limit = min(len(plu_tops)-1, plu_bottom_index+max_slip)
			# -1 was off-by-1 error i think
			pt_upper_limit = len(plu_tops)-1

			for plu_top_index in range(pt_upper_limit, pt_lower_limit-1, -1):
				bw_pt_idxs |= {plu_top_index}
				for plu_bottom_type in range(self.n_units):
					bw_pb_types |= {plu_bottom_type}
					frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
					frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
					# frame_lower_limit = -1
					# frame_upper_limit = n_frames
					# added -1 to fix off-by-1 error
					for frame_index in range(frame_upper_limit,frame_lower_limit-1,-1):
						bw_frame_idxs |= {frame_index}
						for edit_op in Ops.CODES:
							bw_edit_ops |= {edit_op}
							if edit_op == Ops.IT:
								hmm_range = [self.n_states-1]
							elif edit_op == Ops.IB or edit_op == Ops.SUB:
								hmm_range = [0]
							else:
								hmm_range = range(self.n_states-1,-1,-1)

							for hmm_state in hmm_range:
								bw_hmm_states |= {hmm_state}
								curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)
								
								if curr_state in backward_probs:

									# Update probabilities for previous states
									prevs = self.prev_states((curr_state, backward_probs[curr_state]), plu_tops, state_llh, max_slip, frames_per_top, log05)
									for (prev_state, prob) in prevs:
										if prev_state in backward_probs:
											backward_probs[prev_state] = np.logaddexp(backward_probs[prev_state], prob)
										else:
											backward_probs[prev_state] = prob




									# Sum forwards & backwards probabilities
									if curr_state in forward_probs:
										fw_bw_prob = forward_probs[curr_state] + backward_probs[curr_state]
										# fw_bw_prob = forward_probs.get(curr_state, float('-inf')) + \
										# 			backward_probs.get(curr_state, float('-inf'))
										# assert(fw_bw_prob != np.nan)
										fw_bw_probs[curr_state] = fw_bw_prob

										if curr_state[2] == 0 and plu_tops[curr_state[-1]] == 1 and edit_op == Ops.SUB:

											excessive_fw.append(forward_probs[curr_state])
											excessive_bw.append(backward_probs[curr_state])
											excessive_fw_bw.append(fw_bw_probs[curr_state])

										# now we need to normalize 
										# need to get p(string)
										# can do this by summing over all other item states in p(*item, string)
										if edit_op == Ops.IB:
											# Increase the count in the insert-bottom section of the distribution (add 1 to plu_bottom_type)
											log_op_counts[plu_tops[plu_top_index+1]][plu_bottom_type+1] = np.logaddexp(log_op_counts[plu_tops[plu_top_index+1]][plu_bottom_type+1],
																														fw_bw_prob)
											tot_ibs += 1
										elif edit_op == Ops.IT:
											# Increase the count in the insert-top section of the distribution (the first slot)
											log_op_counts[plu_tops[plu_top_index]][0] = np.logaddexp(log_op_counts[plu_tops[plu_top_index]][0],
																								fw_bw_prob)
											tot_its += 1
										elif edit_op == Ops.SUB:
											# Increase the count in the substitute section of the distribution (add 1+self.n_units to plu_bottom_type)
											log_op_counts[plu_tops[plu_top_index]][plu_bottom_type+self.n_units+1] = np.logaddexp(log_op_counts[plu_tops[plu_top_index]][plu_bottom_type+self.n_units+1],
																																fw_bw_prob)
											tot_subs += 1

										# Update per-frame HMM expected counts
										log_state_counts[plu_bottom_type*self.n_states+hmm_state, frame_index] = np.logaddexp(log_state_counts[plu_bottom_type*self.n_states+hmm_state, frame_index], 
																													fw_bw_prob)

		
		# print("asserting log_op_counts don't start as nan")
		for i in range(len(log_op_counts)):
			# see if these counts are nan 
			try:
				assert(not np.any(np.isnan(log_op_counts[i])))
			except AssertionError:
				print("isnan assertion excepted in file", file, log_op_counts)
				assert(not np.any(np.isnan(log_op_counts[i])))

			# if np.any(log_op_counts[i]>-100):
			# 	print("over -100 for:: ")
			# 	print("=============================================")
			# 	print("\n\n")
			# 	print(plu_tops)
			# 	print(list(state_llh))
			# 	print(log_op_counts)


		# print("passed outside assertion")
		with open('fw_probs', "wb") as f1:
			pickle.dump(forward_probs, f1)
		with open('bw_probs', "wb") as f1:
			pickle.dump(backward_probs, f1)
		with open('fw_bw_probs', "wb") as f1:
			pickle.dump(fw_bw_probs, f1)

		start_items = [ x[0] for x in self.generate_start_items(plu_tops, state_llh) ]
		start_item_bw_probs = [ backward_probs.get(x, float('-inf')) for x in start_items]
		start_item_total = logsumexp(start_item_bw_probs)

		end_items = [ x[0] for x in self.generate_end_items(plu_tops, state_llh, max_slip) ]
		end_item_fw_probs = [ forward_probs.get(x, float('-inf')) for x in end_items]
		end_item_total = logsumexp(end_item_fw_probs)


		# print("Frames: "+str(n_frames)+"   Pb types: "+str(self.n_units)+"   Pt types: "+str(max(plu_tops)+1)+"   Pt indices: "+str(len(plu_tops)))

		# with open('logfile', 'a') as f:
		# 	f.write('=====================\n')
		# 	f.write('forward_backward_noisy_channel\n')
		# 	f.write('log_op_counts: '+str(log_op_counts)+'\n')
		# 	f.write('log_state_counts: '+str(log_state_counts)+'\n')
		# print("end item total: ", end_item_total)
		excessive_fw = np.array(excessive_fw) - end_item_total
		excessive_bw = np.array(excessive_bw)- end_item_total
		excessive_fw_bw = np.array(excessive_fw_bw)- end_item_total

		# print("counting too many forward:")
		# print(excessive_fw)
		# print("backward:")
		# print(excessive_bw)
		# print("both:")
		# print(excessive_fw_bw)
		print("log op counts are: ", type(log_op_counts))
		print("end item total is", type(end_item_total))
		log_op_counts_normalized = log_op_counts - end_item_total

		# if np.any(np.isnan(log_op_counts_normalized)):
		# 	print("end items:")
		# 	print(sorted(end_items))
		# 	print("forward:")
		# 	fw_ends = np.array([forward_probs.get(x, float('-inf')) for x in sorted(end_items)])
		# 	print(fw_ends)
		# 	print("backward:")
		# 	bw_ends = np.array([backward_probs.get(x,float('-inf')) for x in sorted(end_items)])
		# 	print(bw_ends)
		# 	print("sum:")
		# 	print(fw_ends+bw_ends)
		# 	print("fw/bw:")
		# 	print([fw_bw_probs.get(x,float('-inf')) for x in sorted(end_items)])

		# print("checking indices:")
		assert(len(fw_pb_idxs ^ bw_pb_idxs) == 0)
		try:
			assert(len(fw_pt_idxs ^ bw_pt_idxs) == 0)
		except AssertionError:
			# # print(fw_pt_idxs ^ bw_pt_idxs)
			# # print(sorted(fw_pt_idxs))
			# # print(sorted(bw_pt_idxs))
			# sys.exit()
			pass
		assert(len(fw_pb_types ^ bw_pb_types) == 0)
		try:
			assert(len(fw_frame_idxs ^ bw_frame_idxs) == 0)
		except AssertionError:
			# print(fw_frame_idxs ^ bw_frame_idxs)
			# print(sorted(fw_frame_idxs))
			# print(sorted(bw_frame_idxs))
			# sys.exit()
			pass
		assert(len(fw_edit_ops ^ bw_edit_ops) == 0)
		assert(len(fw_hmm_states ^ bw_hmm_states) == 0)
		
		fw_states = set(forward_probs.keys()) 
		fw_ends = fw_states & set(end_items)
		bw_states = set(backward_probs.keys()) 
		bw_ends = bw_states & set(end_items)

		fw_starts = fw_states & set(start_items)
		bw_starts = bw_states  & set(start_items)

		total_states = fw_states | bw_states
		# print("total states: {}".format(len(total_states)))
		# print("forward states: {}".format(len(fw_states)))
		# print("backward states: {}".format(len(bw_states)))
		# print("in forward but not backwards:")
		# print(len(fw_states - bw_states))
		# print("in backward but not forwards:")
		# print(len(bw_states - fw_states))

		# print("ends in forward: ", len(fw_ends))
		# print("ends in backward: ", len(bw_ends))
		# print("starts in forward: ", len(fw_starts))
		# print("starts in backward: ", len(bw_starts))
		# print("log op counts:")
		# print(log_op_counts)
		# print("log_op_counts_normalized:")
		# print(log_op_counts_normalized)


		assert(not np.any(np.isnan(log_state_counts)))
		assert(not np.any(np.isnan(log_op_counts_normalized)))

		return log_op_counts_normalized, log_state_counts

	def generate_start_items(self, plu_tops, state_llh):

		log_prob_all_ops = [self.op_latent_posteriors[plu_tops[0]].grad_log_partition[0]]
		log_prob_all_ops.extend(self.op_latent_posteriors[plu_tops[0]].grad_log_partition[1:self.n_units+1])
		log_prob_all_ops.extend(self.op_latent_posteriors[plu_tops[0]].grad_log_partition[self.n_units+1:])

		prob_all_ops = np.exp(log_prob_all_ops)
		prob_all_ops/= np.sum(prob_all_ops)

		prob_it = prob_all_ops[0]
		prob_ib = prob_all_ops[1:self.n_units+1]
		prob_sub = prob_all_ops[self.n_units + 1: ]

		log_prob_it = np.log(prob_it)
		log_prob_ib = np.log(prob_ib)
		log_prob_sub = np.log(prob_sub)
		# log_prob_it = self.op_latent_posteriors[plu_tops[0]].grad_log_partition[0]
		# log_prob_ib = self.op_latent_posteriors[plu_tops[0]].grad_log_partition[1:self.n_units+1]
		# log_prob_sub = self.op_latent_posteriors[plu_tops[0]].grad_log_partition[self.n_units+1:]

		try:
			assert(np.abs(logsumexp([logsumexp(log_prob_it), logsumexp(log_prob_ib), logsumexp(log_prob_sub)]) - 0.0) < 0.000001)
		except AssertionError:
			print(np.abs(logsumexp([logsumexp(log_prob_it), logsumexp(log_prob_ib), logsumexp(log_prob_sub)]) - 0.0))
			sys.exit()
		# ((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p)


		# Insert-bottom start items
		ib_start_items = [((0, 0, pb, 0, Ops.IB, -1), (state_llh[0,(pb*self.n_states)] + log_prob_ib[pb])) for pb in range(self.n_units)]
		
		# with open('logfile', 'a') as f:
		# 	for pb in range(self.n_units):
		# 		f.write('state_llh[0,('+str(pb)+'*self.n_states)]: '+str(state_llh[0,(pb*self.n_states)])+'\n')
		# 		f.write('log_prob_ib['+str(pb)+']: '+str(log_prob_ib[pb])+'\n')
		# 		f.write('state_llh[0,('+str(pb)+'*self.n_states)] + log_prob_ib['+str(pb)+']: : '+str(state_llh[0,(pb*self.n_states)]+log_prob_ib[pb])+'\n')

		# Insert-top start item
		it_start_item = ((-1,self.n_states-1,0,-1,Ops.IT,0), (log_prob_it))

		# Substitute start items
		sub_start_items = [((0, 0, pb, 0, Ops.SUB, 0),(state_llh[0,(pb*self.n_states)] + log_prob_sub[pb])) for pb in range(self.n_units)]

		items = ib_start_items
		items.append(it_start_item)
		items.extend(sub_start_items)

		# with open('logfile', 'a') as f:
		# 	f.write('=====================\n')
		# 	f.write('start_items: '+str(items)+'\n')

		return items

	def generate_end_items(self, plu_tops, state_llh, max_slip):

		n_frames = state_llh.shape[0]

		min_final_pb_index = max(len(plu_tops)-1-max_slip,0)
		max_final_pb_index = len(plu_tops)-1+max_slip
		#print("min="+str(min_final_pb_index)+", max="+str(max_final_pb_index))

		items = [(n_frames-1, self.n_states-1, pb, pb_index, op, len(plu_tops)-1) for pb in range(self.n_units) for pb_index in range(min_final_pb_index, max_final_pb_index+1) for op in [Ops.IT, Ops.NONE]]
		
		log_prob = 0. # log of 1 is 0

		return [(item, log_prob) for item in items]


	# Takes as input a tuple representing the current state
	# in the form ((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p)
	# and returns a list containing tuples of the form (next_state, log_prob)
	def next_states(self, current_state, plu_tops, state_llh, max_slip, frames_per_top, log05, logging):

		((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p) = current_state

		# with open('logfile', 'a') as f:
		# 	f.write('========\n')
		# 	f.write('len(self.renorms) '+str(len(self.renorms))+'\n')
		# 	f.write('len(self.renorms[0]) '+str(len(self.renorms[0]))+'\n')
		# 	f.write('plu_bottom_index '+str(plu_bottom_index)+'\n')
		# 	f.write('plu_bottom_type '+str(plu_bottom_type)+'\n')
		# 	f.write('plu_top_index '+str(plu_top_index)+'\n')
		# 	f.write('len(plu_tops) '+str(len(plu_tops))+'\n')
		# 	f.write('plu_tops[plu_top_index] '+str(plu_tops[plu_top_index])+'\n')
			# if plu_top_index < len(plu_tops)-1:
			# 	f.write('plu_tops[plu_top_index+1] '+str(plu_tops[plu_top_index+1])+'\n')
			# else:
			# 	f.write('n/a')

		if plu_top_index == len(plu_tops)-1:
			log_prob_all_ops = None

		else:
			if plu_bottom_index == -1:
				log_prob_all_ops = self.op_latent_posteriors[plu_tops[plu_top_index+1]].grad_log_partition
			else:
				log_prob_all_ops = self.renorms[plu_tops[plu_top_index+1]][plu_bottom_type]

			prob_all_ops = np.exp(log_prob_all_ops)
			prob_all_ops/= np.sum(prob_all_ops)

			prob_it = prob_all_ops[0]
			prob_ib = prob_all_ops[1:self.n_units+1]
			prob_sub = prob_all_ops[self.n_units + 1: ]

			log_prob_it = np.log(prob_it)
			log_prob_ib = np.log(prob_ib)
			log_prob_sub = np.log(prob_sub)
		# print('log_prob_ib')
		# print(log_prob_ib)
		# print('log_prob_sub')
		# print(log_prob_sub)

		n_frames = state_llh.shape[0]

		next_states = []

		# Insert bottom op (for all possible bottom PLUs)
		if (hmm_state == self.n_states-1) and (plu_top_index < len(plu_tops)-1) and (frame_index < n_frames-1) and (plu_bottom_index-plu_top_index < max_slip):
			next_states.extend([((frame_index+1, 0, pb, plu_bottom_index+1, Ops.IB, plu_top_index), \
				(p + state_llh[frame_index+1,(pb*self.n_states)] + log_prob_ib[pb] + log05)) for pb in range(self.n_units)])
			# if logging:
			# 	with open('logfile', 'a') as f:
			# 		f.write('=====================\n')
			# 		f.write('a sample next probability calculation for insert bottom:\n')
			# 		f.write('item: '+str(next_states[1])+'\n')
			# 		f.write('p + state_llh[frame_index+1,(pb*self.n_states)] + log_prob_ib[pb] + log05 =\n')
			# 		f.write(str(p)+' + '+str(state_llh[frame_index+1,(1*self.n_states)])+' + '+str(log_prob_ib[1])+' + '+str(log05)+'=')
			# 		f.write(str(next_states[1][1])+'\n')

		# Insert top op
		if (hmm_state == self.n_states-1) and (plu_top_index < len(plu_tops)-1) and (plu_top_index-plu_bottom_index < max_slip):
			next_states.append(((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, Ops.IT, plu_top_index+1), \
				(p + log_prob_it)))

			# if logging:
			# 	with open('logfile', 'a') as f:
			# 		f.write('=====================\n')
			# 		f.write('a sample next probability calculation for insert top:\n')
			# 		f.write('item: '+str(next_states[-1])+'\n')
			# 		f.write('p + log_prob_it =\n')
			# 		f.write(str(p)+' + '+str(log_prob_it)+'=')
			# 		f.write(str(next_states[-1][1])+'\n')


		# Substitute op (for all possible bottom PLUs)
		if (hmm_state == self.n_states-1) and (frame_index < n_frames-1) and (plu_top_index < len(plu_tops)-1):
			next_states.extend([((frame_index+1, 0, pb, plu_bottom_index+1, Ops.SUB, plu_top_index+1), \
				(p + state_llh[frame_index+1,(pb*self.n_states)] + log_prob_sub[pb] + log05)) for pb in range(self.n_units)])


			# if plu_bottom_index+1 == 0 and plu_tops[plu_top_index+1] == 1: 
			# 	print("components:")
			# 	print()
			# 	print(["pb: {}, state_llh: {}, log_prob_sub[pb]: {}".format(pb, state_llh[frame_index+1,(pb*self.n_states)], log_prob_sub[pb]) for pb in range(self.n_units)])
			# if logging:
			# 	with open('logfile', 'a') as f:
			# 		f.write('=====================\n')
			# 		f.write('a sample next probability calculation for substitute:\n')
			# 		f.write('p + state_llh[frame_index+1,(pb*self.n_states)] + log_prob_ib[pb] + log05 =\n')
			# 		f.write(str(p)+' + '+str(state_llh[frame_index+1,((self.n_units-1)*self.n_states)])+' + '+str(log_prob_sub[-1])+' + '+str(log05)+'=')
			# 		f.write(str(next_states[-1][1])+'\n')



		# HMM-state-internal transition
		if (edit_op != Ops.IT) and (frame_index < n_frames-1):
			next_states.append(((frame_index+1, hmm_state, plu_bottom_type, plu_bottom_index, Ops.NONE, plu_top_index), \
				(p + state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state)] + log05)))

			# if logging:
			# 	with open('logfile', 'a') as f:
			# 		f.write('=====================\n')
			# 		f.write('a sample next probability calculation for HMM-state-internal transition:\n')
			# 		f.write('p + state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state)] + log05 =\n')
			# 		f.write(str(p)+' + '+str(state_llh[frame_index+1,(plu_bottom_type*self.n_states)+hmm_state])+' + '+str(log05)+'=')
			# 		f.write(str(next_states[-1][1])+'\n')


		# PLU-internal HMM state transition
		if (edit_op != Ops.IT) and (hmm_state < self.n_states-1) and (frame_index < n_frames-1):
			next_states.append(((frame_index+1, hmm_state+1, plu_bottom_type, plu_bottom_index, Ops.NONE, plu_top_index), \
				(p + state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state+1)] + log05)))

			# if logging:
			# 	with open('logfile', 'a') as f:
			# 		f.write('=====================\n')
			# 		f.write('a sample next probability calculation for HMM-state transition:\n')
			# 		f.write('p + state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state+1)] + log05 =\n')
			# 		f.write(str(p)+' + '+str(state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state+1)])+' + '+str(log05)+'=')
			# 		f.write(str(next_states[-1][1])+'\n')

		#print([x for x in next_states if x[1] == float('-inf')])
		# if frame_index==283:
		# 	print("**283** Current state:", current_state, "  next_states", next_states)
		return next_states

	# Takes as input a tuple representing the current state
	# in the form ((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p)
	# and returns a list containing tuples of the form (next_state, log_prob)

	def prev_states(self, current_state, plu_tops, state_llh, max_slip, frames_per_top, log05):

		((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p) = current_state

		n_frames = state_llh.shape[0]

		# log_prob_all_ops = self.renorms[plu_tops[plu_top_index]][plu_bottom_type-1]

		# log_prob_it = log_prob_all_ops[0]
		# log_prob_it = self.renorms[plu_tops[plu_top_index]][pb][0]
		# log_prob_ib = log_prob_all_ops[1:self.n_units+1]
		# log_prob_ib = self.renorms[plu_tops[plu_top_index]][pb][plu_bottom_type+1]
		# log_prob_sub = log_prob_all_ops[self.n_units+1:]
		# log_prob_sub = self.renorms[plu_tops[plu_top_index]][pb][plu_bottom_type+self.n_units+1]

		prev_states = []

		# I am not very confident that these probabilities are correct

		# Reverse of insert bottom op (for all possible previous bottom PLUs)
		if (hmm_state == 0) and (edit_op == Ops.IB) and (plu_bottom_index > -1) and (frame_index > -1) and (plu_top_index-plu_bottom_index < max_slip):
			prev_states.extend([((frame_index-1, self.n_states-1, pb, plu_bottom_index-1, op, plu_top_index), \
				(p + state_llh[frame_index,(pb*self.n_states)] + self.renorms[plu_tops[plu_top_index]][pb][plu_bottom_type+1] + log05) ) for pb in range(self.n_units) for op in [Ops.IT, Ops.NONE]])

		# Reverse of insert top op (for all possible previous edit ops)
		if (hmm_state == self.n_states-1) and (edit_op == Ops.IT) and (plu_top_index > -1) and (plu_bottom_index-plu_top_index < max_slip):
			prev_states.extend([((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, op, plu_top_index-1), \
				(p + self.renorms[plu_tops[plu_top_index]][plu_bottom_type][0])) for op in [Ops.IT, Ops.NONE]])

		# Reverse of substitute op (for all possible previous bottom PLUs and edit ops)
		if (hmm_state == 0) and (edit_op == Ops.SUB) and (plu_bottom_index > -1) and \
				(plu_top_index > -1) and (frame_index > -1):
			prev_states.extend([((frame_index-1, self.n_states-1, pb, plu_bottom_index-1, op, plu_top_index-1), \
				(p + state_llh[frame_index,(pb*self.n_states)] + self.renorms[plu_tops[plu_top_index]][pb][plu_bottom_type+self.n_units+1] + log05)) for pb in range(self.n_units) for op in [Ops.IT, Ops.NONE]])

		# Reverse of HMM-state-internal transition
		if (edit_op == Ops.NONE) and (frame_index > 0):
			prev_states.append(((frame_index-1, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), \
				(p + state_llh[frame_index,(plu_bottom_type*self.n_states+hmm_state)] + log05)))
			if (hmm_state==0):
				prev_states.extend([((frame_index-1, hmm_state, plu_bottom_type, plu_bottom_index, op, plu_top_index), \
					(p + state_llh[frame_index,(plu_bottom_type*self.n_states+hmm_state)] + log05)) for op in [Ops.IB, Ops.SUB]])

		# Reverse of PLU-internal HMM state transition
		if (edit_op == Ops.NONE) and (hmm_state > 0) and (frame_index > 0):
			prev_states.append(((frame_index-1, hmm_state-1, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), \
				(p + state_llh[frame_index,(plu_bottom_type*self.n_states+hmm_state)] + log05)))
			if (hmm_state == 1):
				prev_states.extend([((frame_index-1, hmm_state-1, plu_bottom_type, plu_bottom_index, op, plu_top_index), \
					(p + state_llh[frame_index,(plu_bottom_type*self.n_states+hmm_state)] + log05)) for op in [Ops.IB, Ops.SUB]])

		return prev_states



