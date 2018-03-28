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
from bisect import bisect
from itertools import groupby
from scipy.special import logsumexp

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
		op_latent_priors.append(Dirichlet(np.ones(n_units)))
		op_latent_posteriors = [Dirichlet(np.ones(num_ops)) for _ in range(n_top_units)]
		op_latent_posteriors.append(Dirichlet(np.ones(n_units)))

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
		self.n_top_units = len(op_latent_priors)-1
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
		self.renorms = [[None for _ in range(self.n_units)] for __ in range(self.n_top_units+1)]
		for i in range(self.n_top_units+1):
			dist = self.op_latent_posteriors[i].grad_log_partition
			for j in range(self.n_units):
				renorm_dist = np.copy(dist)
				if i == self.n_top_units:
					# Special case for the last Dirichlet
					renorm_dist = np.concatenate((np.array([float('-inf')]), renorm_dist, np.full(self.n_units,float('-inf'))))

				# Set insert-bottom log-probability of previous phone to -inf (exp(-inf)=0)
				renorm_dist[j+1] = float('-inf')
				# Set substitute log-probability of previous phone to -inf (exp(-inf)=0)
				renorm_dist[j+1+self.n_units] = float('-inf')

				renorm_dist = renorm_dist - logsumexp(renorm_dist)
				self.renorms[i][j] = renorm_dist

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


	def units_stats(self, c_llhs, log_alphas, log_betas):
		log_units_stats = np.zeros(self.n_units)
		norm = logsumexp(log_alphas[-1] + log_betas[-1])
		log_A = np.log(self.trans_mat.toarray())

		for n_unit in range(self.n_units):
			index1 = n_unit * self.n_states + 1
			index2 = index1 + 1
			log_prob_trans = log_A[index1, index2]
			log_q_zn1_zn2 = log_alphas[:-1, index1] + c_llhs[1:, index2] + \
				log_prob_trans + log_betas[1:, index2]
			log_q_zn1_zn2 -= norm
			log_units_stats[n_unit] = logsumexp(log_q_zn1_zn2)

		return np.exp(log_units_stats)



	def decode(self, data, plu_tops, state_path=False, phone_intervals=False, edit_ops=False):
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
									nexts = self.next_states((curr_state, forward_probs[curr_state][0]), plu_tops, state_llh, max_slip, frames_per_top, log05)

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
		best_item = None
		best_prob = float("-inf")
		for (item, _) in end_items:
			value = forward_probs.get(item, (float("-inf"), None))
			if value[0] > best_prob:
				best_prob = forward_probs[item][0]
				best_item = item

		print("BEST END ITEM:")
		print(best_item)

		(prob, back_pointer_tuple) = forward_probs[best_item]

		back_path = [best_item]

		# Do backtrace
		while back_pointer_tuple is not None:
			back_path.append(back_pointer_tuple[0])
			(prob, back_pointer_tuple) = forward_probs[back_path[-1]]
			print("APPENDING")
			print(prob, " ", back_pointer_tuple)

		back_path.reverse()

		# Get PLU list
		plu_path = [state[2] for state in back_path]



		if edit_ops:
			edit_path = [ (Ops.to_string(state[4]), plu_tops[state[5]], state[2]) for state in back_path if state[4] != Ops.NONE ]
			(frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

		if phone_intervals:
			groups = groupby(plu_path)

			interval_path = []
			begin_index = 0
			for group in groups:
				end_index = begin_index + sum(1 for item in group[1])
				interval_path.append((group[0], begin_index, end_index))
				begin_index = end_index

		if phone_intervals and edit_ops:
			return (interval_path, edit_path)
		if phone_intervals:
			return interval_path
		if edit_ops:
			return edit_path
		else:
			return plu_path


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

		for comp in self.components:
			retval += comp.posterior.kl_div(comp.prior)

		for idx, post in enumerate(self.state_posteriors):
			retval += post.kl_div(self.state_priors[idx])

		return retval

	# @profile(immediate=True)
	def get_posteriors(self, s_stats, top_seq, accumulate=False,filename=None):
		import time
		state_llh, c_given_s_resps = self._get_state_llh(s_stats)

		# print("state_llh at {}:".format(time.clock()))
		# print(state_llh.shape)
		# print(logsumexp(state_llh, axis=1))
		# print(state_llh)

		# The workhorse
		
		log_op_counts, log_state_counts = self.forward_backward_noisy_channel(top_seq, state_llh, filename)
		# get last column of log state counts, which is the normalizer (i.e. P(X_{1:T})), prob of whole frame sequence
		log_prob_observations = logsumexp(log_state_counts[:,-1])
		# normalize everything else by prob_obs to get expected count
		print("got posteriors for file {} ".format(filename.strip()))
		print("log_op_counts: ", log_op_counts)
		print("log state counts: ", log_state_counts)
		# Compute the posteriors

		# # Do we actually need to normalize
		# op_counts_normalized = []
		# for i in range(len(log_op_counts)):
		# 	print("log_op_counts shape ", log_op_counts[i].shape)
		# 	op_counts_normalized.append(np.exp(log_op_counts[i] - log_prob_observations))
		# quick hacky fix to not re-re-normalize op counts
		op_counts_normalized = [log_op_counts[i] for i in range(len(log_op_counts))]
		# Normalize by frame
		state_norm = logsumexp(log_state_counts, axis=0)
		state_counts_perframe_normalized = np.exp((log_state_counts - state_norm))
		# ^ this is equivalent to state_resps in the normal phone loop

		if accumulate:
			tot_resps = state_counts_perframe_normalized[:, np.newaxis, :] * c_given_s_resps
			gauss_resps = tot_resps.reshape(-1, tot_resps.shape[-1])
			
			# We don't need units_stats because we have the op counts

			state_stats = tot_resps.sum(axis=2)
			gauss_stats = gauss_resps.dot(s_stats)
			efdstats = [state_stats, gauss_stats]
			# efdstats.extend(op_counts_normalized)
			efdstats.extend(log_op_counts)
			acc_stats = EFDStats(efdstats)

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
			if i == len(op_counts)-1:
				op_count_i = op_count_i[1:self.n_units+1]
			op_grad = self.op_latent_priors[i].natural_params + op_count_i - self.op_latent_posteriors[i].natural_params
			self.op_latent_posteriors[i].natural_params += lrate * op_grad

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


	def forward_backward_sample(self, plu_tops, state_llh, n_samples):
		print("RUNNING FORWARD BACKWARD \n\n\n STARTS HERE")
		import sys
		n_frames = state_llh.shape[0]
		max_slip = math.ceil(len(plu_tops)*self.max_slip_factor)
		max_plu_bottom_index = len(plu_tops) + max_slip

		n_frames = state_llh.shape[0]

		frames_per_top = n_frames/len(plu_tops)

		# Calculate forward probabilities
		forward_probs = {}
		# Insert starting items
		start_items = self.generate_start_items(plu_tops, state_llh)
		for (item, prob) in start_items:
			forward_probs[item] = prob


		fw_ibs = 0

		pb_upper_limit = len(plu_tops)-1 + max_slip
		pb_lower_limit = -1
		for plu_bottom_index in range(pb_lower_limit,pb_upper_limit+1):
			print("forward ** plu_bottom_index = "+str(plu_bottom_index))
			print("len(forward_probs) = "+str(len(forward_probs)))
			pt_lower_limit = max(-1,plu_bottom_index-max_slip)
			pt_upper_limit = min(len(plu_tops), plu_bottom_index+max_slip)
			for plu_top_index in range(pt_lower_limit,pt_upper_limit):
				for plu_bottom_type in range(self.n_units):
					frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
					frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
					for frame_index in range(frame_lower_limit,frame_upper_limit):
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
									nexts = self.next_states((curr_state, forward_probs[curr_state]), plu_tops, state_llh, max_slip, frames_per_top)
									
									for next_state_and_prob in nexts:
										(next_state, prob) = next_state_and_prob
										if next_state in forward_probs:
											forward_probs[next_state] = np.logaddexp(forward_probs[next_state], prob)
										else:
											forward_probs[next_state] = prob
			
		# sample backwards probs
		# when we renormalize the chart, what are we summing across? 
		# need to pick a transition
		# so we should sum across all variables that aren't included in the transition? 
		forward_sum  = logsumexp(list(forward_probs.values()))


		log_state_counts = np.full((self.n_units*self.n_states, n_frames), float("-inf"))
		log_ib_counts = np.full(self.n_units, float("-inf"))
		log_it_counts = np.full(self.n_top_units, float("-inf"))
		log_sub_counts = np.full(self.n_units*self.n_top_units, float("-inf"))

		tot_ibs = 0
		tot_subs = 0
		tot_its = 0

		pb_upper_limit = len(plu_tops)-1 + max_slip
		pb_lower_limit = -1

		
		for s in range(n_samples):
			# choose an start item 
			items, item_probs = zip(*start_items)
			item_probs -= forward_sum
			item_idx = np.random.choice(np.arange(0,len(items)), p = np.exp(item_probs))

			item = items[item_idx]
			nexts = self.next_states((item, forward_probs[item]), plu_tops, state_llh, max_slip, frames_per_top)

			while len(nexts) > 0:
				next_states, next_probs = zip(*nexts)
				
				if len(next_probs) > 1:
					# choose one previous state
					next_probs -= forward_sum
					state_idx = np.random.choice(np.arange(0,len(next_states)), p=np.exp(next_probs))
				else:
					state_idx = 0

				new_state, prob = nexts[state_idx]
				frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index = new_state
				# Update edit operation expected counts
				if edit_op == Ops.IB:
					log_ib_counts[plu_bottom_type] = np.logaddexp(log_ib_counts[plu_bottom_type],0.0)
					tot_ibs += 1
				elif edit_op == Ops.IT:
					log_it_counts[plu_tops[plu_top_index]] = np.logaddexp(log_it_counts[plu_tops[plu_top_index]],0.0)
					tot_its += 1
				elif edit_op == Ops.SUB:
					log_sub_counts[plu_bottom_type*self.n_top_units+plu_tops[plu_top_index]] =\
					 		np.logaddexp(log_sub_counts[plu_bottom_type*self.n_top_units+plu_tops[plu_top_index]],0.0)
					tot_subs += 1
				# Update per-frame HMM expected counts
				log_state_counts[plu_bottom_type*self.n_states+hmm_state, frame_index] =\
				 			np.logaddexp(log_state_counts[plu_bottom_type*self.n_states+hmm_state, frame_index], 0.0)
				
				nexts = self.next_states((next_state, forward_probs[next_state]), plu_tops, state_llh, max_slip, frames_per_top)

		# print(log_ib_counts, log_it_counts, log_sub_counts, log_state_counts)
		print("totals: ", tot_ibs, tot_subs, tot_its)
		lg_n_samples = np.log(n_samples)
		print(log_ib_counts - lg_n_samples, log_it_counts- lg_n_samples, log_sub_counts - lg_n_samples, log_state_counts - lg_n_samples)
		return log_ib_counts - lg_n_samples, log_it_counts- lg_n_samples, log_sub_counts - lg_n_samples, log_state_counts - lg_n_samples


	def forward_backward_noisy_channel(self, plu_tops, state_llh, file):

		n_frames = state_llh.shape[0]
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

		# pb_upper_limit = len(plu_tops)-1 + max_slip
		# make same as decode
		pb_upper_limit = len(plu_tops) + max_slip
		pb_lower_limit = -1
		for plu_bottom_index in range(pb_lower_limit,pb_upper_limit+1):
			# print("forward ** plu_bottom_index = "+str(plu_bottom_index))
			# print("len(forward_probs) = "+str(len(forward_probs)))
			pt_lower_limit = max(-1,plu_bottom_index-max_slip)
			pt_upper_limit = min(len(plu_tops), plu_bottom_index+max_slip)
			for plu_top_index in range(pt_lower_limit,pt_upper_limit):
				for plu_bottom_type in range(self.n_units):
					frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
					frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
					for frame_index in range(frame_lower_limit,frame_upper_limit):
						for edit_op in Ops.CODES:
							if edit_op == Ops.IT:
								hmm_range = [self.n_states-1]
							elif edit_op == Ops.IB or edit_op == Ops.SUB:
								hmm_range = [0]
							else:
								hmm_range = range(self.n_states)
							for hmm_state in hmm_range:
								curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)
								if curr_state in forward_probs and forward_probs[curr_state] > neg_inf:
									nexts = self.next_states((curr_state, forward_probs[curr_state]), plu_tops, state_llh, max_slip, frames_per_top, log05)
									for (next_state, prob) in nexts:
										# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op.value, plu_top_index)
										if next_state in forward_probs:
											#if plu_bottom_type == next_state[2] and (next_state[4] in [Ops.IB, Ops.SUB]):
												#print("adding transition to same state")
												#print("yikes")
												#print(next_state, prob)
											forward_probs[next_state] = np.logaddexp(forward_probs[next_state], prob)
										else:
											forward_probs[next_state] = prob

		# Calculate backward probabilities, and also sum forwards+backwards probabilities in the same pass
		backward_probs = {}
		fw_bw_probs = {}
		any_item = False
		# Insert ending items
		end_items = self.generate_end_items(plu_tops, state_llh, max_slip)
		for (item, prob) in end_items:
			backward_probs[item] = prob
			try:
				fw_bw_probs[item] = forward_probs[item] + prob
				any_item = True
			except KeyError:
				pass

		if not any_item:
			print("THERE ARE NO END ITEMS FOR FILE ", file.strip())



		fp = list(forward_probs.keys())[0]
		ei = list(end_items[0][0])
		print("a forward item is : ", fp, " with length " , len(fp))
		print("an end item is : ", ei, "with length ", len(ei))

		assert(len(set(forward_probs.keys()) & set([x[0] for x in end_items]))	> 0)

		# Initialize data structures for expected counts of HMM states for each frame, and also expected counts of edit operations
		log_op_counts = [np.full((1 + 2*self.n_units), float("-inf")) for _ in range(len(self.op_latent_posteriors))]
		log_state_counts = np.full((self.n_units*self.n_states, n_frames), float("-inf"))

		tot_ibs = 0
		tot_subs = 0
		tot_its = 0

		log_fb_norm = np.log(0)

		pb_upper_limit = len(plu_tops)-1 + max_slip
		pb_lower_limit = -1
		for plu_bottom_index in range(pb_upper_limit, pb_lower_limit-1, -1):
			# print("backward ** plu_bottom_index = "+str(plu_bottom_index))
			# print("len(backward_probs) = "+str(len(backward_probs)))
			pt_lower_limit = max(-1,plu_bottom_index-max_slip)
			pt_upper_limit = min(len(plu_tops)-1, plu_bottom_index+max_slip)

			for plu_top_index in range(pt_upper_limit, pt_lower_limit-1, -1):
				for plu_bottom_type in range(self.n_units):
					frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
					frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
					for frame_index in range(frame_upper_limit,frame_lower_limit,-1):
						for edit_op in Ops.CODES:
							if edit_op == Ops.IT:
								hmm_range = [self.n_states-1]
							elif edit_op == Ops.IB or edit_op == Ops.SUB:
								hmm_range = [0]
							else:
								hmm_range = range(self.n_states-1,-1,-1)

							for hmm_state in hmm_range:
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
										# assert(fw_bw_prob != np.nan)
										fw_bw_probs[curr_state] = fw_bw_prob
										# now we need to normalize 
										# need to get p(string)
										# can do this by summing over all other item states in p(*item, string)
										# log_fb_norm[frame_index] = np.logaddexp(log_fb_norm[frame_index], fw_bw_prob)
										if edit_op == Ops.IB:
											# Increase the count in the insert-bottom section of the distribution (add 1 to plu_bottom_type)
											if plu_top_index == len(plu_tops)-1:
												# Special case for last PLU
												log_op_counts[-1][plu_bottom_type] = np.logaddexp(log_op_counts[-1][plu_bottom_type],
																								fw_bw_prob) 
																						
											else:
												log_op_counts[plu_tops[plu_top_index+1]][plu_bottom_type+1] = np.logaddexp(log_op_counts[plu_tops[plu_top_index]][plu_bottom_type+1],
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


		print("asserting log_op_counts don't start as nan")
		for i in range(len(log_op_counts)):
			# see if these counts are nan 
			assert(not np.any(np.isnan(log_op_counts[i])))

		print("accumulate normalizer")
		for (item, _) in end_items:
			if item in fw_bw_probs.keys():
				if np.any(np.isnan(fw_bw_probs[item])):
					print("got a nan at ", item)
					assert(2==1)
				log_fb_norm = np.logaddexp(log_fb_norm, fw_bw_probs[item])
		# make sure something got added
		print('asserting log_fb_norm is greater than zero')
		print("log fb norm is : ", log_fb_norm, " in file ", file.strip())
		try:
			assert(log_fb_norm > np.log(0))
		except AssertionError:
			print("increasing fb norm")
			log_fb_norm = 2e-76

		for i in range(len(log_op_counts)):
			for j in range(len(log_op_counts[i])):
				log_op_counts[i][j] -= log_fb_norm


			# now do a pass to normalize and add

		assert(not np.any(np.isnan(log_fb_norm)))
		print("passed outside assertion")

		

		print("Frames: "+str(n_frames)+"   Pb types: "+str(self.n_units)+"   Pt types: "+str(max(plu_tops)+1)+"   Pt indices: "+str(len(plu_tops)))

		return log_op_counts, log_state_counts

	def generate_start_items(self, plu_tops, state_llh):

		log_prob_it = self.op_latent_posteriors[plu_tops[0]].grad_log_partition[0]
		log_prob_ib = self.op_latent_posteriors[-1].grad_log_partition
		log_prob_sub = self.op_latent_posteriors[plu_tops[0]].grad_log_partition[self.n_units:]


		# Insert-bottom start items
		ib_start_items = [((0, 0, pb, 0, Ops.IB, -1), (state_llh[0,(pb*self.n_states)] + log_prob_ib[pb])) for pb in range(self.n_units)]
		
		# Insert-top start item
		it_start_item = ((-1,self.n_states-1,0,-1,Ops.IT,0), (log_prob_it))

		# Substitute start items
		sub_start_items = [((0, 0, pb, 0, Ops.SUB, 0),(state_llh[0,(pb*self.n_states)] + log_prob_sub[pb])) for pb in range(self.n_units)]

		items = ib_start_items
		items.append(it_start_item)
		items.extend(sub_start_items)
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
	def next_states(self, current_state, plu_tops, state_llh, max_slip, frames_per_top, log05):
		# print('current_state')
		# print(current_state)
		((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p) = current_state

		if plu_bottom_index == -1:
			log_prob_all_ops = self.op_latent_posteriors[plu_tops[plu_top_index+1]].grad_log_partition
		elif plu_top_index == len(plu_tops)-1:
			log_prob_all_ops = self.renorms[-1][plu_bottom_type]
		else:
			log_prob_all_ops = self.renorms[plu_tops[plu_top_index+1]][plu_bottom_type]

		log_prob_it = log_prob_all_ops[0]
		log_prob_ib = log_prob_all_ops[1:self.n_units+1]
		log_prob_sub = log_prob_all_ops[self.n_units+1:]

		# print('log_prob_ib')
		# print(log_prob_ib)
		# print('log_prob_sub')
		# print(log_prob_sub)

		n_frames = state_llh.shape[0]

		next_states = []

		# Insert bottom op (for all possible bottom PLUs)
		if (hmm_state == self.n_states-1) and (frame_index < n_frames-1) and (plu_bottom_index-plu_top_index < max_slip):
			next_states.extend([((frame_index+1, 0, pb, plu_bottom_index+1, Ops.IB, plu_top_index), \
				(p + state_llh[frame_index+1,(pb*self.n_states)] + log_prob_ib[pb] + log05)) for pb in range(self.n_units)])

		# Insert top op
		if (hmm_state == self.n_states-1) and (plu_top_index < len(plu_tops)-1) and (plu_top_index-plu_bottom_index < max_slip):
			next_states.append(((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, Ops.IT, plu_top_index+1), \
				(p + log_prob_it)))

		# Substitute op (for all possible bottom PLUs)
		if (hmm_state == self.n_states-1) and (frame_index < n_frames-1) and (plu_top_index < len(plu_tops)-1):
			next_states.extend([((frame_index+1, 0, pb, plu_bottom_index+1, Ops.SUB, plu_top_index+1), \
				(p + state_llh[frame_index+1,(pb*self.n_states)] + log_prob_sub[pb] + log05)) for pb in range(self.n_units)])

		# HMM-state-internal transition
		if (edit_op != Ops.IT) and (frame_index < n_frames-1):
			next_states.append(((frame_index+1, hmm_state, plu_bottom_type, plu_bottom_index, Ops.NONE, plu_top_index), \
				(p + state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state)] + log05)))

		# PLU-internal HMM state transition
		if (edit_op != Ops.IT) and (hmm_state < self.n_states-1) and (frame_index < n_frames-1):
			next_states.append(((frame_index+1, hmm_state+1, plu_bottom_type, plu_bottom_index, Ops.NONE, plu_top_index), \
				(p + state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state+1)] + log05)))

		#print([x for x in next_states if x[1] == float('-inf')])
		return next_states

	# Takes as input a tuple representing the current state
	# in the form ((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p)
	# and returns a list containing tuples of the form (next_state, log_prob)

	def prev_states(self, current_state, plu_tops, state_llh, max_slip, frames_per_top, log05):

		((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p) = current_state

		n_frames = state_llh.shape[0]

		log_prob_all_ops = self.renorms[plu_tops[plu_top_index-1]][plu_bottom_type-1]

		log_prob_it = log_prob_all_ops[0]
		log_prob_ib = log_prob_all_ops[1:self.n_units+1]
		log_prob_sub = log_prob_all_ops[self.n_units+1:]

		prev_states = []

		# I am not very confident that these probabilities are correct

		# Reverse of insert bottom op (for all possible previous bottom PLUs)
		if (hmm_state == 0) and (edit_op == Ops.IB) and (plu_bottom_index > 0) and (frame_index > 0) and (plu_top_index-plu_bottom_index < max_slip):
			prev_states.extend([((frame_index-1, self.n_states-1, pb, plu_bottom_index-1, op, plu_top_index), \
				(p + state_llh[frame_index,(pb*self.n_states)] + log_prob_ib[plu_bottom_type] + log05) ) for pb in range(self.n_units) for op in [Ops.IT, Ops.NONE]])

		# Reverse of insert top op (for all possible previous edit ops)
		if (hmm_state == self.n_states-1) and (edit_op == Ops.IT) and (plu_top_index > 0) and (plu_bottom_index-plu_top_index < max_slip):
			prev_states.extend([((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, op, plu_top_index-1), \
				(p + log_prob_it)) for op in [Ops.IT, Ops.NONE]])

		# Reverse of substitute op (for all possible previous bottom PLUs and edit ops)
		if (hmm_state == 0) and (edit_op == Ops.SUB) and (plu_bottom_index > 0) and \
				(plu_top_index > 0) and (frame_index > 0):
			prev_states.extend([((frame_index-1, self.n_states-1, pb, plu_bottom_index-1, op, plu_top_index-1), \
				(p + state_llh[frame_index,(pb*self.n_states)] + log_prob_sub[plu_bottom_type] + log05)) for pb in range(self.n_units) for op in [Ops.IT, Ops.NONE]])

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



