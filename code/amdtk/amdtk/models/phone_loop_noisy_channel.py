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
from numba import jit
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

from enum import Enum

class Ops(Enum):
	IB = 0
	IT = 1
	SUB = 2
	NONE = 3

class PhoneLoopNoisyChannel(DiscreteLatentModel):
	"""Bayesian Phone Loop model with noisy channel addition.

	Bayesian Phone Loop with a sequence of top-level PLUs
	and a Dirichlet prior over edit operations.

	"""

	def create(n_units, n_states, n_comp_per_state, n_top_units, mean, var):
		"""Create and initialize a Bayesian Phone Loope Model.

		Parameters
		----------
		n_units : int
			Number of acoustic units i.e. phones.
		n_states : int
			Number of states for each acoustic unit.
		n_comp_per_state : int
			Number of compent per emission.
		top_symbols : list
			List of top-level symbols to match to
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

		# Initialize the Dirichlets over operation type
		op_type_latent_prior = Dirichlet(np.ones(3))
		op_type_latent_posterior = Dirichlet(np.ones(3))

		# Initialize the Dirichlets over PLUs within operation type
		it_latent_prior = Dirichlet(np.ones(n_top_units))
		it_latent_posterior = Dirichlet(np.ones(n_top_units))

		ib_latent_prior = Dirichlet(np.ones(n_units))
		ib_latent_posterior = Dirichlet(np.ones(n_units))

		sub_latent_prior = Dirichlet(np.ones(n_units*n_top_units))
		sub_latent_posterior = Dirichlet(np.ones(n_units*n_top_units))

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

		return PhoneLoopNoisyChannel(op_type_latent_prior, op_type_latent_posterior,
						 it_latent_prior, it_latent_posterior,
						 ib_latent_prior, ib_latent_posterior,
						 sub_latent_prior, sub_latent_posterior,
						 state_priors, state_posteriors, components)

	def __init__(self, op_type_latent_prior, op_type_latent_posterior,
						 it_latent_prior, it_latent_posterior,
						 ib_latent_prior, ib_latent_posterior,
						 sub_latent_prior, sub_latent_posterior,
						 state_priors, state_posteriors, components):

		# Ok I think we're not gonna do this here, we're just gonna implement
		# our own version of the DiscreteLatentModel functions (because we have
		# many more distributions over which we are doing inference)
		#DiscreteLatentModel.__init__(self, latent_prior, latent_posterior, components)
		self._components = components
		self._exp_np_matrix = self._get_components_params_matrix()

		self.n_units = len(ib_latent_prior.natural_params)
		self.n_states = len(state_priors) // self.n_units
		self.n_comp_per_states = len(state_priors[0].natural_params)
		self.n_top_units = len(it_latent_prior.natural_params)

		self.max_slip_factor = 0.05

		self.op_type_latent_prior = op_type_latent_prior
		self.op_type_latent_posterior = op_type_latent_posterior
		self.ib_latent_prior = ib_latent_prior
		self.ib_latent_posterior = ib_latent_posterior
		self.it_latent_prior = it_latent_prior
		self.it_latent_posterior = it_latent_posterior
		self.sub_latent_prior = sub_latent_prior
		self.sub_latent_posterior = sub_latent_posterior

		self.state_priors = state_priors
		self.state_posteriors = state_posteriors

		self.post_update()


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


	def decode(self, data, plu_tops, state_path=False, phone_intervals=False, context=False):
		s_stats = self.get_sufficient_stats(data)

		state_llh, c_given_s_resps = self._get_state_llh(s_stats)

		n_frames = state_llh.shape[0]
		max_slip = math.ceil(len(plu_tops)*self.max_slip_factor)
		max_plu_bottom_index = len(plu_tops) + max_slip

		log_prob_ops = self.op_type_latent_posterior.grad_log_partition
		log_prob_ib = self.ib_latent_posterior.grad_log_partition
		log_prob_it = self.it_latent_posterior.grad_log_partition
		log_prob_sub = self.sub_latent_posterior.grad_log_partition
		n_frames = state_llh.shape[0]

		frames_per_top = n_frames/len(plu_tops)

		# Calculate forward probabilities WITH BACKPOINTERS
		forward_probs = {}
		# Insert starting items
		for (item, prob) in self.generate_start_items(plu_tops, state_llh):
			forward_probs[item] = (prob, None)

		for plu_bottom_index in range(-1,max_plu_bottom_index):
			print("forward ** plu_bottom_index = "+str(plu_bottom_index))
			print("len(forward_probs) = "+str(len(forward_probs)))
			pt_lower_limit = max(-1,plu_bottom_index-max_slip)
			pt_upper_limit = min(len(plu_tops), plu_bottom_index+max_slip)
			for plu_top_index in range(pt_lower_limit, pt_upper_limit):
				for plu_bottom_type in range(self.n_units):
					for edit_op in Ops:
						for hmm_state in range(self.n_states):
							frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
							frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
							for frame_index in range(frame_lower_limit, frame_upper_limit):
								curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op.value, plu_top_index)
								if curr_state in forward_probs:
									nexts = self.next_states((curr_state, forward_probs[curr_state][0]), plu_tops, state_llh, max_slip, frames_per_top)
									
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

		(prob, back_pointer_tuple) = forward_probs[best_item]

		back_path = [best_item]

		# Do backtrace
		while back_pointer_tuple is not None:
			back_path.append(back_pointer_tuple[0])
			(prob, back_pointer_tuple) = forward_probs[back_path[-1]]

		# Get PLU list
		back_path.reverse()

		path = [state[2] for state in back_path]
		if context:
			context = [(state[2], state[-1]) for state in back_path]

		if phone_intervals:
			groups = groupby(path)
			if context:
				context_groups = groupby(context, key=lambda x: x[0])
			interval_path = []
			begin_index = 0
			for group in groups:
				end_index = begin_index + sum(1 for item in group[1])
				interval_path.append((group[0], begin_index, end_index))
				begin_index = end_index
			if context:
				return interval_path, context_groups
			return interval_path

		else:
			return path


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
		retval += self.op_type_latent_posterior.kl_div(self.op_type_latent_prior)
		retval += self.ib_latent_posterior.kl_div(self.ib_latent_prior)
		retval += self.it_latent_posterior.kl_div(self.it_latent_prior)
		retval += self.sub_latent_posterior.kl_div(self.sub_latent_prior)

		for comp in self.components:
			retval += comp.posterior.kl_div(comp.prior)

		for idx, post in enumerate(self.state_posteriors):
			retval += post.kl_div(self.state_priors[idx])

		return retval

	#@profile(immediate=True)
	
	def get_posteriors(self, s_stats, top_seq, accumulate=False):
		state_llh, c_given_s_resps = self._get_state_llh(s_stats)

		# The workhorse
		log_ib_counts, log_it_counts, log_sub_counts, log_state_counts = self.forward_backward_noisy_channel(top_seq, state_llh)

		# Compute the posteriors

		log_edit_op_counts = np.array([logsumexp(log_ib_counts), logsumexp(log_it_counts), logsumexp(log_sub_counts)])
		edit_op_counts_normalized = np.exp(log_edit_op_counts - logsumexp(log_edit_op_counts))

		ib_counts_normalized = np.exp(log_ib_counts - logsumexp(log_ib_counts))
		it_counts_normalized = np.exp(log_it_counts - logsumexp(log_it_counts))
		sub_counts_normalized = np.exp(log_sub_counts - logsumexp(log_sub_counts))

		# Normalize by frame
		state_norm = logsumexp(log_state_counts, axis=0)
		state_counts_perframe_normalized = np.exp(log_state_counts - state_norm)
		# ^ this is equivalent to state_resps in the normal phone loop

		if accumulate:
			tot_resps = state_counts_perframe_normalized[:, np.newaxis, :] * c_given_s_resps
			gauss_resps = tot_resps.reshape(-1, tot_resps.shape[-1])
			
			# We don't need units_stats because we have the ib, it, sub counts

			state_stats = tot_resps.sum(axis=2)
			gauss_stats = gauss_resps.dot(s_stats)
			acc_stats = EFDStats([edit_op_counts_normalized, ib_counts_normalized, it_counts_normalized, sub_counts_normalized, 
				state_stats, gauss_stats])

			return state_counts_perframe_normalized, state_norm[-1], acc_stats

		return state_counts_perframe_normalized, state_norm[-1]

	def natural_grad_update(self, acc_stats, lrate):
		"""Natural gradient update."""
		edit_op_counts = acc_stats[0]
		ib_counts = acc_stats[1]
		it_counts = acc_stats[2]
		sub_counts = acc_stats[3]
		state_stats = acc_stats[4]
		gauss_stats = acc_stats[5]


		# Update edit op type counts
		op_grad = self.op_type_latent_prior.natural_params + edit_op_counts - self.op_type_latent_posterior.natural_params
		self.op_type_latent_posterior.natural_params += lrate * op_grad

		# Update edit op counts
		ib_grad = self.ib_latent_prior.natural_params + ib_counts - self.ib_latent_posterior.natural_params
		self.ib_latent_posterior.natural_params += lrate * ib_grad
		it_grad = self.it_latent_prior.natural_params + it_counts - self.it_latent_posterior.natural_params
		self.it_latent_posterior.natural_params += lrate * it_grad
		sub_grad = self.sub_latent_prior.natural_params + sub_counts - self.sub_latent_posterior.natural_params
		self.sub_latent_posterior.natural_params += lrate * sub_grad

		print("self.op_type_latent_posterior.natural_params")
		print(self.op_type_latent_posterior.natural_params)
		print("self.ib_latent_posterior.natural_params")
		print(self.ib_latent_posterior.natural_params)
		print("self.it_latent_posterior.natural_params")
		print(self.it_latent_posterior.natural_params)
		print("self.sub_latent_posterior.natural_params")
		print(self.sub_latent_posterior.natural_params)


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
	@jit
	def forward_backward_noisy_channel(self, plu_tops, state_llh):

		n_frames = state_llh.shape[0]
		max_slip = math.ceil(len(plu_tops)*self.max_slip_factor)
		max_plu_bottom_index = len(plu_tops) + max_slip

		log_prob_ops = self.op_type_latent_posterior.grad_log_partition
		log_prob_ib = self.ib_latent_posterior.grad_log_partition
		log_prob_it = self.it_latent_posterior.grad_log_partition
		log_prob_sub = self.sub_latent_posterior.grad_log_partition
		n_frames = state_llh.shape[0]

		frames_per_top = n_frames/len(plu_tops)

		# Calculate forward probabilities
		forward_probs = {}
		# Insert starting items
		for (item, prob) in self.generate_start_items(plu_tops, state_llh):
			forward_probs[item] = prob

		for plu_bottom_index in range(-1,max_plu_bottom_index):
			print("forward ** plu_bottom_index = "+str(plu_bottom_index))
			print("len(forward_probs) = "+str(len(forward_probs)))
			pt_lower_limit = max(-1,plu_bottom_index-max_slip)
			pt_upper_limit = min(len(plu_tops), plu_bottom_index+max_slip)
			for plu_top_index in range(pt_lower_limit,pt_upper_limit):
				for plu_bottom_type in range(self.n_units):
					for edit_op in Ops:
						for hmm_state in range(self.n_states):
							frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
							frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
							for frame_index in range(frame_lower_limit,frame_upper_limit):
								curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op.value, plu_top_index)
								if curr_state in forward_probs:
									nexts = self.next_states((curr_state, forward_probs[curr_state]), plu_tops, state_llh, max_slip, frames_per_top)
									for next_state_and_prob in nexts:
										(next_state, prob) = next_state_and_prob
										forward_probs[next_state] = np.logaddexp(forward_probs.get(next_state, 1), prob)

		# Calculate backward probabilities
		backward_probs = {}
		# Insert ending items
		for (item, prob) in self.generate_end_items(plu_tops, state_llh, max_plu_bottom_index):
			backward_probs[item] = prob
		for plu_bottom_index in range(max_plu_bottom_index-1, 0, -1):
			print("backward ** plu_bottom_index = "+str(plu_bottom_index))
			print("len(backward_probs) = "+str(len(backward_probs)))
			for plu_top_index in range(min(len(plu_tops)-1, plu_bottom_index+max_slip), max(-1, plu_bottom_index-max_slip), -1):
				for plu_bottom_type in range(self.n_units):
					for edit_op in Ops:
						for hmm_state in range(self.n_states-1,-1,-1):
							frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
							frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
							for frame_index in range(frame_upper_limit,frame_lower_limit,-1):
								curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op.value, plu_top_index)
								if curr_state in backward_probs:
									prevs = self.prev_states((curr_state, backward_probs[curr_state]), plu_tops, n_frames, state_llh, max_slip, frames_per_top)
									for prev_state_and_prob in prevs:
										(prev_state, prob) = prev_state_and_prob
										backward_probs[prev_state] = np.logaddexp(backward_probs.get(prev_state, 1), prob)

		# Sum over forwards & backwards probabilities to get the expected counts
		# We need expected counts of HMM states for each frame, and also expected counts of edit operations
		log_state_counts = np.ones((self.n_units*self.n_states, n_frames))
		log_ib_counts = np.ones(self.n_units)
		log_it_counts = np.ones(self.n_top_units)
		log_sub_counts = np.ones(self.n_units*self.n_top_units)
		for plu_bottom_index in range(-1,max_plu_bottom_index):
			print("together ** plu_bottom_index = "+str(plu_bottom_index))
			for plu_top_index in range(max(-1,plu_bottom_index-max_slip),min(len(plu_tops), plu_bottom_index+max_slip)):
				for plu_bottom_type in range(self.n_units):
					for edit_op in Ops:
						for hmm_state in range(self.n_states):
							frame_lower_limit = max(-1, math.floor((plu_bottom_index-max_slip)*frames_per_top))
							frame_upper_limit = min(n_frames, math.ceil((plu_bottom_index+max_slip)*frames_per_top))
							for frame_index in range(n_frames):
								curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op.value, plu_top_index)

								if curr_state in forward_probs and curr_state in backward_probs:
									fw_bw_prob = forward_probs[curr_state] + backward_probs[curr_state]

									# Update edit operation expected counts
									if edit_op.value == Ops.IB.value:
										log_ib_counts[plu_bottom_type] = np.logaddexp(log_ib_counts[plu_bottom_type],fw_bw_prob)
									elif edit_op.value == Ops.IT.value:
										log_it_counts[plu_tops[plu_top_index]] = np.logaddexp(log_it_counts[plu_tops[plu_top_index]],fw_bw_prob)
									elif edit_op.value == Ops.SUB.value:
										log_sub_counts[plu_bottom_type*self.n_top_units+plu_tops[plu_top_index]] = np.logaddexp(log_sub_counts[plu_bottom_type*self.n_top_units+plu_tops[plu_top_index]],fw_bw_prob)

									# Update per-frame HMM expected counts
									log_state_counts[plu_bottom_type*self.n_states+hmm_state, frame_index] = np.logaddexp(log_state_counts[plu_bottom_type*self.n_states+hmm_state, frame_index], fw_bw_prob)

		print("len(forward_probs)")
		print(len(forward_probs))

		print("len(backward_probs)")
		print(len(backward_probs))

		print("log_ib_counts")
		print(log_ib_counts)

		print("log_it_counts")
		print(log_it_counts)

		print("log_sub_counts")
		print(log_sub_counts)

		print("log_state_counts")
		print(log_state_counts)

		return log_ib_counts, log_it_counts, log_sub_counts, log_state_counts

	def generate_start_items(self, plu_tops, state_llh):

		log_prob_ops = self.op_type_latent_posterior.grad_log_partition
		log_prob_ib = self.ib_latent_posterior.grad_log_partition
		log_prob_it = self.it_latent_posterior.grad_log_partition
		log_prob_sub = self.sub_latent_posterior.grad_log_partition


		# Insert-bottom start items
		ib_start_items = [((0, 0, pb, 0, Ops.IB.value, -1), (state_llh[0,(pb*self.n_states)] + log_prob_ops[Ops.IB.value] + log_prob_ib[pb])) for pb in range(self.n_units)]
		
		# Insert-top start item
		it_start_item = ((-1,self.n_states-1,-1,Ops.IT.value,0), (log_prob_ops[Ops.IT.value] + log_prob_it[plu_tops[0]]))

		# Substitute start items
		sub_start_items = [((0, 0, pb, 0, Ops.IB.value, 0),(state_llh[0,(pb*self.n_states)] + log_prob_ops[Ops.SUB.value] + log_prob_sub[pb*self.n_top_units+plu_tops[0]])) for pb in range(self.n_units)]

		items = ib_start_items
		items.append(it_start_item)
		items.extend(sub_start_items)
		return items

	def generate_end_items(self, plu_tops, state_llh, max_plu_bottom_index):

		n_frames = state_llh.shape[0]

		items = [(n_frames-1, self.n_states-1, pb, pb_index, op.value, len(plu_tops)-1) for pb in range(self.n_units) for pb_index in range(1,max_plu_bottom_index) for op in Ops]
		
		prob = math.log(1./len(items))

		return [(item, prob) for item in items]


	# Takes as input a tuple representing the current state
	# in the form ((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p)
	# and returns a list containing tuples of the form (next_state, log_prob)
	def next_states(self, current_state, plu_tops, state_llh, max_slip, frames_per_top):
		((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p) = current_state

		log_prob_ops = self.op_type_latent_posterior.grad_log_partition
		log_prob_ib = self.ib_latent_posterior.grad_log_partition
		log_prob_it = self.it_latent_posterior.grad_log_partition
		log_prob_sub = self.sub_latent_posterior.grad_log_partition
		n_frames = state_llh.shape[0]

		next_states = []

		# Insert bottom op (for all possible bottom PLUs)
		if (hmm_state == self.n_states-1) and (frame_index < n_frames-1) and (plu_bottom_index-plu_top_index < max_slip):
			next_states.extend([((frame_index+1, 0, pb, plu_bottom_index+1, Ops.IB.value, plu_top_index), \
				(p + state_llh[frame_index+1,(pb*self.n_states)] + log_prob_ops[Ops.IB.value] + log_prob_ib[pb] + math.log(0.5))) for pb in range(self.n_units)])

		# Insert top op
		if (hmm_state == self.n_states-1) and (plu_top_index < len(plu_tops)-1) and (plu_top_index-plu_bottom_index < max_slip):
			next_states.append(((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, Ops.IT.value, plu_top_index+1), \
				(p + log_prob_ops[Ops.IT.value] + log_prob_it[plu_tops[plu_top_index+1]])))

		# Substitute op (for all possible bottom PLUs)
		if (hmm_state == self.n_states-1) and (frame_index < n_frames-1) and (plu_top_index < len(plu_tops)-1):
			next_states.extend([((frame_index+1, 0, pb, plu_bottom_index+1, Ops.SUB.value, plu_top_index+1), \
				(p + state_llh[frame_index+1,(pb*self.n_states)] + log_prob_ops[Ops.SUB.value] + log_prob_sub[pb*self.n_top_units+plu_tops[plu_top_index+1]] + math.log(0.5))) for pb in range(self.n_units)])

		# HMM-state-internal transition
		if (edit_op != Ops.IT.value) and (frame_index < n_frames-1):
			next_states.append(((frame_index+1, hmm_state, plu_bottom_type, plu_bottom_index, Ops.NONE.value, plu_top_index), \
				(p + state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state)] + math.log(0.5))))

		# PLU-internal HMM state transition
		if (edit_op != Ops.IT.value) and (hmm_state < self.n_states-1) and (frame_index < n_frames-1):
			next_states.append(((frame_index+1, hmm_state+1, plu_bottom_type, plu_bottom_index, Ops.NONE.value, plu_top_index), \
				(p + state_llh[frame_index+1,(plu_bottom_type*self.n_states+hmm_state)] + math.log(0.5))))

		return next_states

	# Takes as input a tuple representing the current state
	# in the form ((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p)
	# and returns a list containing tuples of the form (next_state, log_prob)

	def prev_states(self, current_state, plu_tops, n_frames, state_llh, max_slip, frames_per_top):
		((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p) = current_state

		log_prob_ops = self.op_type_latent_posterior.grad_log_partition
		log_prob_ib = self.ib_latent_posterior.grad_log_partition
		log_prob_it = self.it_latent_posterior.grad_log_partition
		log_prob_sub = self.sub_latent_posterior.grad_log_partition

		prev_states = []

		# NOTE: THE PROBABILITY UPDATES ARE *NOT* RIGHT IN THIS FUNCTION --
		# I AM NOT SURE HOW TO UPDATE THE PROBABILITIES CORRECTLY FOR BACKWARDS ALGORITHM

		# Reverse of insert bottom op (for all possible previous bottom PLUs and edit ops)
		if (hmm_state == 0) and (edit_op == Ops.IB.value) and (plu_bottom_index > 0) and (frame_index > 0) and (plu_top_index-plu_bottom_index < max_slip):
			prev_states.extend([((frame_index-1, self.n_states-1, pb, plu_bottom_index-1, op.value, plu_top_index), \
				(p + state_llh[frame_index-1,(pb*self.n_states)] + log_prob_ops[Ops.IB.value] + log_prob_ib[pb] + math.log(0.5)) ) for pb in range(self.n_units) for op in [Ops.IT, Ops.NONE]])

		# Reverse of insert top op (for all possible previous edit ops)
		if (hmm_state == self.n_states-1) and (edit_op == Ops.IT.value) and (plu_top_index > 0) and (plu_bottom_index-plu_top_index < max_slip):
			prev_states.extend([((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, op.value, plu_top_index-1), \
				(p + log_prob_ops[Ops.IT.value] + log_prob_it[plu_tops[plu_top_index-1]])) for op in [Ops.IT, Ops.NONE]])

		# Reverse of substitute op (for all possible previous bottom PLUs and edit ops)
		if (hmm_state == 0) and (edit_op == Ops.SUB.value) and (plu_bottom_index > 0) and \
				(plu_top_index > 0) and (frame_index > 0):
			prev_states.extend([((frame_index-1, self.n_states-1, pb, plu_bottom_index-1, op.value, plu_top_index-1), \
				(p + state_llh[frame_index-1,(pb*self.n_states)] + log_prob_ops[Ops.SUB.value] + log_prob_sub[pb*self.n_top_units+plu_tops[plu_top_index-1]] + math.log(0.5))) for pb in range(self.n_units) for op in Ops])

		# Reverse of HMM-state-internal transition
		if (edit_op != Ops.IT.value) and (frame_index > 0):
			prev_states.append(((frame_index-1, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), \
				(p + state_llh[frame_index-1,(plu_bottom_type*self.n_states+hmm_state)] + math.log(0.5))))

		# Reverse of PLU-internal HMM state transition
		if (edit_op != Ops.IT.value) and (hmm_state > 0) and (frame_index > 0):
			prev_states.append(((frame_index-1, hmm_state-1, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), \
				(p + state_llh[frame_index-1,(plu_bottom_type*self.n_states+hmm_state)] + math.log(0.5))))

		return prev_states



