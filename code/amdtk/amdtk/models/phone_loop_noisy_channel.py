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

import math

from .hmm_utils import create_phone_loop_transition_matrix
from .hmm_utils import forward_backward
from .hmm_utils import viterbi
from .model import EFDStats, DiscreteLatentModel
from ..densities import Dirichlet, NormalGamma, NormalDiag


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

		return PhoneLoop(op_type_latent_prior, op_type_latent_posterior,
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

		self.state_priors = state_priors
		self.state_posteriors = state_posteriors

		# Will be initialized later.
		self.init_prob = None
		self.trans_mat = None
		self.init_states = None
		self.final_states = None

		self.post_update()


	def post_update(self):
		DiscreteLatentModel.post_update(self)

		# Update the states' weights.
		self.state_log_weights = np.zeros((self.n_units * self.n_states,
										   self.n_comp_per_states))
		for idx in range(self.n_units * self.n_states):
				self.state_log_weights[idx, :] = \
					self.state_posteriors[idx].grad_log_partition

		# vvv  We don't need to do any of this  vvv
		# Update the log transition matrix.
		# unigram_lm = np.exp(self.latent_posterior.grad_log_partition)
		# unigram_lm /= unigram_lm.sum()
		# self.init_prob = unigram_lm
		# self.trans_mat, self.init_states, self.final_states = \
		#     create_phone_loop_transition_matrix(self.n_units, self.n_states,
		#                                         unigram_lm)

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


	def decode(self, data, state_path=False, phone_intervals=False):
		s_stats = self.get_sufficient_stats(data)

		state_llh, c_given_s_resps = self._get_state_llh(s_stats)

		path = viterbi(
			self.init_prob,
			self.trans_mat,
			self.init_states,
			self.final_states,
			state_llh
		)

		if state_path:
			return path


		path = [bisect(self.init_states, state) for state in path]
		if phone_intervals:
			groups = groupby(path)
			path = []
			begin_index = 0
			for group in groups:
				end_index = begin_index + sum(1 for item in group[1])
				path.append((group[0], begin_index, end_index))
				begin_index = end_index

		else:
			path = [x[0] for x in groupby(path)]

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
		retval = DiscreteLatentModel.kl_div_posterior_prior(self)
		for idx, post in enumerate(self.state_posteriors):
			retval += post.kl_div(self.state_priors[idx])
		return retval

	def get_posteriors(self, s_stats, top_seq, accumulate=False):
		state_llh, c_given_s_resps = self._get_state_llh(s_stats)

		# Okay, this is where the workhorse of this algorithm we've worked out goes

		log_ib_counts, log_it_counts, log_sub_counts = forward_backward_noisy_channel(top_seq, state_llh)

		# Compute the posteriors.
		log_q_Z = (log_alphas + log_betas).T
		log_norm = logsumexp(log_q_Z, axis=0)
		state_resps = np.exp((log_q_Z - log_norm))

		if accumulate:
			tot_resps = state_resps[:, np.newaxis, :] * c_given_s_resps
			gauss_resps = tot_resps.reshape(-1, tot_resps.shape[-1])
			if self.n_states > 1 :
				units_stats = self.units_stats(state_llh, log_alphas,
											   log_betas)
			else:
				units_stats = resps.sum(axis=0)

			state_stats = tot_resps.sum(axis=2)
			gauss_stats = gauss_resps.dot(s_stats)
			acc_stats = EFDStats([units_stats, state_stats, gauss_stats])

			return state_resps, log_norm[-1], acc_stats

		return state_resps, log_norm[-1]

	def natural_grad_update(self, acc_stats, lrate):
		"""Natural gradient update."""
		# Update unigram language model.
		grad = self.latent_prior.natural_params + acc_stats[0]
		grad -= self.latent_posterior.natural_params
		self.latent_posterior.natural_params = \
			self.latent_posterior.natural_params + lrate * grad

		# Update the states' weights.
		for idx, post in enumerate(self.state_posteriors):
			grad = self.state_priors[idx].natural_params + acc_stats[1][idx]
			grad -= post.natural_params
			post.natural_params = post.natural_params + lrate * grad

		# Update Gaussian components.
		for idx, stats in enumerate(acc_stats[2]):
			comp = self.components[idx]
			grad = comp.prior.natural_params + stats
			grad -= comp.posterior.natural_params
			comp.posterior.natural_params = \
				comp.posterior.natural_params + lrate * grad

		self.post_update()


		def forward_backward_noisy_channel(plu_tops, state_llh):

			n_frames = state_llh.shape[0]
			max_plu_bottom_index = len(plu_tops)*2

			# Calculate forward probabilities
			forward_probs = {}
			for plu_bottom_index in range(max_plu_bottom_index):
				for plu_top_index in range(len(plu_tops)):
					for plu_bottom_type in range(self.model.n_units):
						for edit_op in Ops:
							for hmm_state in range(self.model.n_states):
								for frame_index in range(n_frames):
									curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)
									if curr_state in forward_probs:
										nexts = next_states((curr_state, forward_probs[state]), plu_tops, state_llh)
										for next_state_and_prob in nexts:
											(next_state, prob) = next_state_and_prob
											forward_probs[next_state] = logsumexp(np.array([forward_probs.get(next_state, 0), prob]))

			# Calculate backward probabilities
			backward_probs = {}
			for plu_bottom_index in range(max_plu_bottom_index-1, 0, -1):
				for plu_top_index in range(len(plu_tops)-1, -1, -1):
					for plu_bottom_type in range(self.model.n_units):
						for edit_op in Ops:
							for hmm_state in range(self.model.n_states-1,-1,-1):
								for frame_index in range(n_frames-1,-1,-1):
									curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)
									if curr_state in backward_probs:
										prevs = prev_states((curr_state, forward_probs[state]), plu_tops, state_llh)
										for prev_state_and_prob in prevs:
											(prev_state, prob) = next_state_and_prob
											backward_probs[prev_state] = logsumexp(np.array([backward_probs.get(prev_state, 0), prob]))

			# Sum over forwards & backwards probabilities to get the expected counts
			log_ib_counts = np.zeros(n_units)
			log_it_counts = np.zeros(n_units)
			log_sub_counts = np.zeros((n_units, n_units))
			for plu_bottom_index in range(max_plu_bottom_index):
				for plu_top_index in range(len(plu_tops)):
					for plu_bottom_type in range(self.model.n_units):
						for edit_op in Ops:
							for hmm_state in range(self.model.n_states):
								for frame_index in range(n_frames):
									curr_state = (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)
									if curr_state in forward_probs and curr_state in backward_probs:
										fw_bw_prob = forward_probs[curr_state] + backward_probs[curr_state]
										if edit_op == Ops.IB:
											log_ib_counts[plu_bottom_type] = logsumexp(np.array([log_ib_counts[plu_bottom_type],fw_bw_prob]))
										elif edit_op == Ops.IT:
											log_it_counts[plu_tops[plu_top_index]] = logsumexp(np.array([log_it_counts[plu_tops[plu_top_index]],fw_bw_prob]))
										elif edit_op == Ops.SUB:
											log_sub_counts[plu_bottom_type,plu_tops[plu_top_index]] = logsumexp(np.array(log_sub_counts[plu_bottom_type,plu_tops[plu_top_index]],fw_bw_prob))

			return log_ib_counts, log_it_counts, log_sub_counts


		# Takes as input a tuple representing the current state
		# in the form ((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p)
		# and returns a list containing tuples of the form (next_state, log_prob)
		def next_states(current_state, plu_tops, state_llh):
			((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p) = current_state

			log_prob_ops = self.op_type_latent_posterior.grad_log_partition
			log_prob_ib = self.ib_latent_posterior.grad_log_partition
			log_prob_it = self.it_latent_posterior.grad_log_partition
			log_prob_sub = self.sub_latent_posterior.grad_log_partition
			n_frames = state_llh.shape[0]

			next_states = []

			# Insert bottom op (for all possible bottom PLUs)
			if (hmm_state == self.n_states-1) and (frame_index < n_frames-1):
				next_states.extend([((frame_index+1, 0, pb, plu_bottom_index+1, Ops.IB, plu_top_index), \
					(p + state_llh[frame_index+1,(pb*n_states)] + log_prob_ops[Ops.IB] + log_prob_ib[pb] + log(0.5))) for pb in n_units])

			# Insert top op
			if (hmm_state == self.n_states-1) and (plu_top_index < len(plu_tops)-1):
				next_states.append(((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, Ops.IT, plu_top_index+1), \
					(p + log_prob_ops[Ops.IT] + log_prob_it[plu_tops[plu_top_index+1]])))

			# Substitute op (for all possible bottom PLUs)
			if (hmm_state == self.n_states-1) and (frame_index < n_frames-1) and (plu_top_index < len(plu_tops-1)):
				next_states.extend([((frame_index+1, 0, pb, plu_bottom_index+1, Ops.SUB, plu_top_index+1), \
					(p + state_llh[frame_index+1,(pb*n_states)] + log_prob_ops[Ops.SUB] + log_prob_sub[plu_tops[plu_top_index+1],pb] + log(0.5))) for pb in n_units])

			# HMM-state-internal transition
			if (edit_op != Ops.IT) and (frame_index < n_frames-1):
				next_states.append(((frame_index+1, hmm_state, plu_bottom_type, plu_bottom_index, Ops.NONE, plu_top_index), \
					(p + state_llh[frame_index+1,(pb*n_states+hmm_state)] + log(0.5))))

			# PLU-internal HMM state transition
			if (edit_op != Ops.IT) and (hmm_state < self.n_states-1) and (frame_index < n_frames-1):
				next_states.append(((frame_index+1, hmm_state+1, plu_bottom_type, plu_bottom_index, Ops.NONE, plu_top_index), \
					(p + state_llh[frame_index+1,(pb*n_states+hmm_state)] + log(0.5))))

			return next_states

		# Takes as input a tuple representing the current state
		# in the form ((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p)
		# and returns a list containing tuples of the form (next_state, log_prob)
		def prev_states(current_state, plu_tops, n_frames, state_llh):
			((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), p) = current_state

			log_prob_ops = self.op_type_latent_posterior.grad_log_partition
			log_prob_ib = self.ib_latent_posterior.grad_log_partition
			log_prob_it = self.it_latent_posterior.grad_log_partition
			log_prob_sub = self.sub_latent_posterior.grad_log_partition

			prev_states = []

			# NOTE: THE PROBABILITY UPDATES ARE *NOT* RIGHT IN THIS FUNCTION --
			# I AM NOT SURE HOW TO UPDATE THE PROBABILITIES CORRECTLY FOR BACKWARDS ALGORITHM

			# Reverse of insert bottom op (for all possible previous bottom PLUs and edit ops)
			if (hmm_state == 0) and (edit_op == Ops.IB) and (plu_bottom_index > 0) and (frame_index > 0):
				prev_states.extend([[((frame_index-1, self.n_states-1, pb, plu_bottom_index-1, op, plu_top_index), \
					(p + state_llh[frame_index+1,(pb*n_states)] + log_prob_ops[Ops.IB] + log_prob_ib[pb] + log(0.5)) ) for pb in n_units] for op in Ops])

			# Reverse of insert top op (for all possible previous edit ops)
			if (hmm_state == self.n_states-1) and (edit_op == Ops.IT) and (plu_top_index > 0):
				next_states.extend([((frame_index, hmm_state, plu_bottom_type, plu_bottom_index, Ops.IT, plu_top_index-1), \
					(p + log_prob_ops[Ops.IT] + log_prob_it[plu_tops[plu_top_index+1]])) for op in Ops])

			# Reverse of substitute op (for all possible previous bottom PLUs and edit ops)
			if (hmm_state == 0) and (edit_op == Ops.SUB) and (plu_bottom_index > 0) and \
					(plu_top_index > 0) and (frame_index > 0):
				next_states.extend([[((frame_index-1, self.n_states-1, pb, plu_bottom_index-1, ops, plu_top_index-1), \
					(p + state_llh[frame_index+1,(pb*n_states)] + log_prob_ops[Ops.SUB] + log_prob_sub[plu_tops[plu_top_index+1],pb] + log(0.5))) for pb in n_units] for op in Ops])

			# Reverse of HMM-state-internal transition
			if (edit_op != Ops.IT) and (frame_index > 0):
				next_states.append(((frame_index-1, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), \
					(p + state_llh[frame_index+1,(pb*n_states+hmm_state)] + log(0.5))))

			# Reverse of PLU-internal HMM state transition
			if (edit_op != Ops.IT) and (hmm_state > 0) and (frame_index > 0):
				next_states.append(((frame_index-1, hmm_state-1, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index), \
					(p + state_llh[frame_index+1,(pb*n_states+hmm_state)] + log(0.5))))

			return prev_states



