"""
Code for running VB optimization on the noisy channel model

"""


import abc
import time
import numpy as np
from ipyparallel.util import interactive


from enum import Enum


class Ops(Enum):
	NONE = 0
    IB = 1
    IT = 2
    SUB = 3


class NoisyChannelOptimizer(Optimizer):

	def __init__(self, dview, data_stats, args, model):

		Optimizer.__init__(self, dview, data_stats, args, model)


	# I think we can just use the superclass implementation Optimizer.run(),
	# because the real difference comes in train()
	# def run(self, data, callback):

	def train(self, fea_list, epoch, time_step):

		# Propagate the model to all the remote clients.
        self.dview.push({
            'model': self.model,
        })

        # Parallel accumulation of the sufficient statistics.
        stats_list = self.dview.map_sync(NoisyChannelOptimizer.e_step,
                                         fea_list)

        # Accumulate the results from all the jobs.
        exp_llh = stats_list[0][0]
        acc_stats = stats_list[0][1]
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
        exp_llh = 0.
        acc_stats = None
        n_frames = 0

        for arg in args_list:
            (fea_file, top_seq) = arg

            # Mean / Variance normalization.
            data = read_htk(fea_file)
            data -= data_stats['mean']
            data /= numpy.sqrt(data_stats['var'])

            # Get the accumulated sufficient statistics for the
            # given set of features.
            s_stats = model.get_sufficient_stats(data)
            posts, llh, new_acc_stats = model.get_posteriors(s_stats, top_seq,
                                                             accumulate=True)

            exp_llh += numpy.sum(llh)
            n_frames += len(data)
            if acc_stats is None:
                acc_stats = new_acc_stats
            else:
                acc_stats += new_acc_stats

        return (exp_llh, acc_stats, n_frames)



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

	    log_prob_ops = self.model.op_type_latent_posterior.grad_log_partition
	    log_prob_ib = self.model.ib_latent_posterior.grad_log_partition
	    log_prob_it = self.model.it_latent_posterior.grad_log_partition
	    log_prob_sub = self.model.sub_latent_posterior.grad_log_partition
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


