
import glob
import os
import numpy as np
import time as systime

from bokeh.plotting import show
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from ipyparallel import Client

import math

import sys
sys.path.insert(0, 'amdtk')

import amdtk

print("successfully completed imports")
#amdtk.utils.test_import()

def create_small_model(max_slip=0.05):

	n_units = 3
	n_states = 3
	n_comp_per_state = 2

	n_frames = 11

	n_mfccs = 4

	max_slip_factor = max_slip

	plu_tops = [3,0,1]
	num_tops = max(plu_tops)+1

	# state_llh must be n_frames x (n_units * n_states)
	state_llh = np.array([[0.01,0.01,0.01,  0.20,0.10,0.02,  0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.01,0.01,  0.01,0.20,0.20,  0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.01,0.01,  0.01,0.01,0.20,  0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.40,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.40,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.40,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.40,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.30,0.01],
						  [0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01,  0.10,0.10,0.10],
						  [0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01,  0.01,0.01,0.01]])
	state_llh = np.log(state_llh)

	# Create phone loop model
	model = amdtk.PhoneLoopNoisyChannel.create(
	    n_units=n_units,  # number of acoustic units
	    n_states=n_states,   # number of states per unit
	    n_comp_per_state=n_comp_per_state,   # number of Gaussians per emission
	    n_top_units=num_tops, # size of top PLU alphabet
	    max_slip_factor=max_slip_factor, # difference allowed between top and bottom PLU index
	    mean=np.zeros(n_mfccs), 
	    var=np.ones(n_mfccs) #,
	    #concentration=conc
	)

	return model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames

def perturb_small_model(model, n_mfccs):

	# Perturb the parameters just a little
	acc_stats = amdtk.EFDStats([
		np.random.rand(model.n_units*model.n_states,model.n_comp_per_states), # state stats
		np.random.rand(model.n_units*model.n_states*model.n_comp_per_states,n_mfccs*4), # gauss stats
		np.array([0.21, 0.12, 0.13, 0.14, 0.25, 0.16, 0.17]), # 4 distributions over bottom edit ops, one per top PLU
		np.array([0.11, 0.22, 0.13, 0.14, 0.15, 0.26, 0.17]), 
		np.array([0.11, 0.12, 0.23, 0.14, 0.15, 0.16, 0.27]), 
		np.array([0.11, 0.12, 0.13, 0.24, 0.15, 0.16, 0.17])
	])
	lrate = 0.1

	model.natural_grad_update(acc_stats, lrate)


def start_items_test(verbose=False):

	print("Running start_items_test()...")

	model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames = create_small_model()

	perturb_small_model(model, n_mfccs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	log_prob_ops = [ model.op_latent_posteriors[i].grad_log_partition for i in range(len(model.op_latent_posteriors)) ]

	# Test generate_start_items
	start_items = model.generate_start_items(plu_tops, state_llh)
	start_items_set = set(start_items)

	expected_start_items = [
		# Insert-top start item
		((-1,2,0,-1,amdtk.Ops.IT,0), log_prob_ops[3][0]),
		# Insert-bottom start items
		((0,0,0,0,amdtk.Ops.IB,-1), state_llh[0,0] + log_prob_ops[3][1]),
		((0,0,1,0,amdtk.Ops.IB,-1), state_llh[0,3] + log_prob_ops[3][2]),
		((0,0,2,0,amdtk.Ops.IB,-1), state_llh[0,6] + log_prob_ops[3][3]),
		# Substitute start items
		((0,0,0,0,amdtk.Ops.SUB,0), state_llh[0,0] + log_prob_ops[3][4]), # =0*4+3
		((0,0,1,0,amdtk.Ops.SUB,0), state_llh[0,3] + log_prob_ops[3][5]), # =1*4+3
		((0,0,2,0,amdtk.Ops.SUB,0), state_llh[0,6] + log_prob_ops[3][6])  # =2*4+3
	]
		
	expected_start_items_set = set(expected_start_items)

	if verbose:
		print("Start items:")
		expected_start_items.sort()
		start_items.sort()
		print("*********EXPECTED***********")
		for item in expected_start_items:
			print(item)
		print("*********ACTUAL*************")
		for item in start_items:
			print(item)

	assert expected_start_items_set==start_items_set
	print("TEST PASSED")


def next_states_test_1(verbose=False):

	print("Running next_states_test_1()...")

	model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames = create_small_model()

	perturb_small_model(model, n_mfccs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	log_prob_ops = [ model.op_latent_posteriors[i].grad_log_partition for i in range(len(model.op_latent_posteriors)) ]

	# Test next_states
	max_slip = math.ceil(len(plu_tops)*max_slip_factor) # = 1
	assert max_slip==1 # Just making sure
	frames_per_top = math.ceil(float(n_frames)/len(plu_tops)) # =ceil(11/3)=4
	assert frames_per_top==4

	curr_state = ((-1,2,0,-1,amdtk.Ops.IT,0), log_prob_ops[3][0])
	nexts = model.next_states(curr_state, plu_tops, state_llh, max_slip, frames_per_top, log05=math.log(0.5), logging=False)
	nexts_set = set(nexts)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)
	# PLU tops are [3,0,1]
	expected_nexts = [
		# Next via performing an insert-top
		# 	Can't! We are already ahead of the bottom by one frame
		# Next via performing an insert-bottom
		((0,0,0,0,amdtk.Ops.IB,0), curr_state[1] + state_llh[0,0] + log_prob_ops[0][1] + math.log(0.5)),
		((0,0,1,0,amdtk.Ops.IB,0), curr_state[1] + state_llh[0,3] + log_prob_ops[0][2] + math.log(0.5)),
		((0,0,2,0,amdtk.Ops.IB,0), curr_state[1] + state_llh[0,6] + log_prob_ops[0][3] + math.log(0.5)),
		# Next via performing a substitute
		((0,0,0,0,amdtk.Ops.SUB,1), curr_state[1] + state_llh[0,0] + log_prob_ops[0][4] + + math.log(0.5)), # =0*4+0
		((0,0,1,0,amdtk.Ops.SUB,1), curr_state[1] + state_llh[0,3] + log_prob_ops[0][5] + + math.log(0.5)), # =1*4+0
		((0,0,2,0,amdtk.Ops.SUB,1), curr_state[1] + state_llh[0,6] + log_prob_ops[0][6] + + math.log(0.5)), # =2*4+0
		# Can't do any HMM transitions because we are in an insert-top state,
		# which doesn't allow HMM transitions
	]
	
	expected_nexts_set = set(expected_nexts)

	if verbose:
		print("Next states after "+str(curr_state)+":")
		expected_nexts.sort()
		nexts.sort()
		print("*********EXPECTED***********")
		for item in expected_nexts:
			print(item)
		print("*********ACTUAL*************")
		for item in nexts:
			print(item)

	assert expected_nexts_set==nexts_set

	print("TEST PASSED")


def next_states_test_1a(verbose=False):

	print("Running next_states_test_1a()...")

	n_mfccs = 4

	model = amdtk.PhoneLoopNoisyChannel.create(
	    n_units=2,  # number of acoustic units
	    n_states=3,   # number of states per unit
	    n_comp_per_state=2,   # number of Gaussians per emission
	    n_top_units=2, # size of top PLU alphabet
	    max_slip_factor=1.0, # difference allowed between top and bottom PLU index
	    mean=np.zeros(n_mfccs), 
	    var=np.ones(n_mfccs) #,
	    #concentration=conc
	)

	# state_llh must be n_frames x (n_units * n_states)
	state_llh = np.array([[0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.40,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.40,0.01,  0.01,0.01,0.01],
						  [0.01,0.40,0.01,  0.01,0.01,0.01],
						  [0.01,0.40,0.01,  0.01,0.01,0.01],
						  [0.01,0.01,0.01,  0.01,0.01,0.01],
						  [0.01,0.01,0.01,  0.01,0.30,0.01],
						  [0.01,0.01,0.01,  0.10,0.10,0.10],
						  [0.01,0.01,0.01,  0.01,0.01,0.01]])
	state_llh = np.log(state_llh)


	plu_tops = [1,0,1,0]

	n_frames = 11

	#model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames = create_small_model(max_slip=1.0)

	#perturb_small_model(model, n_mfccs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	log_prob_ops = [ model.op_latent_posteriors[i].grad_log_partition for i in range(len(model.op_latent_posteriors)) ]

	# Test next_states
	max_slip_factor = 1.0
	max_slip = math.ceil(len(plu_tops)*max_slip_factor) # = 1
	assert max_slip==4 # Just making sure
	frames_per_top = math.ceil(float(n_frames)/len(plu_tops)) # =ceil(11/3)=4
	print('frames_per_top', frames_per_top)
	assert frames_per_top==3

	curr_state = ((-1,2,0,-1,amdtk.Ops.IT,1), log_prob_ops[0][0])

	print("State:", curr_state)

	nexts = model.next_states(curr_state, plu_tops, state_llh, max_slip, frames_per_top, log05=math.log(0.5), logging=False)
	nexts_set = set(nexts)

	print("Nexts:", nexts)


def next_states_test_2(verbose=False):

	print("Running next_states_test_2()...")

	model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames = create_small_model()

	perturb_small_model(model, n_mfccs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	log_prob_ops = model.op_type_latent_posterior.grad_log_partition
	log_prob_ib = model.ib_latent_posterior.grad_log_partition
	log_prob_it = model.it_latent_posterior.grad_log_partition
	log_prob_sub = model.sub_latent_posterior.grad_log_partition

	# Test next_states
	max_slip = math.ceil(len(plu_tops)*max_slip_factor) # = 1
	assert max_slip==1 # Just making sure
	frames_per_top = math.ceil(float(n_frames)/len(plu_tops)) # =ceil(11/3)=4
	assert frames_per_top==4

	curr_state = ((0,0,1,0,amdtk.Ops.IB,-1), state_llh[0,3] + log_prob_ops[amdtk.Ops.IB] + log_prob_ib[1])
	nexts = model.next_states(curr_state, plu_tops, state_llh, max_slip, frames_per_top)
	nexts_set = set(nexts)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	expected_nexts = [
		# Can't do any new-PLU transitions because we are in HMM state 0

		# PLU-internal HMM transition
		((1,1,1,0,amdtk.Ops.NONE, -1), curr_state[1] + state_llh[1,4] + math.log(0.5)),

		# HMM self-transition
		((1,0,1,0,amdtk.Ops.NONE, -1), curr_state[1] + state_llh[1,3] + math.log(0.5))

	]
	
	expected_nexts_set = set(expected_nexts)

	if verbose:
		print("Next states after "+str(curr_state)+":")
		expected_nexts.sort()
		nexts.sort()
		print("*********EXPECTED***********")
		for item in expected_nexts:
			print(item)
		print("*********ACTUAL*************")
		for item in nexts:
			print(item)

	assert expected_nexts_set==nexts_set

	print("TEST PASSED")


	#log_ib_counts, log_it_counts, log_sub_counts, log_state_counts = model.forward_backward_noisy_channel(plu_tops, state_llh)

def end_items_test(verbose=False):

	print("Running end_items_test()...")

	model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames = create_small_model()

	perturb_small_model(model, n_mfccs)

	log_prob_ops = model.op_type_latent_posterior.grad_log_partition
	log_prob_ib = model.ib_latent_posterior.grad_log_partition
	log_prob_it = model.it_latent_posterior.grad_log_partition
	log_prob_sub = model.sub_latent_posterior.grad_log_partition

	# Test generate_end_items
	max_slip = math.ceil(len(plu_tops)*max_slip_factor) # = 1
	assert max_slip==1 # Just making sure
	frames_per_top = math.ceil(float(n_frames)/len(plu_tops)) # =ceil(11/3)=4
	assert frames_per_top==4 # Just making sure

	max_pb = len(plu_tops)-1+max_slip
	assert max_pb==3 # Just making sure

	min_pb = len(plu_tops)-1-max_slip
	assert min_pb==1 # Just making sure

	end_items = model.generate_end_items(plu_tops, state_llh, max_slip)
	end_items_set = set(end_items)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	expected_end_items = [
	 	# Insert-top end items (for all possible PLUs and final PLU indices)
		((10,2,0,3,amdtk.Ops.IT,2), 0.),
		((10,2,0,2,amdtk.Ops.IT,2), 0.),
		((10,2,0,1,amdtk.Ops.IT,2), 0.),

		((10,2,1,3,amdtk.Ops.IT,2), 0.),
		((10,2,1,2,amdtk.Ops.IT,2), 0.),
		((10,2,1,1,amdtk.Ops.IT,2), 0.),

		((10,2,2,3,amdtk.Ops.IT,2), 0.),
		((10,2,2,2,amdtk.Ops.IT,2), 0.),
		((10,2,2,1,amdtk.Ops.IT,2), 0.),

	 	# Insert-bottom and substitute end items
		# 	None, because these would require the HMM state to be 0

	 	# HMM transition end items
	 	#	We can't distinguish what kind of HMM transition,
	 	#	all we know is that they just came from another HMM state in the same PLU,
	 	#	since operation is NONE.
	 	((10,2,0,3,amdtk.Ops.NONE,2), 0.),
		((10,2,0,2,amdtk.Ops.NONE,2), 0.),
		((10,2,0,1,amdtk.Ops.NONE,2), 0.),

		((10,2,1,3,amdtk.Ops.NONE,2), 0.),
		((10,2,1,2,amdtk.Ops.NONE,2), 0.),
		((10,2,1,1,amdtk.Ops.NONE,2), 0.),

		((10,2,2,3,amdtk.Ops.NONE,2), 0.),
		((10,2,2,2,amdtk.Ops.NONE,2), 0.),
		((10,2,2,1,amdtk.Ops.NONE,2), 0.)

	]
	expected_end_items_set = set(expected_end_items)

	if verbose:
		print("End items:")
		expected_end_items.sort()
		end_items.sort()
		print("*********EXPECTED***********")
		for item in expected_end_items:
			print(item)
		print("*********ACTUAL*************")
		for item in end_items:
			print(item)

	assert expected_end_items_set==end_items_set
	print("TEST PASSED")

def prev_states_test_1(verbose=False):

	print("Running prev_states_test_1()...")

	model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames = create_small_model()

	perturb_small_model(model, n_mfccs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	log_prob_ops = model.op_type_latent_posterior.grad_log_partition
	log_prob_ib = model.ib_latent_posterior.grad_log_partition
	log_prob_it = model.it_latent_posterior.grad_log_partition
	log_prob_sub = model.sub_latent_posterior.grad_log_partition

	# Test prev_states
	max_slip = math.ceil(len(plu_tops)*max_slip_factor) # = 1
	assert max_slip==1 # Just making sure
	frames_per_top = math.ceil(float(n_frames)/len(plu_tops)) # =ceil(11/3)=4
	assert frames_per_top==4

	curr_state = ((10,2,0,3,amdtk.Ops.IT,2), 0.)
	prevs = model.prev_states(curr_state, plu_tops, state_llh, max_slip, frames_per_top)
	prevs_set = set(prevs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	expected_prevs = [
		# Prev via reversing the insert-top operation
			# Can't! The bottom is already 1 bottom PLU ahead of the top, we can't back up
			# the top PLU index anymore
		# And we can't do any other operation while in IT mode
		# So there should be ZERO possible next states from here
	]
	
	expected_prevs_set = set(expected_prevs)

	if verbose:
		print("Prev states before "+str(curr_state)+":")
		expected_prevs.sort()
		prevs.sort()
		print("*********EXPECTED***********")
		for item in expected_prevs:
			print(item)
		print("*********ACTUAL*************")
		for item in prevs:
			print(item)

	assert expected_prevs_set==prevs_set

	print("TEST PASSED")

def prev_states_test_2(verbose=False):

	print("Running prev_states_test_2()...")

	model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames = create_small_model()

	perturb_small_model(model, n_mfccs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	log_prob_ops = model.op_type_latent_posterior.grad_log_partition
	log_prob_ib = model.ib_latent_posterior.grad_log_partition
	log_prob_it = model.it_latent_posterior.grad_log_partition
	log_prob_sub = model.sub_latent_posterior.grad_log_partition

	# Test prev_states
	max_slip = math.ceil(len(plu_tops)*max_slip_factor) # = 1
	assert max_slip==1 # Just making sure
	frames_per_top = math.ceil(float(n_frames)/len(plu_tops)) # =ceil(11/3)=4
	assert frames_per_top==4

	curr_state = ((10,2,1,1,amdtk.Ops.IT,2), 0.)
	prevs = model.prev_states(curr_state, plu_tops, state_llh, max_slip, frames_per_top)
	prevs_set = set(prevs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	expected_prevs = [
		# Prev via reversing the insert-top operation (for all possible prev edit ops)
			((10,2,1,1,amdtk.Ops.NONE,1), curr_state[1] + log_prob_ops[amdtk.Ops.IT] + log_prob_it[plu_tops[2]]),
			((10,2,1,1,amdtk.Ops.IT,1), curr_state[1] + log_prob_ops[amdtk.Ops.IT] + log_prob_it[plu_tops[2]]),

		# Which is actually all we can do in insert-top mode, so this should be right.
	]
	
	expected_prevs_set = set(expected_prevs)

	if verbose:
		print("Prev states before "+str(curr_state)+":")
		expected_prevs.sort()
		prevs.sort()
		print("*********EXPECTED***********")
		for item in expected_prevs:
			print(item)
		print("*********ACTUAL*************")
		for item in prevs:
			print(item)

	assert expected_prevs_set==prevs_set

	print("TEST PASSED")

def prev_states_test_3(verbose=False):

	print("Running prev_states_test_3()...")

	model, max_slip_factor, plu_tops, state_llh, n_mfccs, n_frames = create_small_model()

	perturb_small_model(model, n_mfccs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	log_prob_ops = model.op_type_latent_posterior.grad_log_partition
	log_prob_ib = model.ib_latent_posterior.grad_log_partition
	log_prob_it = model.it_latent_posterior.grad_log_partition
	log_prob_sub = model.sub_latent_posterior.grad_log_partition

	# Test prev_states
	max_slip = math.ceil(len(plu_tops)*max_slip_factor) # = 1
	assert max_slip==1 # Just making sure
	frames_per_top = math.ceil(float(n_frames)/len(plu_tops)) # =ceil(11/3)=4
	assert frames_per_top==4

	curr_state = ((10,2,2,2,amdtk.Ops.NONE,2), 0.)
	prevs = model.prev_states(curr_state, plu_tops, state_llh, max_slip, frames_per_top)
	prevs_set = set(prevs)

	# (frame_index, hmm_state, plu_bottom_type, plu_bottom_index, edit_op, plu_top_index)

	expected_prevs = [
		# Prev via PLU-internal HMM state transition
		((9,1,2,2,amdtk.Ops.NONE,2), curr_state[1] + state_llh[10,8] + math.log(0.5)),

		# Prev via HMM self-transition
		((9,2,2,2,amdtk.Ops.NONE,2), curr_state[1] + state_llh[10,8] + math.log(0.5))
	]
	
	expected_prevs_set = set(expected_prevs)

	if verbose:
		print("Prev states before "+str(curr_state)+":")
		expected_prevs.sort()
		prevs.sort()
		print("*********EXPECTED***********")
		for item in expected_prevs:
			print(item)
		print("*********ACTUAL*************")
		for item in prevs:
			print(item)

	assert expected_prevs_set==prevs_set

	print("TEST PASSED")


if __name__ == "__main__":

	if len(sys.argv) > 1 and sys.argv[1]=='-v':
		verbose = True
	else:
		verbose = False

	print("Running noisy channel tests...")

	start_items_test(verbose)
	next_states_test_1(verbose)
	next_states_test_1a(verbose)
	next_states_test_2(verbose)

	end_items_test(verbose)
	prev_states_test_1(verbose)
	prev_states_test_2(verbose)
	prev_states_test_3(verbose)


# Bugs I have found so far:

# 	- Insert-top item in generate_start_items had wrong number of entries in tuple
#	- PLU-internal HMM state transition in next_states was looking at state_llh for the current state,
#	  not the next state
#	- Reverse insert-top transition probability in prev_states was using probability of prev top PLU insert,
#	  not probability of current top PLU insert.

