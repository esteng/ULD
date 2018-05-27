import numpy as np

# we need to start with a top level string 
# do some edit operations to get bottom level
# have each bottom level type index a 3-state hmm
# for each 3 state hmm
	# sample some transitions etc 
	# have each state index a GMM
	# each GMM generate a 2-dimensional vector 

# input: top level string
# parameters we need to specify: 
	# edit operation parameters
		# ins_top (1)
		# ins_bot (n)
		# sub (n) 
	# hmm 
		# initial (3)
		# transition (3x3)
		# emission (3xm)
	# GMM
		# num of components 
		# mean, variance for each component 
		# mixture proportions 



# for simplicity, let's make all top level strings 10 characters long
# let's have a 4 character alphabet 

alphabet = [0,1,2,3]


