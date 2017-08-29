import numpy as np
import math
import time


# Returns the probabilities of a sequence of edit operations on a top array `s`
# which produce a bottom string of length t. (Computes forward probabilities)
# Throughout the code, `ib` stands for `insert bottom` (aka insert), and
# `it` stands for `insert top` (aka delete).
# 
# Arguments:
#   - top           : top-level array 
#                       (n)
#   - len_bot       : length of bottom-level array 
#                       ()
#   - alphabet_size : alphabet size of top & bottom level strings
#                       ()
#   - op_probs      : probability of insert bottom, insert top, and substitution
#                     operations, in that order. MUST SUM TO 1
#                       (3)
#   - prob_ib       : probability of insert bottom operations for each character
#                     GIVEN THAT an insert bottom operation has been chosen. MUST SUM TO 1
#                       (alphabet_size)
#   - prob_sub      : probability of insert bottom operations for each pair of characters
#                     GIVEN THAT an insert bottom operation has been chosen. Each ROW should sum to 1.
#                       (alphabet_size x alphabet_size)
#   - likelihoods   : likelihood of each alphabet character at each position in bottom string.
#                     Each ROW should sum to 1.
#                       (len_bot, alphabet_size)
#
# Note that we do not need a prob_it variable as long as we have a probability of the insert operation
# in general, since there is never a choice to be made about which character to insert, since it is
# given in the top string.
#
# Returns
#   - prob          : a 3-dimensional chart containing cumulative probabilities,
#                     The first dimension corresponds to the length of the top string prefix.
#                     The second dimension corresponds to the length of the bottom string prefix.
#                     The third dimension corresponds to the choice of edit operation for the current step.

def noisy_channel(top, len_bot, alphabet_size, op_probs, 
    prob_ib, prob_sub, likelihoods):

    # Do some integrity checks on the arguments

    if len(top.shape) != 1 or \
            len(prob_ib.shape) != 1 or \
            len(prob_sub.shape) != 2 or \
            len(likelihoods.shape) != 2:
        raise ValueError("top and prob_ib must be 1D numpy arrays, and prob_sub and likelihoods must be 2D numpy arrays.")

    if prob_ib.shape != (alphabet_size,) or \
            prob_sub.shape != (alphabet_size, alphabet_size):
        raise ValueError("Dimensions of prob_ib and and prob_sub must correspond to alphabet size.")

    if likelihoods.shape != (len_bot, alphabet_size):
        raise ValueError("likelihoods must be a numpy array of dimensions (len_bot, alphabet_size).")

    for x in top:
        if x > alphabet_size-1:
            raise ValueError("All elements of top must be less than alphabet size.")

    # Should probably also check that likelihoods and probability matrices are normalized,
    # but I'm not sure how best to do that given rounding errors.

    # Separate op_probs into 3 separate variables
    op_prob_it, op_prob_ib, op_prob_sub = op_probs

    # Begin

    len_top = top.size
    # len_bot given

    # Create numpy array `prob`
    # prob[i,j] contains the probability of a sequence of edit operations generating
    # a bottom string of length len_bot from the string top[:i-1], separated over
    # the last edit operation in the sequence.
    # prob[i,j,x] contains the probability of generating such a string pair from
    # a sequence of edit operations which ends with operation x.
    prob = np.zeros((len_top+1, len_bot+1, 2*alphabet_size+1))

    # Iterate through the array
    # For each cell prob[i,j,c], we will calculate a list of L probabilities for that square:
    #   - sum(prob[i-1,j,:]) * prob_it[top[i-1]] (insert top) (length 1)
    #   - sum(prob[i,j-1,:]) * prob_ib[top[i-1]] * likelihoods[top[i-1],j-1] (insert top) (length alphabet_size)
    #   - sum(prob[i-1,j-1,:]) * prob_sub[top[i-1]] * likelihoods[top[i-1],j-1] (substitute) (length alphabet_size)
    #       (Note: we don't need two cases for substitute anymore, because the s=t case can be taken care of
    #        in the substitutions matrix.)

    # Initialize the first cell
    # It doesn't matter what the individual values are as long as
    # the cell sums to 1
    # So we'll just set the first element to 1
    prob[0,0,0] = 1

    for i in range(0,len_top+1):
        for j in range(0,len_bot+1):

            if i>0:
                # Insert top operation
                prob[i,j,0] = sum(prob[i-1,j,:]) * op_prob_it #* prob_it[top[i-1]]

            if j>0:
                # Insert bottom operation
                prob[i,j,1:alphabet_size+1] = sum(prob[i,j-1,:]) * op_prob_ib * prob_ib * likelihoods[j-1,:]

            if i>0 and j>0:
                # Sub operation
                prob[i,j,alphabet_size+1:2*alphabet_size+2] = sum(prob[i-1,j-1,:]) * op_prob_sub * prob_sub[top[i-1]] * likelihoods[j-1,:]                    

    return prob

# Computes backwards probabilities, given the same inputs
# as the above function
def noisy_channel_backwards(top, len_bot, alphabet_size, op_probs, 
    prob_ib, prob_sub, likelihoods):
    
    # Do some integrity checks on the arguments

    if len(top.shape) != 1 or \
            len(prob_ib.shape) != 1 or \
            len(prob_sub.shape) != 2 or \
            len(likelihoods.shape) != 2:
        raise ValueError("top and prob_ib must be 1D numpy arrays, and prob_sub and likelihoods must be 2D numpy arrays.")

    if prob_ib.shape != (alphabet_size,) or \
            prob_sub.shape != (alphabet_size, alphabet_size):
        raise ValueError("Dimensions of prob_ib and and prob_sub must correspond to alphabet size.")

    if likelihoods.shape != (len_bot, alphabet_size):
        raise ValueError("likelihoods must be a numpy array of dimensions (len_bot, alphabet_size).")

    for x in top:
        if x > alphabet_size-1:
            raise ValueError("All elements of top must be less than alphabet size.")

    # Should probably also check that likelihoods and probability matrices are normalized,
    # but I'm not sure how best to do that given rounding errors.

    # Separate op_probs into 3 separate variables
    op_prob_it, op_prob_ib, op_prob_sub = op_probs

    # Begin

    len_top = top.size
    # len_bot given

    # Create numpy array `prob`
    # prob[i,j] contains the probability of a sequence of edit operations generating
    # a bottom string of length len_bot-j from the string top[i:], separated over
    # the first edit operation in the sequence.
    # prob[i,j,x] contains the probability of generating such a string pair from
    # a sequence of edit operations which begins with operation x.
    prob = np.zeros((len_top+1, len_bot+1, 2*alphabet_size+1))

    # Iterate through the array
    # For each cell prob[i,j,c], we will calculate a list of L probabilities for that square:
    #   - sum(prob[i-1,j,:]) * prob_it[top[i-1]] (insert top) (length 1)
    #   - sum(prob[i,j-1,:]) * prob_ib[top[i-1]] * likelihoods[top[i-1],j-1] (insert top) (length alphabet_size)
    #   - sum(prob[i-1,j-1,:]) * prob_sub[top[i-1]] * likelihoods[top[i-1],j-1] (substitute) (length alphabet_size)
    #       (Note: we don't need two cases for substitute anymore, because the s=t case can be taken care of
    #        in the substitutions matrix.)

    # Initialize the 'last' cell
    # It doesn't matter what the individual values are as long as
    # the cell sums to 1
    # So we'll just set the first element to 1
    prob[len_top,len_bot,0] = 1

    for i in range(len_top,-1,-1):
        for j in range(len_bot,-1,-1):

            if i<len_top:
                # Insert top operation
                prob[i,j,0] = sum(prob[i+1,j,:]) * op_prob_it #* prob_it[top[i-1]]

            if j<len_bot:
                # Insert bottom operation
                prob[i,j,1:alphabet_size+1] = sum(prob[i,j+1,:]) * op_prob_ib * prob_ib * likelihoods[j,:]

            if i<len_top and j<len_top:
                # Sub operation
                prob[i,j,alphabet_size+1:2*alphabet_size+2] = sum(prob[i+1,j+1,:]) * op_prob_sub * prob_sub[top[i]] * likelihoods[j,:]                    

    return prob



# Given a Levenshtein chart of the format produced by noisy_channel(),
# return a corresponding edit sequence.
#    
# Deterministic mode (stochastic=False): Returns the most likely edit sequence
# indicated by the chart
# 
# Stochastic mode (stochastic=True): Samples a valid edit sequence according
# to the probabilities given in the chart

def backtrace(prob, stochastic=True):
    # TODO
    pass


# Returns the stochastic edit distance, as defined by Ristad & Yianilos,
# between array `s` and array `t`.
# Probabilities of insert, delete, and sub operations *for each character* 
# must be specified 

def stochastic_edit_dist(s, t, alphabet_size, prob_ins, prob_del, prob_sub):

    # Do some integrity checks on the arguments

    if len(s.shape) != 1 or \
            len(prob_ins.shape) != 1 or \
            len(prob_del.shape) != 1 or \
            len(prob_sub.shape) != 2:
        raise ValueError("s, prob_ins, and prob_del must be 1D numpy arrays, and prob_sub must be a 2D numpy array.")

    if prob_ins.shape != (alphabet_size,) or \
            prob_del.shape != (alphabet_size,) or \
            prob_sub.shape != (alphabet_size, alphabet_size):
        raise ValueError("Dimensions of prob_ins, prob_del, and prob_sub must correspond to alphabet size.")

    for x in s:
        if x > alphabet_size-1:
            raise ValueError("All elements of s must be less than alphabet size.")

    # Should probably also check that probability matrices are normalized,
    # or normalize them here, but I'm not going to do that yet bc I'm lazy.

    # Ok, now that we've got that out of the way, begin algorithm.

    len_s = s.size
    len_t = t.size

    # Create numpy array `prob`
    # prob[i,j] will the stochastic edit distance between s[:i-1] and t[:j-1];
    # that is, the i-length prefix of s and the j-length prefix of t
    prob = np.zeros((len_s+1, len_t+1))

    # The first row & column represent the distance between one string
    # and the empty string. Therefore the transformation must consist of entirely
    # insert/delete operations. Fill in these cells of the matrix.

    # The probability of empty string -> empty string is 1.
    prob[0,0] = 1

    # Iterate through the array
    # For each cell prob[i,j], we will sum 3 probabilities for that square:
    #   - prob[i-1,j] + prob_del (deleting the last char of source)
    #   - prob[i,j-1] + prob_ins (inserting the last char of target)
    #   - prob[i-1,j-1] + prob_sub (substituting last char of target for last char of source)

    for i in range(len_s+1):
        for j in range(len_t+1):

            if i>0:
                # Delete operation
                prob[i,j] += prob[i-1,j]*prob_del[s[i-1]]

            if j>0:
                # Insert operation
                prob[i,j] += prob[i,j-1]*prob_ins[t[j-1]]

            if i>0 and j>0:
                # Sub operation
                prob[i,j] += prob[i-1,j-1]*prob_sub[s[i-1],t[j-1]]
            

    print(prob)

    return prob[i,j]





# Returns the Levenshtein distance between array `s` and array `t`
# Costs of insert, delete, and sub operations *for each character* 
# must be specified. 

def lev_dist_advanced(s, t, cost_ins, cost_del, cost_sub):

    len_s = s.size
    len_t = t.size

    # Create numpy array `dist`
    # dist[i,j] contains the Levenshtein distance between s[:i-1] and t[:j-1];
    # that is, the first i characters of s and the first j characters of t
    dist = np.full((len_s+1, len_t+1), math.inf)

    # The first row & column represent the distance between one string
    # and the empty string. Therefore the transformation must consist of entirely
    # insert/delete operations. Fill in these cells of the matrix.

    # First column: Distance between prefixes of s and the empty string (delete operations)
    for i in range(len_s+1):
        # The number of delete operations is equal to the length of the s prefix -- that is, i.
        # Multiply this by the cost of a delete operation.
        dist[i,0] = i*cost_del[0,s[i-1]]

    # First row: Distance between prefixes of t and the empty string (insert operations)
    for j in range(len_t+1):
        # The number of insert operations is equal to the length of the t prefix -- that is, j.
        # Multiply this by the cost of an insert operation.
        dist[0,j] = j*cost_ins[0,t[j-1]]

    # Iterate through the array
    # For each cell dist[i,j], we will pick the minimum of 3 potential L distances for that square:
    #   - dist[i-1,j] + cost_del (deleting the last character of source)
    #   - dist[i,j-1] + cost_ins (inserting the last character of target)
    #   - dist[i-1,j-1]
    #       + cost_sub (IF s[i-1] != t[j-1])
    #       + 0 otherwise

    for i in range(1,len_s+1):
        for j in range(1,len_t+1):
            
            delete = dist[i-1,j] + cost_del[0,s[i-1]]
            insert = dist[i,j-1] + cost_ins[0,t[j-1]]

            if s[i-1]==t[j-1]:
                substitute = dist[i-1,j-1] + 0
            else: 
                substitute = dist[i-1,j-1] + cost_sub[s[i-1],t[j-1]]

            # Pick the minimum
            dist[i,j] = min(delete, insert, substitute)

    print(dist)

    # Trace the optimal path backwards to find the sequence of operations
    # i = len_s+1
    # j = len_t+1
    # path = [(i,j)]
    # while i>=0 or j>=0:

    #     if i==0:
    #         delete = math.inf
    #         substitute = math.inf
    #     else:
    #         delete = dist[i-1,j]
    #     if j==0:
    #         insert = math.inf
    #     else:

    #         pass

    return dist[len_s, len_t]




# Returns the Levenshtein distance between string `s` and string `t`
# Optionally, costs of insert, delete, and sub operations may be specified
# By default, they are all 1

def lev_dist(s, t, cost_ins=1, cost_del=1, cost_sub=1):

    len_s = len(s)
    len_t = len(t)
    
    # Create numpy array `dist`
    # dist[i,j] contains the Levenshtein distance between s[:i-1] and t[:j-1];
    # that is, the first i characters of s and the first j characters of t
    dist = np.full((len_s+1, len_t+1), math.inf)

    # The first row & column represent the distance between one string
    # and the empty string. Therefore the transformation must consist of entirely
    # insert/delete operations. Fill in these cells of the matrix.

    # First column: Distance between prefixes of s and the empty string (delete operations)
    for i in range(len_s+1):
        # The number of delete operations is equal to the length of the s prefix -- that is, i.
        # Multiply this by the cost of a delete operation.
        dist[i,0] = i*cost_del

    # First row: Distance between prefixes of t and the empty string (insert operations)
    for j in range(len_t+1):
        # The number of insert operations is equal to the length of the t prefix -- that is, j.
        # Multiply this by the cost of an insert operation.
        dist[0,j] = j*cost_ins

    # Iterate through the array
    # For each cell dist[i,j], we will pick the minimum of 3 potential L distances for that square:
    #   - dist[i-1,j] + cost_del (deleting the last character of source)
    #   - dist[i,j-1] + cost_ins (inserting the last character of target)
    #   - dist[i-1,j-1]
    #       + cost_sub (IF s[i-1] != t[j-1])
    #       + 0 otherwise

    for i in range(1,len_s+1):
        for j in range(1,len_t+1):
            
            delete = dist[i-1,j] + cost_del
            insert = dist[i,j-1] + cost_ins

            if s[i-1]==t[j-1]:
                substitute = dist[i-1,j-1] + 0
            else: 
                substitute = dist[i-1,j-1] + cost_sub

            # Pick the minimum
            dist[i,j] = min(delete, insert, substitute)

    print(dist)

    # Trace the optimal path backwards to find the sequence of operations
    # i = len_s+1
    # j = len_t+1
    # path = [(i,j)]
    # while i>=0 or j>=0:

    #     if i==0:
    #         delete = math.inf
    #         substitute = math.inf
    #     else:
    #         delete = dist[i-1,j]
    #     if j==0:
    #         insert = math.inf
    #     else:

    #         pass

    return dist[len_s, len_t]



'''
function LevenshteinDistance(char s[1..m], char t[1..n]):
  // for all i and j, d[i,j] will hold the Levenshtein distance between
  // the first i characters of s and the first j characters of t
  // note that d has (m+1)*(n+1) values
  declare int d[0..m, 0..n]
 
  set each element in d to zero
 
  // source prefixes can be transformed into empty string by
  // dropping all characters
  for i from 1 to m:
      d[i, 0] := i
 
  // target prefixes can be reached from empty source prefix
  // by inserting every character
  for j from 1 to n:
      d[0, j] := j
 
  for j from 1 to n:
      for i from 1 to m:
          if s[i] = t[j]:
            substitutionCost := 0
          else:
            substitutionCost := 1
          d[i, j] := minimum(d[i-1, j] + 1,                   // deletion
                             d[i, j-1] + 1,                   // insertion
                             d[i-1, j-1] + substitutionCost)  // substitution
 
  return d[m, n]
'''

def test_lev_dist():

    print('Testing lev_dist...')

    s = '''
function LevenshteinDistance(char s[1..m], char t[1..n]):
  // for all i and j, d[i,j] will hold the Levenshtein distance between
  // the first i characters of s and the first j characters of t
  // note that d has (m+1)*(n+1) values
  declare int d[0..m, 0..n]
 
  set each element in d to zero
 
  // source prefixes can be transformed into empty string by
  // dropping all characters
  for i from 1 to m:
      d[i, 0] := i
 
  // target prefixes can be reached from empty source prefix
  // by inserting every character
  for j from 1 to n:
      d[0, j] := j
 
  for j from 1 to n:
      for i from 1 to m:
          if s[i] = t[j]:
            substitutionCost := 0
          else:
            substitutionCost := 1
          d[i, j] := minimum(d[i-1, j] + 1,                   // deletion
                             d[i, j-1] + 1,                   // insertion
                             d[i-1, j-1] + substitutionCost)  // substitution
 
  return d[m, n]
'''
    t = '''
function LevnshteinDistance(char s[1..m], char t[1..n]):
  // for all i and j, d[i,j] will hold the Levenshtein distance between
  // the first i characters of s and the first j characters of t
  // note that d has (m+1)*(n+1) values
  declare int d[0..m, 0..n]
 
  set each element in d to zero
 
  // source prefixes can be transformed into empty string by
  // dropping all characters
  for i from 1 to m:
      d[i, 0] := i
 
  // target prefixes can be reached from empty source prefix
  // by inserting every character
  for j from 1 to n:
      d[0, j] := j
 
  for j from 1 to n:
      for i from 1 to m:
          if s[i] = t[j]:
            substitutionCost := 0
          else:
            substitutionCost := 1
          d[i, j] := minimum(d[i-1, j] + 1,                   // deletion
                             d[i, j-1] + 1,                   // insertion
                             d[i-1, j-1] + substitutiionCost)  // substitution
 
  return d[m, n]
'''

    cost_ins = 1
    cost_del = 1
    cost_sub = 1

    start = time.time()
    distance = lev_dist(s,t, cost_ins,cost_del,cost_sub)
    elapsed = time.time()-start


    print('lev_dist between strings \''+s+'\' and \''+t+ \
        '\' with costs ('+str(cost_ins)+','+str(cost_del)+','+str(cost_sub)+ \
        ') is '+str(distance))
    print('Took {0:.2f} seconds'.format(elapsed))

def test_lev_dist_advanced():

    print('Testing lev_dist_advanced...')

    s = np.array([2,0,2,2,2,2])
    t = np.array([2,0,1])

    cost_ins = np.ones((1,3))
    cost_del = np.ones((1,3))
    cost_sub = np.ones((3,3))

    distance = lev_dist_advanced(s, t, cost_ins, cost_del, cost_sub)

    print(distance)


def test_modified_lev_dist():

    print('Testing modified_lev_dist...')

    s = np.array([2,0,1])
    alphabet_size = 4
    prob_ins = np.array([0.1,0.4,0.4,0.1])
    prob_del = np.array([0.25,0.25,0.25,0.25])
    prob_sub = np.array([[0.9,0.01,0.01,0.08],
                         [0.01,0.9,0.01,0.08],
                         [0.01,0.01,0.9,0.08],
                         [0.01,0.01,0.08,0.9]])
    priors = np.array([[0.01,0.01,0.9,0.08],
                       [0.01,0.9,0.01,0.08],
                       [0.9,0.01,0.01,0.08]])

    result = modified_lev_dist(s, alphabet_size, prob_ins, prob_del, prob_sub, priors)
    print("Result: "+str(result))

def test_stochastic_edit_dist():

    print('Testing stochastic_edit_dist...')

    s = np.array([2])
    t = np.array([1])
    alphabet_size = 4
    prob_ins = np.array([0.01,0.01,0.01,0.01])
    prob_del = np.array([0.02,0.02,0.02,0.02])
    prob_sub = np.array([[0.13,0.03,0.03,0.03],
                         [0.03,0.13,0.03,0.03],
                         [0.03,0.03,0.13,0.03],
                         [0.03,0.03,0.03,0.13]])

    result = stochastic_edit_dist(s, t, alphabet_size, prob_ins, prob_del, prob_sub)
    print("Result: "+str(result))

def test_noisy_channel():

    print('Testing noisy_channel...')

    top = np.array([2,0,1])
    len_bot = 3
    alphabet_size = 4
    op_probs = (0.1,0.1,0.8)
    prob_ib = np.array([1,1,1,1])
    prob_it = np.array([0.25,0.25,0.25,0.25])
    prob_sub = np.array([[0.9,0.01,0.01,0.08],
                         [0.01,0.9,0.01,0.08],
                         [0.01,0.01,0.9,0.08],
                         [0.01,0.01,0.08,0.9]])
    likelihoods = np.array([[0.01,0.01,0.9,0.08],
                       [0.01,0.9,0.01,0.08],
                       [0.9,0.01,0.01,0.08]])

    result = noisy_channel(top, len_bot, alphabet_size, op_probs,
         prob_ib, prob_it, prob_sub, likelihoods)
    print("Result: "+str(result))

def test_noisy_channel_simple():

    print('Testing noisy_channel...')

    top = np.array([0])
    len_bot = 2
    alphabet_size = 1
    op_probs = (0.1,0.1,0.8)
    prob_ib = np.array([1])
    prob_sub = np.array([[1]])
    likelihoods = np.array([[1],[1]])

    result_chart = noisy_channel(top, len_bot, alphabet_size, op_probs,
         prob_ib, prob_sub, likelihoods)
    result_prob = sum(result_chart[-1,-1])
    print("Chart:\n"+str(result_chart))
    print("Result: "+str(result_prob))

def test_noisy_channel_back_simple():

    print('Testing noisy_channel_backwards...')

    top = np.array([0])
    len_bot = 2
    alphabet_size = 1
    op_probs = (0.1,0.1,0.8)
    prob_ib = np.array([1])
    prob_sub = np.array([[1]])
    likelihoods = np.array([[1],[1]])

    result_chart = noisy_channel_backwards(top, len_bot, alphabet_size, op_probs,
         prob_ib, prob_sub, likelihoods)
    result_prob = sum(result_chart[-1,-1])
    print("Chart:\n"+str(result_chart))
    print("Result: "+str(result_prob))


if __name__ == '__main__':

    test_noisy_channel_back_simple()

    





