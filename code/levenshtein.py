import numpy as np
import math
import time

# THIS IS NOT REALLY CORRECT
# WIP
# Returns the best Levenshtein distance between array `s` and some array
# modified by a sequence of edit operations.
# Costs of insert, delete, and sub operations *for each character* 
# must be specified, as well as an array of prior probabilities
# of each segment for each 

def modified_lev_dist(s, alphabet_size, prob_ins, prob_del, prob_sub, priors):

    # Do some integrity checks on the arguments

    if len(s.shape) != 1 or \
            len(prob_ins.shape) != 1 or \
            len(prob_del.shape) != 1 or \
            len(prob_sub.shape) != 2 or \
            len(priors.shape) != 2:
        raise ValueError("s, prob_ins, and prob_del must be 1D numpy arrays, and prob_sub and priors must be 2D numpy arrays.")

    if prob_ins.shape != (alphabet_size,) or \
            prob_del.shape != (alphabet_size,) or \
            prob_sub.shape != (alphabet_size, alphabet_size):
        raise ValueError("Dimensions of prob_ins, prob_del, and prob_sub must correspond to alphabet size.")

    if priors.shape != (s.shape[0], alphabet_size):
        raise ValueError("priors must be a numpy array of dimensions (len(s), alphabet_size).")

    for x in s:
        if x > alphabet_size-1:
            raise ValueError("All elements of s must be less than alphabet size.")

    # Should probably also check that priors and probability matrices are normalized,
    # or normalize them here, but I'm not going to do that yet bc I'm lazy.

    # Ok, now that we've got that out of the way, begin algorithm.

    len_s = s.size
    len_t = len_s # Will need to change this later when I understand things better

    # Create numpy array `prob`
    # prob[i,j] contains the Levenshtein probabilities between s[:i-1] and a hypothetical
    # target string t', along with costs for all of the last operations on the string.
    # that is, the first i characters of s and the first j characters of t'.
    prob = np.zeros((len_s+1, len_t+1, 2*alphabet_size+1))

    # The first row represents the probabilities for a target string of length j
    # given an empty source string prefix. This means a series of insert operations.
    prob[0,0,alphabet_size+1:2*alphabet_size+1] = 1
    for j in range(1,len_t+1):
        prev_max = max(prob[0,j-1])
        # The new probabilities are the product of the previous max,
        # the insert probabilities for each segment, and the prior probabilities for the
        # current index for each segment.
        prob[0,j,1:alphabet_size+1] = prev_max * prob_ins * priors[j-1,:] # This is correct

    # The first column represents the probabilities for an empty target prefix and
    # a starting string prefix of length i. This means a series of delete operations.
    for i in range(len_s+1):
        prev_max = max(prob[i-1,0])
        # Unlike inserts, there is only one option for delete.
        # Also, we don't need to multiply by the priors because we are not using
        # an actual segment here.
        prob[i,0,0] = prev_max * prob_del[s[i-1]]

    # Iterate through the array
    # For each cell prob[i,j,c], we will calculate a list of L probabilities for that square:
    #       (with prev_max_del = max(prob[i-1,j,:]),
    #             prev_max_ins = max(prob[i,j-1,:]),
    #             prev_max_sub = max(prob[i-1,j-1,:]) )
    #   - prev_max_del * prob_del[s[i-1]] (deleting the last character of source)
    #   - prev_max_ins * prob_ins[s[i-1]] * priors[s[i-1],j-1] (inserting the last character of target, for all possible characters)
    #   - prev_max_sub * prob_sub[s[i-1]] * priors[s[i-1],j-1] (substituting last character of source for last character of target,
    #                                                                for all possible characters)
    #       (note: we don't need two cases for this anymore, because the s=t case can be taken care of
    #        in the substitutions matrix: just put all high probabilities along the diagonal -- or don't, if you want to
    #        mix things up a little.)

    for i in range(1,len_s+1):
        for j in range(1,len_t+1):

            prev_max_del = max(prob[i-1,j,:])
            prev_max_ins = max(prob[i,j-1,:])
            prev_max_sub = max(prob[i-1,j-1,:])

            # Set the probabilities array prob[i,j,:]
            # First the delete, then the inserts, then the subs

            # deleting the last character of source
            prob[i,j,0] = prev_max_del * prob_del[s[i-1]] 

            # inserting the last character of target, for all possible characters
            prob[i,j,1:alphabet_size+1] = prev_max_ins * prob_ins[s[i-1]] * priors[s[i-1],j-1]

            # substituting last character of source for last character of target, for all possible characters
            prob[i,j,alphabet_size+1:2*alphabet_size+2] = prev_max_sub * prob_sub[s[i-1]] * priors[s[i-1],j-1]
            

    print(prob)


    # Backtrace to get the likeliest sequence (WIP)

    # i = len_s+1
    # j = len_t+1
    # path = []
    # while i>=0 or j>=0:


    return max(prob[len_s,len_t,:])

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


if __name__ == '__main__':

    test_stochastic_edit_dist()

    





