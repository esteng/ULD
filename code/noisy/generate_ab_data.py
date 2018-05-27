import numpy as np 
import sys 


def get_prob(idx, high):
    pseudocounts = np.ones(3)
    if high:    
        pseudocounts[idx] = 10
    else:
        pseudocounts[idx] = 2
    return np.random.dirichlet(pseudocounts)

def generate_data():
    np.random.seed(1234)

    top_alphabet = ['a', 'b', 'c']
    bottom_alphabet = ['d', 'e', 'f']

    max_length = 10
    min_length = 3
    n_sents = 100


    lengths = np.random.randint(min_length, max_length, n_sents)

    top_strings, bottom_strings, all_likelihoods = [], [], []
    for l in lengths:
        indexes = np.random.randint(0, len(top_alphabet), l)
        top_string = [top_alphabet[idx] for idx in indexes]
        # add some noise for the bottom string 
        noisy_probs = np.random.binomial(1, .8, l)
        bottom_string = []
        likelihoods = []
        for idx in indexes:
            if noisy_probs[idx] == 1:
                # do corresponding 
                bottom_string.append(bottom_alphabet[idx])
                likelihoods.append(get_prob(idx, True))
            else:
                bottom_string.append(np.random.choice(bottom_alphabet))
                likelihoods.append(get_prob(idx, False))

        top_strings.append(top_string)
        bottom_strings.append(bottom_string)
        all_likelihoods.append(likelihoods)

    # print(top_strings[0:3])
    # print(bottom_strings[0:3])
    # print(all_likelihoods[0:3])
    return (top_strings, bottom_strings, all_likelihoods)

