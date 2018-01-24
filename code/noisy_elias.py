import numpy as np
import sys 
from generate_ab_data import generate_data

class Element(object):
    """
    Element object 
    Each entry in the matrix is an element
    An element has attributes:
        sub_prob: the probability of substitution (for each plu to each plu) (|plus|x|plus| matrix)
        ins_bot_prob: the prob of inserting each bottom-level plu (|plus|-length vector)
        ins_top_prob: probability of inserting the top-level phone (scalar)
        contribs: dict of the contributions of the cells at i-1, j-1, and diagonal to this cell
    """
    def __init__(self, num_plus, top):
        super(Element, self).__init__()
        self.top_plu = top
        self.sub_prob = np.zeros((num_plus))
        self.ins_bot_prob = np.zeros((num_plus))
        self.ins_top_prob = 0
        self.contribs = {"i-1":0, "j-1": 0, "i-1,j-1": 0}

    def __str__(self):
        return "sub: {}, ins_bot: {}, ins_top: {}".format(self.sub_prob, self.ins_bot_prob, self.ins_top_prob)
    def cell_sum(self):
        # sum up the whole cell 
        return sum([sum(self.sub_prob), sum(self.ins_bot_prob), self.ins_top_prob])

class Matrix(object):
    """A Matrix stores/aggregates the pseudocounts for each operation"""
    def __init__(self, top_string, bottom_alphabet, likelihoods, max_edit, prob_dict):
        super(Matrix, self).__init__()
        self.top_string = top_string
        # self.bottom_string = bottom_string  
        self.bottom_string_length = len(likelihoods) 
        self.max_edit = max_edit
        self.likelihoods = likelihoods

        self.states = set(top_string)
        self.obs = bottom_alphabet
        self.num_top = len(self.states)
        # self.num_plus = len(self.obs)
        self.num_plus = 3
        # initialize empty charts with |bottom| rows and |top| columns
        self.full_chart = [[Element(self.num_plus, None)for i in range(len(self.top_string) + 1)] for j in range(self.bottom_string_length + 1)]

        self.forward_chart = [[Element(self.num_plus, None)for i in range(len(self.top_string) + 1)] for j in range(self.bottom_string_length + 1)]
        self.backward_chart = [[Element(self.num_plus, None)for i in range(len(self.top_string) + 1)] for j in range(self.bottom_string_length + 1)]

        self.operation_pseudocounts = prob_dict["ops"]
        self.ins_bot_pseudocounts = prob_dict["ib"]
        self.sub_pseudocounts = prob_dict["sub"]

        # initialize first one (doesn't matter how it's initialized since it's P(aligning nothing w/ nothing))
        self.forward_chart[0][0].ins_top_prob = 1
        self.backward_chart[-1][-1].ins_top_prob = 1 

        # initialize first row/column
        for i in range(1, self.bottom_string_length + 1):
            prev_sum = self.forward_chart[i-1][0].cell_sum()
            self.forward_chart[i][0].ins_bot_prob += prev_sum*self.operation_pseudocounts[1]*self.likelihoods[i-1]*self.ins_bot_pseudocounts

        for j in range(1, len(self.top_string) + 1):
            prev_sum = self.forward_chart[0][j-1].cell_sum()
            self.forward_chart[0][j].ins_top_prob += prev_sum*self.operation_pseudocounts[0]

        for i in range(self.bottom_string_length-1, -1, -1):
            prev_sum = self.backward_chart[i+1][-1].cell_sum()
            self.backward_chart[i][-1].ins_bot_prob += prev_sum*self.operation_pseudocounts[1]*self.likelihoods[i]*self.ins_bot_pseudocounts

        for j in range(len(self.top_string)-1, -1, -1):
            prev_sum = self.backward_chart[-1][j+1].cell_sum()
            self.backward_chart[-1][j].ins_top_prob += prev_sum*self.operation_pseudocounts[0]

        # fill in the top string:
        for j in range(1, len(self.top_string) + 1):
            for i in range(len(self.full_chart[0])):
                self.full_chart[i][j].top_plu = self.top_string[i-1]

        # mappings to go from plu/character to index in each list (e.g. the sub list)
        self.plu_to_idx = {x:i for i, x in enumerate(sorted(self.obs))}
        self.idx_to_plu = {i:x for x, i in self.plu_to_idx.items()}
        self.top_to_idx = {x:i for i, x in enumerate(sorted(self.states))}
        self.idx_to_top = {i:x for x, i in self.top_to_idx.items()}

    def __str__(self):
        list_form = [[0 for i in range(len(self.full_chart[0]))] for j in range(len(self.full_chart))]
        for i, row in enumerate(self.full_chart):
            for j, element in enumerate(row):
                list_form[i][j] = self.full_chart[i][j].cell_sum()

        to_ret = str(list_form)
        return(to_ret)

    def forward(self):
        """
        each entry in the forward chart is the probability of the alignment up to now. each chart entry stores the probability of ins_bot, sub, and 
        ins_top, which are added to get the total cell probability. Essentially the prob of getting to
        the current chart cell from the start. 
        """
        # column_totals = [[np.zeros((self.num_top)), np.zeros((self.num_top)), 0]]*len(self.top_string)

        for i in range(1, self.bottom_string_length + 1):
            for j in range(1, len(self.top_string) + 1):
                top_char = self.top_string[j-1]
                top_idx = self.top_to_idx[top_char]
                if abs(i-j) > self.max_edit:
                    continue
                if i > 0:
                    # insert bottom
                    to_add = self.forward_chart[i-1][j].cell_sum()*\
                                                        self.operation_pseudocounts[1] *\
                                                            self.ins_bot_pseudocounts *\
                                                                self.likelihoods[i-1]

                    self.forward_chart[i][j].ins_bot_prob = to_add
                    self.forward_chart[i][j].contribs['i-1'] = np.sum(to_add)
                  
                if j > 0:
                    # insert top
                    to_add = self.forward_chart[i][j-1].cell_sum() *\
                                                         self.operation_pseudocounts[0]
                    self.forward_chart[i][j].ins_top_prob = to_add
                    self.forward_chart[i][j].contribs['j-1'] = to_add
                    self.full_chart[i][j].top_plu = self.top_string[j-1]
                  
                if i > 0 and j > 0:
                    # substitution
                    to_add = self.forward_chart[i-1][j-1].cell_sum() *\
                                                    self.operation_pseudocounts[2] *\
                                                        self.sub_pseudocounts[top_idx] * \
                                                            self.likelihoods[i-1]
                    self.forward_chart[i][j].sub_prob = to_add
                    self.forward_chart[i][j].contribs['i-1,j-1'] = np.sum(to_add)
                    self.full_chart[i][j].top_plu = self.top_string[j-1]
                    
    def backward(self):
        for i in range(self.bottom_string_length-1, -1, -1):
            for j in range(len(self.top_string)-1, -1, -1): 
                top_char = self.top_string[j-1]
                top_idx = self.top_to_idx[top_char]
                if abs(i-j) > self.max_edit:
                    continue
                if i < self.bottom_string_length:
                    # insert bottom
                    to_add = self.backward_chart[i+1][j].cell_sum()*\
                                                        self.operation_pseudocounts[1] *\
                                                            self.ins_bot_pseudocounts *\
                                                                self.likelihoods[i]
                    self.backward_chart[i][j].ins_bot_prob = to_add

                if j < len(self.top_string):
                    # insert top
                    to_add = self.backward_chart[i][j+1].cell_sum() *\
                                                         self.operation_pseudocounts[0]
                    self.backward_chart[i][j].ins_top_prob = to_add
                
                if i < self.bottom_string_length and j < len(self.top_string):
                    # substitution
                    to_add = self.backward_chart[i+1][j+1].cell_sum() *\
                                                    self.operation_pseudocounts[2] *\
                                                        self.sub_pseudocounts[top_idx] * \
                                                            self.likelihoods[i]
                    self.backward_chart[i][j].sub_prob = to_add

    def forward_backward(self):
        # combine forward and backward to get expected counts of each operation
        self.forward()
        self.backward()
        return self.matrix_product(self.forward_chart, self.backward_chart)

    def matrix_product(self, forward, backward):
        # helper function to multiply charts 
        for row_idx in range(len(forward)):
            for col_idx in range(len(forward[row_idx])):
                self.full_chart[row_idx][col_idx].ins_bot_prob = self.forward_chart[row_idx][col_idx].ins_bot_prob * self.backward_chart[row_idx][col_idx].ins_bot_prob
                self.full_chart[row_idx][col_idx].sub_prob = self.forward_chart[row_idx][col_idx].sub_prob * self.backward_chart[row_idx][col_idx].sub_prob
                self.full_chart[row_idx][col_idx].ins_top_prob = self.forward_chart[row_idx][col_idx].ins_top_prob * self.backward_chart[row_idx][col_idx].ins_top_prob

        return self.full_chart

    def decode(self): 
        # find path of max contribution 
        output_str = []
        i, j = len(self.forward_chart)-1,len(self.forward_chart[0])-1
        while i >0 and j >0:
            # sort contribs by value
            contribs = sorted(self.forward_chart[i][j].contribs.items(), key= lambda x: x[1])
            max_contrib_key = contribs[-1][0].strip()
            if max_contrib_key == 'i-1':
                # insert bottom of most likely phone from cell i-1
                ib_probs = self.forward_chart[i-1][j].ins_bot_prob
                best_idx = np.argmax(ib_probs)
                best_plu = self.idx_to_plu[best_idx]
                output_str.append(best_plu)
                i = i-1
            if max_contrib_key == 'j-1':
                # insert top, do nothing to bottom string
                j = j-1
            else:
                # substitute, 
                sub_probs = self.forward_chart[i-1][j-1].sub_prob
                best_idx = np.argmax(sub_probs)
                best_plu = self.idx_to_plu[best_idx]

                output_str.append(best_plu)

                i = i-1
                j = j-1

        output_str.reverse()
        return output_str

class InferenceEngine(object):
    """docstring for InferenceEngine"""
    def __init__(self, top_strings, bottom_strings, likelihoods):
        super(InferenceEngine, self).__init__()
        self.top_strings = top_strings
        self.bottom_strings = bottom_strings
        self.likelihoods = likelihoods
        self.bottom_alphabet = set([x for string in bottom_strings for x in string])
        self.top_alphabet = set([x for string in top_strings for x in string])


        self.num_top = len(self.top_alphabet)
        self.num_plus = len(self.bottom_alphabet)

        # define dirichlet over all edits
        self.operation_pseudocounts = np.ones((3))/3
        self.operation_suff_stats = np.zeros((3))

        self.ins_bot_pseudocounts = np.ones((self.num_plus))/self.num_plus
        self.ins_bot_suff_stats = np.zeros((self.num_plus))

        self.sub_pseudocounts = np.ones((self.num_top, self.num_plus))
        self.sub_suff_stats = np.zeros((self.num_top, self.num_plus))


    def learn(self):
        prob_dict = {'ops': self.operation_pseudocounts, 'ib': self.ins_bot_pseudocounts, 'sub': self.sub_pseudocounts}
        for t,b,l in zip(self.top_strings, self.bottom_strings, self.likelihoods):

            m = Matrix(t,self.bottom_alphabet,l,2, prob_dict)
            print(t)
            print(b)
            self.acc_sufficient_stats(m)


    def acc_sufficient_stats(self, m):
        chart = m.forward_backward()
        print(m.decode())
        pprint_chart(chart)

        self.operation_suff_stats += m.operation_pseudocounts

        for i in range(1, len(chart)):
            for j in range(1, len(chart[0])):
                element = chart[i][j]
                top_plu_idx = m.top_to_idx[element.top_plu]

                self.ins_bot_suff_stats += element.ins_bot_prob
                self.sub_suff_stats[top_plu_idx] += element.sub_prob

    def local_updates(self):
        pass
    def global_updates(self):
        pass



def pprint_chart(chart):
    list_form = [[0 for i in range(len(chart[0]))] for j in range(len(chart))]
    for i, row in enumerate(chart):
        str_row = ""
        for j, element in enumerate(row):

            # list_form[i][j] = chart[i][j].cell_sum()
            str_row += " {:5.4f} ".format(chart[i][j].cell_sum())
        print(str_row)
    # print(to_ret)


if __name__ == '__main__':
    # top_string = ['a','b','c']
    # bottom_string = ['a', 'b', 'b']
    # bottom_string = ['x', 'y', 'z']
    # likelihoods = np.array([[.8, .1, .1], [.33333, .333333, .33333], [.1, .1, .8]])
    # likelihoods = np.array([[.9, .1],[.5,.5],[.9,.1]])

    # likelihoods = np.array([[0, 1],[0,1],[0,1]])

    # m = Matrix(top_string,bottom_string, likelihoods, 2)
    # full_chart = m.forward_backward()
    # print("forward")
    # pprint_chart(m.forward_chart)
    # print("backward")
    # pprint_chart(m.backward_chart)
    # print("combined")
    # pprint_chart(m.full_chart)
    # print("".join(top_string))
    # print("".join(m.decode()))

    top_strings, bottom_strings, likelihoods = generate_data()

    inf_eng = InferenceEngine(top_strings, bottom_strings, likelihoods)
    inf_eng.learn()
    # bottom_alphabet = set([x for string in bottom_strings for x in string])

    # for t,b,l in zip(top_strings, bottom_strings, likelihoods):
    #     m = Matrix(t,bottom_alphabet,l,2)
    #     # have it return pseudocounts 
    #     full_chart = m.forward_backward()

    print('done')



