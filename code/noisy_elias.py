import numpy as np
import sys 

class Element(object):
    """docstring for Element"""
    def __init__(self, num_plus):
        super(Element, self).__init__()
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
    """each row corresponding to a hidden state"""
    def __init__(self, top_string, bottom_string, likelihoods, max_edit):
        super(Matrix, self).__init__()
        self.top_string = top_string
        self.bottom_string = bottom_string    
        self.max_edit = max_edit
        self.likelihoods = likelihoods

        self.states = set(top_string)
        self.obs = set(bottom_string)
        self.num_top = len(self.states)
        self.num_plus = len(self.obs)

        self.full_chart = [[Element(self.num_plus)for i in range(len(self.top_string) + 1)] for j in range(len(self.bottom_string) + 1)]

        self.forward_chart = [[Element(self.num_plus)for i in range(len(self.top_string) + 1)] for j in range(len(self.bottom_string) + 1)]
        self.backward_chart = [[Element(self.num_plus)for i in range(len(self.top_string) + 1)] for j in range(len(self.bottom_string) + 1)]

        self.column_totals = np.zeros((len(bottom_string)))

       # define dirichlet over all edits
        self.operation_pseudocounts = np.ones((3))/3
        self.operation_sufficient_stats = np.zeros((3))

        self.ins_bot_pseudocounts = np.ones((self.num_plus))/self.num_plus
        self.ins_bot_sufficient_stats = np.zeros((self.num_plus))

        self.sub_pseudocounts = np.ones((self.num_top, self.num_plus, ))
        self.sub_sufficient_stats = np.zeros((self.num_top, self.num_plus))

        # initialize first one (doesn't matter)
        self.forward_chart[0][0].ins_top_prob = 1
        self.backward_chart[-1][-1].ins_top_prob = 1 

        # initialize first row/column
        for i in range(1, len(self.bottom_string) + 1):
            prev_sum = self.forward_chart[i-1][0].cell_sum()
            self.forward_chart[i][0].ins_bot_prob += prev_sum*self.operation_pseudocounts[1]*self.likelihoods[i-1]*self.ins_bot_pseudocounts

        for j in range(1, len(self.top_string) + 1):
            prev_sum = self.forward_chart[0][j-1].cell_sum()
            self.forward_chart[0][j].ins_top_prob += prev_sum*self.operation_pseudocounts[0]


        for i in range(len(self.bottom_string)-1, -1, -1):
            prev_sum = self.backward_chart[i+1][-1].cell_sum()
            self.backward_chart[i][-1].ins_bot_prob += prev_sum*self.operation_pseudocounts[1]*self.likelihoods[i]*self.ins_bot_pseudocounts

        for j in range(len(self.top_string)-1, -1, -1):
            prev_sum = self.backward_chart[-1][j+1].cell_sum()
            self.backward_chart[-1][j].ins_top_prob += prev_sum*self.operation_pseudocounts[0]

        # print("forward:")
        # self.pprint_chart(self.forward_chart)
        # print("backward:")
        # self.pprint_chart(self.backward_chart)
        # sys.exit()

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

        for i in range(1, len(self.bottom_string) + 1):
            cur_likelihoods = self.likelihoods[i-1]
            for j in range(1, len(self.top_string) + 1):
                top_char = self.top_string[j-1]
                bottom_plu = self.bottom_string[i-1]
                top_idx = self.top_to_idx[top_char]
                bottom_idx = self.plu_to_idx[bottom_plu]
                if abs(i-j) > self.max_edit:
                    continue
                if i > 0:
                    # insert bottom
                    to_add = self.forward_chart[i-1][j].cell_sum()*\
                                                        self.operation_pseudocounts[1] *\
                                                            self.ins_bot_pseudocounts *\
                                                                cur_likelihoods
                    print("changing forward_chart contribs")                                            
                    self.forward_chart[i][j].contribs["i-1"] = sum(to_add)
                    self.forward_chart[i][j].ins_bot_prob = to_add
                  
                if j > 0:
                    # insert top
                    to_add = self.forward_chart[i][j-1].cell_sum() *\
                                                         self.operation_pseudocounts[0]
                    self.forward_chart[i][j].contribs["j-1"] = to_add
                    self.forward_chart[i][j].ins_top_prob = to_add
                  
                if i > 0 and j > 0:
                    # substitution
                    to_add = self.forward_chart[i-1][j-1].cell_sum() *\
                                                    self.operation_pseudocounts[2] *\
                                                        self.sub_pseudocounts[top_idx] * \
                                                            cur_likelihoods
                    self.forward_chart[i][j].contribs['i-1,j-1'] =  sum(to_add)                                     
                    self.forward_chart[i][j].sub_prob = to_add
                    
    def backward(self):
        """
        each probability in the backwards chart is the probability of subsequent alignments given that we
        are in a given chart entry. Essentially the prob of finishing from the current cell. 
        """
        for i in range(len(self.bottom_string)-1, -1, -1):
            cur_likelihoods = self.likelihoods[i]
            for j in range(len(self.top_string)-1, -1, -1): 
                top_char = self.top_string[j-1]
                bottom_plu = self.bottom_string[i-1]
                top_idx = self.top_to_idx[top_char]
                bottom_idx = self.plu_to_idx[bottom_plu]
                if abs(i-j) > self.max_edit:
                    continue
                if i < len(self.bottom_string):
                    # insert bottom
                    to_add = self.backward_chart[i+1][j].cell_sum()*\
                                                        self.operation_pseudocounts[1] *\
                                                            self.ins_bot_pseudocounts[bottom_idx] *\
                                                                cur_likelihoods
                    self.backward_chart[i][j].ins_bot_prob = to_add

                if j < len(self.top_string):
                    # insert top
                    to_add = self.backward_chart[i][j+1].cell_sum() *\
                                                         self.operation_pseudocounts[0]
                    self.backward_chart[i][j].ins_top_prob = to_add
                
                if i < len(self.bottom_string) and j < len(self.top_string):
                    # substitution
                    to_add = self.backward_chart[i+1][j+1].cell_sum() *\
                                                    self.operation_pseudocounts[2] *\
                                                        self.sub_pseudocounts[top_idx] * \
                                                            cur_likelihoods
                    self.backward_chart[i][j].sub_prob = to_add

    def normalize(self):
        pass

    def forward_backward(self):
        self.forward()
        self.backward()
        return(self.matrix_product(self.forward_chart, self.backward_chart))

    def matrix_product(self, forward, backward):
        full_chart = [[Element(self.num_plus)for i in range(len(self.top_string) + 1)] for j in range(len(self.bottom_string) + 1)]
        for row_idx in range(len(forward)):
            for col_idx in range(len(forward[row_idx])):
                full_chart[row_idx][col_idx].ins_bot_prob = self.forward_chart[row_idx][col_idx].ins_bot_prob * self.backward_chart[row_idx][col_idx].ins_bot_prob
                full_chart[row_idx][col_idx].sub_prob = self.forward_chart[row_idx][col_idx].sub_prob * self.backward_chart[row_idx][col_idx].sub_prob
                full_chart[row_idx][col_idx].ins_top_prob = self.forward_chart[row_idx][col_idx].ins_top_prob * self.backward_chart[row_idx][col_idx].ins_top_prob
        self.full_chart = full_chart
        return(full_chart)


    def decode(self): 

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
    top_string = ['a','b','c']
    bottom_string = ['a', 'b', 'b']
    # bottom_string = ['x', 'y', 'z']
    # likelihoods = np.array([[.8, .1, .1], [.33333, .333333, .33333], [.1, .1, .8]])
    likelihoods = np.array([[.9, .1],[.5,.5],[.1,.9]])


    m = Matrix(top_string,bottom_string, likelihoods, 2)
    full_chart = m.forward_backward()
    print("forward")
    pprint_chart(full_chart)
    # print("backward")
    # m.pprint_chart(m.backward_chart)
    # print("combined")
    # m.pprint_chart(m.full_chart)
    # print(m.full_chart[-1][-1])

    for i, row in enumerate(m.forward_chart):
        for j, e in enumerate(row):
            print(i, j, e, e.contribs)

        