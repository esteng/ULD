import numpy as np

class Element(object):
    """docstring for Element"""
    def __init__(self, num_plus):
        super(Element, self).__init__()

        self.sub_prob = np.zeros((num_plus))
        self.ins_bot_prob = np.zeros((num_plus))
        self.ins_top_prob = 0
    def cell_sum(self):
        # sum up the whole cell 
        print(self.sub_prob, self.ins_bot_prob)
        return sum([sum(self.sub_prob), sum(self.ins_bot_prob)])

class Column(object):
    """each column corresponds to an observation time-step"""
    def __init__(self):
        super(Column, self).__init__()
        self.elements = []
        self.sub_sum = 0
        self.ins_top_sum = 0
        self.ins_bot_sum = 0
    def add_element(self, element):
        self.elements.append(element)
        self.sub_sum = np.logaddexp(self.sub_sum, element.sub_prob)

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
        self.chart = [[Element(self.num_plus)]*(len(self.top_string) + 1)]*(len(self.bottom_string) + 1)
        self.column_totals = np.zeros((len(bottom_string)))

        # initialize first one (doesn't matter)
        self.chart[0][0].ins_top_prob = 1
        # define dirichlet over all edits
        self.operation_pseudocounts = np.ones((3))/3
        self.operation_sufficient_stats = np.zeros((3))

        self.ins_bot_pseudocounts = np.ones((self.num_plus))/self.num_plus
        self.ins_bot_sufficient_stats = np.zeros((self.num_plus))

        self.sub_pseudocounts = np.ones((self.num_plus, self.num_top))
        self.sub_sufficient_stats = np.zeros((self.num_plus, self.num_top))

        print("initialized a matrix object with")
        print("ins_bot_pseudocounts")
        print(self.ins_bot_pseudocounts.shape)
        print("sub_pseudocounts")
        print(self.sub_pseudocounts.shape)
        self.plu_to_idx = {x:i for i, x in enumerate(sorted(self.obs))}
        self.idx_to_plu = {i:x for x, i in self.plu_to_idx.items()}
        self.top_to_idx = {x:i for i, x in enumerate(sorted(self.states))}
        self.idx_to_top = {i:x for x, i in self.top_to_idx.items()}

        print("mappings")
        print(self.plu_to_idx)
        print(self.idx_to_plu)
        print(self.top_to_idx)
        print(self.idx_to_top)

    def __str__(self):
        list_form = [[0]*len(self.chart[0])]*len(self.chart)
        for i, row in enumerate(self.chart):
            for j, element in enumerate(row):
                list_form[i][j] = self.chart[i][j].cell_sum()

        to_ret = str(list_form)
        return(to_ret)



    def forward(self):
        # need to make lookup by plu index, not by index in the string
        for i in range(len(self.bottom_string) + 1):
            for j in range(len(self.top_string) + 1):
                top_char = self.top_string[j-1]
                bottom_plu = self.bottom_string[i-1]
                top_idx = self.top_to_idx[top_char]
                bottom_idx = self.plu_to_idx[bottom_plu]
                print(i,j)
                if abs(i-j) > self.max_edit:
                    continue
                if i > 0:
                    print("ins_bot")
                    # insert bottom
                    to_add = self.chart[i][j-1].cell_sum()*\
                                                        self.operation_pseudocounts[1] *\
                                                            self.ins_bot_pseudocounts[bottom_idx] *\
                                                                self.likelihoods[bottom_idx]

                    self.chart[i][j].ins_bot_prob[bottom_idx] = to_add
                    # self.column_totals[i] = np.logaddexp(self.column_totals[i], to_add)

                    self.operation_sufficient_stats[1] += 1
                    self.ins_bot_sufficient_stats[bottom_idx] += 1

                if j > 0:
                    print("ins_top")
                    # insert top
                    to_add = self.chart[i-1][j].cell_sum() *\
                                                         self.operation_pseudocounts[0]
                    self.chart[i][j].ins_top_prob = to_add
                    # self.column_totals[i] = np.logaddexp(self.column_totals[i], to_add)
                    
                    self.operation_sufficient_stats[0] += 1
                
                if i > 0 and j > 0:
                    # substitution
                    print("sub")
                    to_add = self.chart[i-1][j-1].cell_sum() *\
                                                    self.operation_pseudocounts[2] *\
                                                        self.sub_pseudocounts[bottom_idx][top_idx] * \
                                                            self.likelihoods[bottom_idx]
                    self.chart[i][j].sub_prob[top_idx] = to_add
                    # self.column_totals[i] = np.logaddexp(self.column_totals[i], to_add)

                    self.operation_sufficient_stats[2] += 1
                    self.sub_sufficient_stats[bottom_idx][top_idx] += 1
            # normalize each column when done
            for j2 in range(len(self.top_string)+1):
                pass
                # self.chart[i][j2]. = 



    def backward(self):
        pass

    def normalize(self):
        pass

    def forward_backward(self):
        pass
	"""docstring for Element"""
	def __init__(self, num_plus):
		super(Element, self).__init__()

		self.sub_prob = np.zeros((num_plus))
		self.ins_bot_prob = np.zeros((num_plus))
		self.ins_top_prob = 0
	def cell_sum(self):
		# sum up the whole cell 
		return sum([sum(self.sub_prob), sum(self.ins_bot_prob), self.ins_bot_prob])




class Column(object):
	"""each column corresponds to an observation time-step"""
	def __init__(self):
		super(Column, self).__init__()
		self.elements = []
		self.sub_sum = 0
		self.ins_top_sum = 0
		self.ins_bot_sum = 0
	def add_element(self, element):
		self.elements.append(element)
		self.sub_sum = np.logaddexp(self.sub_sum, element.sub_prob)




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
		self.chart = [[Element(self.num_plus)]*len(self.states)]*len(self.bottom_string)
		self.column_totals = np.zeros((len(bottom_string)))

		# initialize first one (doesn't matter)
		self.chart[0][0].ins_top_prob = 1
		# define dirichlet over all edits
		self.operation_pseudocounts = np.ones((3))/3
		self.operation_sufficient_stats = np.zeros((3))

		self.ins_bot_pseudocounts = np.ones((self.num_plus))/self.num_plus
		self.ins_bot_sufficient_stats = np.zeros((self.num_plus))

		self.sub_pseudocounts = np.ones((self.num_plus, self.num_top))
		self.sub_sufficient_stats = np.zeros((self.num_plus, self.num_top))

		print("initialized a matrix object with")
		print("chart:")
		print(self.chart)
		print("ins_bot_pseudocounts")
		print(self.ins_bot_pseudocounts.shape)
		print("sub_pseudocounts")
		print(self.sub_pseudocounts.shape)


	def forward(self):
		# need to make lookup by plu index, not by index in the string
		for i in range(len(self.bottom_string) + 1):
			for j in range(len(self.top_string) + 1):
				if abs(i-j) > self.max_edit:
					continue
				if i > 0:
					# insert bottom
					to_add = self.chart[i][j-1].cell_sum()*\
														self.operation_pseudocounts[1] *\
															self.ins_bot_pseudocounts[i] *\
																self.likelihoods[i]

					self.chart[i][j].ins_bot_prob = to_add
					# self.column_totals[i] = np.logaddexp(self.column_totals[i], to_add)

					self.operation_sufficient_stats[1] += 1
					self.ins_bot_sufficient_stats[i] += 1

				if j > 0:
					# insert top
					to_add = self.chart[i-1][j].cell_sum() *\
														 self.operation_pseudocounts[0]
					self.chart[i][j].ins_top_prob = to_add
					# self.column_totals[i] = np.logaddexp(self.column_totals[i], to_add)
					
					self.operation_sufficient_stats[0] += 1
				
				if i > 0 and j > 0:
					# substitution
					to_add = self.chart[i-1][j-1].cell_sum() *\
													self.operation_pseudocounts[2] *\
														self.sub_pseudocounts[i][j] * \
															self.likelihoods[i]
					self.chart[i][j].sub_prob = to_add
					# self.column_totals[i] = np.logaddexp(self.column_totals[i], to_add)

					self.operation_sufficient_stats[2] += 1
					self.sub_sufficient_stats[i][i] += 1
			# normalize each column when done
			for j2 in range(len(self.top_string)+1):
				pass
				# self.chart[i][j2]. = 



	def backward(self):
		pass

	def normalize(self):
		pass

	def forward_backward(self):
		pass


top_string = ['a','b','c']
bottom_string = ['a', 'b', 'b']
likelihoods = [.33, .33, .33]

m = Matrix(top_string,bottom_string, likelihoods, 2)
m.forward()

print(m)
        
		