from ipyparallel import Client

import numpy as np
import os



def get_pid(x): 
	import os
	return os.getpid()

def waste_time(x):
	for i in range(100, 1000):
		for j in range(100, 1000):
			for k in range(100, 1000):
				r = i**j * k
	import os
	return os.getpid()

print("starting")
rc = Client(profile="default")
dview = rc[:]
print("got client")

res = dview.map_sync(waste_time, [10]*4)
print(list(res))