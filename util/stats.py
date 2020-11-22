from scipy.stats  import binom_test as bt
import numpy as np

def _binom_test(x, n):
	return bt()

def get_binomial_test(n):
	'''
	Returns np.ufunc for one-tailed binomial test
	'''

	def _binom_test(x):
		return bt(x, n, alternative = "greater")

	return np.frompyfunc(_binom_test, 1, 1)