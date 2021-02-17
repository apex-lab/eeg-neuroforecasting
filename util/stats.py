import numpy as np


def get_binomial_test(n, alternative = "greater"):
	'''
	Returns np.ufunc for one-tailed binomial test
	'''
	from scipy.stats import binom_test as bt

	def _binom_test(x):
		return bt(x, n, alternative = alternative)

	return np.frompyfunc(_binom_test, 1, 1)

def get_mann_whitney_u_test(n1, n2, alternative = "greater", dummy = False):
	'''
	returns np.unfunc that tests AUC against chance (0.5)

	n1, n2 are the number of examples in each class
	'''
	from scipy.stats import norm

	def _mann_whitney_u_test(auc, dummy_auc = None):

		U = n1 * n2 * auc # test statistic
		if dummy_auc:
			chance_U = n1 * n2 * dummy_auc
		else: # compare to chance
			chance_U = n1 * n2 * .5
		sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
		z = (U - chance_U) / sigma # z-score under null distribution

		if alternative == "two-sided":
			p = 2 * norm.sf(abs(z))
		elif alternative == "greater":
			p = norm.sf(z)
		elif alternative == "less":
			p = norm.cdf(z)
		else:
			raise ValueError("alternative should be default 'greater', 'less'"
                         " or 'two-sided'")
		return p

	if dummy: # include dummy AUC vec as input
		return np.frompyfunc(_mann_whitney_u_test, 2, 1)
	else: # no dummy input needed
		return np.frompyfunc(_mann_whitney_u_test, 1, 1)

