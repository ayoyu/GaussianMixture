import numpy as np 
from numpy.linalg import det, solve


class Frozen(object):
	"""
	frozen object
	"""
	_frozen = False
	def __setattr__(self, attr, value):
		if not hasattr(self, attr) and self._frozen:
			raise TypeError(f"{self} is frozen object can't set a new attribute")
		object.__setattr__(self, attr, value)

	def frozen(self):
		self._frozen = True


class Gaussian(Frozen):
	"""
	attributes:
		mean_vect: shape (d,)
		cov_matrix: shape (d, d)
		pi : scalar
		with d: nbr of features && c: nbr of Gaussians

	methods:
		pdf: multivariate gaussian density function
	"""
	def __init__(self, mean_vect, cov_matrix, pi):
		self.mean_vect = mean_vect
		self.cov_matrix = cov_matrix
		self.pi = pi
		self.frozen()


	def __repr__(self):
		return f'Gaussian(pi={self.pi})'


	def pdf(self, x):
		det_cov = det(self.cov_matrix)
		if det_cov < 0:
			raise RuntimeError("the cov_matrix need to be positive definite")
		d = self.mean_vect.shape[0]
		diff = x - self.mean_vect
		# to calculate sigma^-1 * (x - u): solution of the equation sigma * y = (x - u)
		sol = solve(self.cov_matrix, diff.T)
		return 1 / (2 * np.pi)**(d/2) * det_cov**(-1/2) * np.exp(-1/2 * np.sum(diff * sol.T, axis=1))


class GaussianMixture:
	"""
	attributes:
		n_components: nbr of Gaussians
		max_iter: The number of EM iterations to perform.
		tol: The convergence threshold. EM iterations will stop 
			 when the lower bound average gain is below this threshold.
		n_init: The number of initializations to perform. The best results are kept

	methods:
	"""
	def __init__(self, nbr_features, n_components=1, max_iter=100,
		tol=0.001, n_init=1, verbose=True):

		self.nbr_features = nbr_features
		self.n_components = n_components
		self.max_iter = max_iter
		self.tol = tol
		self.n_init = n_init
		self.verbose = verbose
		self.components = None
		

	def __repr__(self):
		return f"GaussiansMixture(C = {self.n_components})"


	def params_initializer(self):
		components = []
		sum_pi = 0
		for c in range(self.n_components):
			mean = np.random.uniform(low=-1.0, high=10.0, size=(self.nbr_features,))
			pi_c = np.random.uniform()
			sigma = sigma = np.eye(self.nbr_features) * np.random.uniform(2., 10.)
			sum_pi += pi_c 
			gauss = Gaussian(mean, sigma, pi_c)
			components.append(gauss)
		for g in components:
			g.pi /= sum_pi
		return components


	def get_loss(self, X, gamma):
		pi = np.array([gauss.pi for gauss in self.components])
		likelihood = np.array([np.sum(gauss.pdf(X)) for gauss in self.components])
		gamma_c = np.sum(gamma, axis=0)
		loss = 0
		return np.sum(np.log((pi * likelihood) / gamma_c + 1e-20) * (gamma_c + 1e-20))


	def E_step(self, X):
		gamma = np.zeros((X.shape[0], self.n_components))
		for c in range(self.n_components):
			gauss = self.components[c]
			gamma[:, c] = gauss.pi * gauss.pdf(X)
		gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
		return gamma


	def M_step(self, X, gamma):
		m = X.shape[0]
		for c in range(self.n_components):
			m_c = np.sum(gamma[:, c], axis=0)
			p_c = m_c / m
			mean = 1 / m_c * np.sum(X * gamma[:, c][:, np.newaxis], axis=0)
			cov = 1 / m_c * np.dot(((X - mean) * gamma[:, c][:, np.newaxis]).T, (X - mean))
			gauss = self.components[c]
			gauss.mean_vect, gauss.cov_matrix, gauss.pi = mean, cov, p_c
		return 


	def fit(self, X):
		best_loss = None
		best_Mixture = None
		for _ in range(self.n_init):
			try:
				self.components = self.params_initializer()
				loss = None
				for j in range(self.max_iter):
					gamma = self.E_step(X)
					self.M_step(X, gamma)
					current_loss = self.get_loss(X, gamma)
					if loss and np.abs(current_loss/loss - 1) < self.tol:
						loss = current_loss
						if self.verbose:
							print('-'*40, '\n', 'Stabalized: ', j, '| loss:', loss)
						break
					loss = current_loss
				if self.verbose:
					print(self.components)
				if best_loss == None or loss < best_loss:
					best_loss = loss
					best_Mixture = self.components
			except np.linalg.LinAlgError:
				print('Bad initialization Singular matrix sigma can not be inversed')
				pass
		self.components = best_Mixture
		return best_loss


	def predict_proba(self, X):
		gamma = self.E_step(X)
		return gamma