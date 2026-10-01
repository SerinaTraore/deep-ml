import numpy as np
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	v = np.array(v)
	L = np.array(L)
	v
	L_norm = np.sqrt(np.sum(L * L)) ** 2
	v_scalaire_L = np.sum(v * L)
	return ((v_scalaire_L * L )/ L_norm)

	pass
