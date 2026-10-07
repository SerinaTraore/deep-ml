import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	magnitude = np.linalg.norm(gradient)
	if magnitude > 0:
		direction = [ x / magnitude for x in gradient]
		descent_direction = [-x for x in direction]
		return {"magnitude": magnitude, "direction": direction,
		"descent_direction": descent_direction}
	return {"magnitude": 0.0, "direction": [0.0] * len(gradient),
		"descent_direction": [0.0] * len(gradient)}
	# Your code here
	pass