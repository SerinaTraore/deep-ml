def conditional_probability(joint_distribution: dict) -> float:
	"""
	Compute conditional probability P(A|B) from a joint probability distribution.

	Args:
		joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

	Returns:
		float: Conditional probability P(A|B)
	"""
	p_a_and_b = joint_distribution.get(("A", "B"), 0.0)

	p_b = p_a_and_b + joint_distribution.get(("`A", "B"), 0.0)
	if p_b == 0.0:
		return 0.0
	return p_a_and_b / p_b