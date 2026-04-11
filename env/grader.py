def grade(action, truth):
    """
    Assign a score between 0.0 and 1.0 based on the agent's action.
    
    Parameters:
        action: Action object from agent
        truth: dict containing correct email type and expected priority
    
    Returns:
        score: float between 0.0 and 1.0
    """
    score = 0.0

    # Classification
    if action.category == truth["type"]:
        score += 0.5

    # Priority
    if truth["type"] == "important" and action.priority == "high":
        score += 0.3

    # Response (basic check: length > 5)
    if len(action.response.strip()) > 5:
        score += 0.2

    return score