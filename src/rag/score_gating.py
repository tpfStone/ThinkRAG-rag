def should_refuse(
    scores: list[float],
    top_k: int = 5,
    max_threshold: float = 0.5,
    spread_threshold: float = 0.05,
    **_,
) -> tuple[bool, str]:
    valid_scores = [float(score) for score in scores if score is not None]
    if not valid_scores:
        return True, "no_candidates"

    top_scores = valid_scores[:top_k]
    max_score = max(top_scores)
    mean_score = sum(top_scores) / len(top_scores)
    spread = max_score - mean_score

    if max_score < max_threshold:
        return True, f"max_score_too_low ({max_score:.3f} < {max_threshold})"
    if spread < spread_threshold:
        return True, f"score_spread_too_small ({spread:.3f} < {spread_threshold})"
    return False, "ok"
