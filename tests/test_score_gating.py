from src.rag.score_gating import should_refuse


def test_refuse_when_no_candidates():
    assert should_refuse([])[0] is True


def test_refuse_when_max_score_too_low():
    refused, reason = should_refuse([0.3, 0.2, 0.1], max_threshold=0.5)
    assert refused is True
    assert "max_score_too_low" in reason


def test_refuse_when_score_spread_too_small():
    refused, reason = should_refuse([0.8, 0.8, 0.8], max_threshold=0.5, spread_threshold=0.05)
    assert refused is True
    assert "score_spread_too_small" in reason


def test_pass_when_scores_are_confident_and_separated():
    refused, reason = should_refuse([0.9, 0.6, 0.5], max_threshold=0.5, spread_threshold=0.05)
    assert refused is False
    assert reason == "ok"
