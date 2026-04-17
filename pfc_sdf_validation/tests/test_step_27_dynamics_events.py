from __future__ import annotations

from pfcsdf.dynamics.events import ballistic_gap_prediction, need_substep_for_event, predict_contact_event
from pfcsdf.dynamics.state import NormalDynamicsState


def test_ballistic_prediction_detects_onset() -> None:
    state = NormalDynamicsState(time=0.0, gap=0.05, velocity=-1.0, mass=1.0)
    a0 = 0.0
    gap_end = ballistic_gap_prediction(state, 0.1, a0)
    assert gap_end < 0.0

    pred = predict_contact_event(state, 0.1, a0)
    assert pred.onset is True
    assert pred.release is False
    assert need_substep_for_event(state, 0.1, a0) is True


def test_ballistic_prediction_detects_release() -> None:
    state = NormalDynamicsState(time=0.0, gap=-0.02, velocity=0.5, mass=1.0)
    pred = predict_contact_event(state, 0.1, 0.0)
    assert pred.onset is False
    assert pred.release is True
    assert need_substep_for_event(state, 0.1, 0.0) is True
