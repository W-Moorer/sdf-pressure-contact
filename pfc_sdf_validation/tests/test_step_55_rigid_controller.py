from __future__ import annotations

import numpy as np

from pfcsdf.contact.active_set import ActiveSetSnapshot, active_set_mismatch_report
from pfcsdf.contact.wrench import PairWrench
from pfcsdf.dynamics.rigid_controller import (RigidEventAwareControllerConfig, rigid_controller_indicators, should_substep_rigid)
from pfcsdf.dynamics.rigid_state import RigidBodyState


def _state(rot=None):
    if rot is None:
        rot = np.eye(3)
    return RigidBodyState(
        position=np.zeros(3),
        rotation=rot,
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        mass=1.0,
        inertia_body=np.diag([1.0,1.2,1.4]),
    )


def test_rigid_controller_uses_torque_and_orientation_mismatch():
    pred = ActiveSetSnapshot(mask=np.array([[[True]], [[False]]]), measure=1.0)
    corr = ActiveSetSnapshot(mask=np.array([[[False]], [[True]]]), measure=1.0)
    mismatch = active_set_mismatch_report(pred, corr)
    rot = np.array([[0.0, -1.0, 0.0],[1.0, 0.0, 0.0],[0.0,0.0,1.0]])
    ind = rigid_controller_indicators(
        PairWrench(np.array([0.0,0.0,1.0]), np.array([0.0,0.0,0.0])),
        PairWrench(np.array([0.0,0.0,1.0]), np.array([1.0,0.0,0.0])),
        PairWrench(np.array([0.0,0.0,1.0]), np.array([0.0,1.0,0.0])),
        start_measure=1.0, predictor_measure=1.0, corrector_measure=1.0,
        mismatch=mismatch, predictor_state=_state(), corrector_state=_state(rot),
    )
    assert ind.torque_jump > 0.1
    assert ind.orientation_mismatch > 0.1
    assert should_substep_rigid(RigidEventAwareControllerConfig(), 0.01, ind)
