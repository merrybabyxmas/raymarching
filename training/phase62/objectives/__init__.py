from training.phase62.objectives.base import VolumeObjective, VolumeOutputs
from training.phase62.objectives.indep_bce import IndependentBCEObjective
from training.phase62.objectives.factorized_fg_id import FactorizedFgIdObjective
from training.phase62.objectives.projected_visible import ProjectedVisibleObjective
from training.phase62.objectives.projected_amodal import ProjectedAmodalObjective
from training.phase62.objectives.center_offset import CenterOffsetObjective

OBJECTIVE_REGISTRY = {
    "independent_bce": IndependentBCEObjective,
    "factorized_fg_id": FactorizedFgIdObjective,
    "projected_visible_only": ProjectedVisibleObjective,
    "projected_amodal_only": ProjectedAmodalObjective,
    "center_offset": CenterOffsetObjective,
}


def build_objective(name: str, **kwargs) -> VolumeObjective:
    if name not in OBJECTIVE_REGISTRY:
        raise ValueError(f"Unknown objective: {name}. Available: {list(OBJECTIVE_REGISTRY.keys())}")
    return OBJECTIVE_REGISTRY[name](**kwargs)
