"""GET /modes and /strategies — list endpoints that power the UI selectors."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from ..schemas import ModeInfo, ModesResponse, StrategiesResponse, StrategyInfo
from ...group_reranker import GROUP_STRATEGIES
from ...reranking import MODE_WEIGHTS

router = APIRouter(tags=["meta"])


MODE_DESCRIPTIONS = {
    "balanced": "Mix of popular and lesser-known films aligned to your taste. Default.",
    "niche": "Hidden gems that fit your taste. Strong popularity penalty, but quality floor stays high.",
    "popular": "Canon and crowd-pleasers, ranked by what users with similar taste rate highly.",
    "serendipitous": "Surprising picks that still sit adjacent to your taste — wider net, higher diversity.",
}


STRATEGY_DESCRIPTIONS = {
    "average": "Mean of per-member scores. Default group strategy.",
    "least_misery": "Maximize the worst member's score — nobody hates the pick.",
    "most_pleasure": "Maximize any single member's enthusiasm — one person will love it.",
    "consensus": "Mean minus variance — rewards uniform agreement.",
    "hybrid": "Mean plus a bonus for the worst score — broadly liked but not punishingly safe.",
    "group_taste_vector": "Fuses everyone's taste into one super-user and re-ranks against that signal. Finds movies the GROUP would love even if no individual would have picked it.",
}


@router.get("/modes", response_model=ModesResponse)
def list_modes() -> ModesResponse:
    modes = []
    for name, weights in MODE_WEIGHTS.items():
        modes.append(ModeInfo(
            name=name,
            description=MODE_DESCRIPTIONS.get(name, ""),
            weights={k: float(v) for k, v in asdict(weights).items()},
        ))
    return ModesResponse(modes=modes)


@router.get("/strategies", response_model=StrategiesResponse)
def list_strategies() -> StrategiesResponse:
    return StrategiesResponse(strategies=[
        StrategyInfo(name=s, description=STRATEGY_DESCRIPTIONS.get(s, ""))
        for s in GROUP_STRATEGIES
    ])
