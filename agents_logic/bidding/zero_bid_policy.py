from typing import Dict, Union

from lux.kit import GameState

def zero_bid_policy(agent: str, obs: Dict[str, GameState]) -> Dict[str, Union[int, str]]:
    faction = "AlphaStrike"
    if agent == "player_1":
        faction = "MotherMars"
    return dict(bid=0, faction=faction)
