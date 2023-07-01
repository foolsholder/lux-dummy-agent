from abc import ABC, abstractmethod

from lux.stats import *
from lux.kit import *


class PhaseRewarder(ABC):

    @abstractmethod
    def estimate_reward(
        self,
        stats: Dict[str, StatsStateDict],
        obs: Dict[str, GameState], 
        done: Dict[str, bool]
    ) -> Dict[str, float]:
        pass
