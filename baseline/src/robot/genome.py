import copy
from dataclasses import dataclass, astuple
import os
from random import Random
from typing import Optional
from abrain import Genome as ANNGenome, plotly_render
from abrain.core.genome import GIDManager

@dataclass
class VisionData:
    w: int = 0
    h: int = 0
    def __iter__(self):
        return iter(astuple(self))


class RVGenome:
    __private_key = object()

    def __init__(self, brain: ANNGenome):
        self.brain = brain
        self.vision = VisionData()

    def id(self):
        return self.brain.id()

    def parents(self):
        return self.brain.parents()
    
    def set_vision(self, w, h):
        self.vision.w=w
        self.vision.h=h

    def get_vision(self):
        return self.vision.w, self.vision.h

    def __repr__(self):
        return f"{str(self.brain)}"


    def mutate(self, rng: Random) -> None:
        self.brain.mutate(rng)

    def mutated(self, rng: Random, id_manager: GIDManager):
        clone = self.copy()
        clone.mutate(rng)
        clone.brain.update_lineage(id_manager, [self.brain])
        return clone

    @staticmethod
    def random(rng: Random, id_manager: GIDManager) -> 'RVGenome':
        return RVGenome(ANNGenome.random(rng, id_manager))

    def copy(self) -> 'RVGenome':
        return RVGenome(
            self.brain.copy()
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, _):
        return self.copy()

    def __getstate__(self):
        return dict(brain=self.brain, vision=self.vision)

    def __setstate__(self, state):
        self.__dict__ = state
        assert isinstance(self.brain, ANNGenome)
        if self.vision is not None:
            assert isinstance(self.vision, VisionData)

    def to_json(self):
        return self.brain.to_json()
        
    
    @staticmethod
    def from_json(data) -> 'RVGenome':
        """Recreate a RVGenome from string json representation
        """
        return RVGenome(ANNGenome.from_json(data))


    @staticmethod
    def from_dot(path: str, rng: Random):
        """Does not make sense for embedded abrain genome"""
        raise RuntimeError
    
    
