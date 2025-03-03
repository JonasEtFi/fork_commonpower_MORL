from enum import Enum, auto

class CEnum(Enum):
    def __str__(self):
        return self.name


class Stage(CEnum):
    Train = auto()
    Deploy = auto()


class Approach(CEnum):
    WithProjectionSafeguard = auto()
    WithReplacementSafeguard = auto()
    OptimalController = auto()


class Penalty(CEnum):
    NoPenalty = auto()
    ConstantPenalty = auto()
    DDPenalty = auto()
    BothPenalties = auto()