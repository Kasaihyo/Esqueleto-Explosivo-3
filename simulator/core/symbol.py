from dataclasses import dataclass, field
from enum import Enum, auto

# Defines the Symbol data structure or class
# Will hold symbol properties like name, type (HP, LP, W, EW, S), payout values etc.

class SymbolType(Enum):
    HP = auto()  # High Pay
    LP = auto()  # Low Pay
    WILD = auto() # Standard Wild
    EXPLOSIVO_WILD = auto() # Explosivo Wild (EW)
    SCATTER = auto() # Scatter/Bonus Symbol
    EMPTY = auto() # Represents an empty space after explosion

@dataclass(frozen=True) # Make symbols immutable
class Symbol:
    name: str
    type: SymbolType
    # Payouts might be stored separately in config based on symbol name and cluster size
    # Add other relevant properties as needed, e.g., color, specific behaviour flags

    def __repr__(self) -> str:
        return self.name
