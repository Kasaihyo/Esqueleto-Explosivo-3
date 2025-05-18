from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict

# Defines the Symbol data structure or class
# Will hold symbol properties like name, type (HP, LP, W, EW, S), payout values etc.


class SymbolType(Enum):
    HP = auto()  # High Pay
    LP = auto()  # Low Pay
    WILD = auto()  # Standard Wild
    EXPLOSIVO_WILD = auto()  # Explosivo Wild (EW)
    SCATTER = auto()  # Scatter/Bonus Symbol
    EMPTY = auto()  # Represents an empty space after explosion


@dataclass(frozen=True)  # Make symbols immutable
class Symbol:
    name: str
    type: SymbolType
    # Map display text (frontend rendering)
    t: str = field(init=False)

    def __post_init__(self):
        # Default display text: strip underscores and non-alphanumeric characters, uppercase
        sanitized = "".join(ch for ch in self.name if ch.isalnum()).upper()
        object.__setattr__(self, "t", sanitized)

    def __repr__(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Symbol to a dictionary suitable for JSON events."""
        return {
            "k": self.name,
            "t": self.t,
            "type": self.type.name,  # Use the enum member name (e.g., 'LP', 'WILD')
        }
