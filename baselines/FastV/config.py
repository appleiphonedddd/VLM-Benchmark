from dataclasses import dataclass

@dataclass
class FastVConfig:

    k: int
    r: float
    image_token_index: int = 32000

    def __post_init__(self):
        if not (0.0 <= self.r < 1.0):
            raise ValueError(f"Pruning ratio r must be in [0.0, 1.0), got {self.r}")
        if self.k < 0:
            raise ValueError(f"Pruning layer k must be non-negative, got {self.k}")