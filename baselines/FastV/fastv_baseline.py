from typing import List, Union
from PIL import Image
from base.base_baseline import BaseBaseline
from .config import FastVConfig
from .patcher import FastVPatcher

class FastVBaseline(BaseBaseline):
    def __init__(self, model, k: int = 2, r: float = 0.50):
        super().__init__(model)
        self.config = FastVConfig(k=k, r=r)
        self.patcher = FastVPatcher(self.config)
        self.patcher.patch_model(self.model.model)

    def batch_generate(self, images: List[Union[Image.Image, str]], prompts: List[str], **gen_kwargs) -> List[str]:
        # FastVPatcher operates on single-sample prefill tensors due to dynamic token pruning masks.
        # Fall back to iterative generation to preserve token ranking integrity.
        return [self.generate(img, p, **gen_kwargs) for img, p in zip(images, prompts)]