import torch
from abc import abstractmethod
from typing import Any, Dict, List, Union
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from base.base_model import BaseVLM


class HuggingFaceBaseVLM(BaseVLM):
    """Base class for Vision-Language Models implemented using HuggingFace Transformers."""

    default_dtype: torch.dtype = torch.float16
    clean_up_tokenization_spaces: bool = False

    def load_model(self):
        dtype = self.kwargs.get("dtype", self.kwargs.get("torch_dtype", self.default_dtype))
        device_map = self.kwargs.get("device_map", self.device)
        extra_kwargs = {
            k: v for k, v in self.kwargs.items()
            if k not in ["dtype", "torch_dtype", "device_map"]
        }

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            **extra_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif hasattr(self.processor, "padding_side"):
            self.processor.padding_side = "left"

    @abstractmethod
    def build_messages(self, image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
        """Construct chat template messages formatted for the specific VLM."""
        pass

    def process_inputs(self, text: Union[str, List[str]], image: Union[Image.Image, List[Image.Image]]):
        """Preprocess text and image into tensors on the model's device."""
        return self.processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

    def batch_generate(
        self,
        images: List[Union[Image.Image, str]],
        prompts: List[str],
        max_new_tokens: int = 16384,
        **gen_kwargs
    ) -> List[str]:
        if not images or not prompts:
            return []

        pil_images = [
            Image.open(img).convert("RGB") if isinstance(img, str) else img
            for img in images
        ]

        messages_list = [
            self.build_messages(img, prompt)
            for img, prompt in zip(pil_images, prompts)
        ]

        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_list
        ]

        inputs = self.process_inputs(texts, pil_images)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)

        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )

    def generate(self, image: Union[Image.Image, str], prompt: str, **gen_kwargs) -> str:
        return self.batch_generate([image], [prompt], **gen_kwargs)[0]

