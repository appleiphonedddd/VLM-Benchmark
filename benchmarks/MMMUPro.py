import ast
from typing import Dict, Any, List
from datasets import load_dataset
from base.base_dataset import BaseBenchmarkDataset
from utils.metrics import OPTION_KEYS

IMAGE_KEYS = [f"image_{i}" for i in range(1, 8)]


class MMMUProDataset(BaseBenchmarkDataset):
    """MMMU-Pro robust multi-discipline multimodal understanding benchmark"""

    def __init__(self, data_path: str = "MMMU/MMMU_Pro", split: str = "test", config_name: str = "standard (10 options)"):
        self.config_name = config_name
        super().__init__(data_path, split)

    def load_data(self) -> List[Dict[str, Any]]:
        hf_dataset = load_dataset(self.data_path, self.config_name, split=self.split)

        samples = []
        for item in hf_dataset:
            # MMMU-Pro questions may reference several images as <image 1>, <image 2>, ...
            
            image = next((item[key] for key in IMAGE_KEYS if item.get(key) is not None), None)

            options = ast.literal_eval(item["options"])
            option_text = "\n".join(f"{key}. {value}" for key, value in zip(OPTION_KEYS, options))
            prompt = "\n".join([
                item["question"],
                option_text,
                "Think step by step, then end your response with a new line "
                "in the exact form 'Answer: <letter>'.",
            ])

            samples.append({
                "id": item.get("id"),
                "image": image,
                "prompt": prompt,
                "ground_truth": item["answer"],
                "subject": item.get("subject"),
                "topic_difficulty": item.get("topic_difficulty"),
            })

        return samples
