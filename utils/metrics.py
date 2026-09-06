"""
Multi-modal benchmark evaluation utilities for MMBench, MMMU, and MMMU-Pro.

Evaluation logic and parsing routines are adapted from:
- MMMU: https://github.com/MMMU-Benchmark/MMMU
- VLMEvalKit / MMBench: https://github.com/open-compass/VLMEvalKit
"""

import ast
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

OPTION_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def parse_choice_response(
    response: str,
    all_choices: List[str] = OPTION_KEYS,
    index2ans: Optional[Dict[str, str]] = None,
) -> str:
    """Extract the option letter (e.g., A, B, C, ...) from the model's generated text."""
    if not response or not isinstance(response, str):
        return ""

    text = response.strip()
    choices_str = "".join(all_choices)

    # e.g. "A", "A.", "(A)", "[A]"
    direct_match = re.match(
        rf"^\s*[\(\[]?\s*([{choices_str}])\s*[\)\]]?[\.\:\s]*$", text, re.IGNORECASE
    )
    if direct_match:
        return direct_match.group(1).upper()

    # e.g. "the answer is (A)", "The correct option is: B"
    explicit_pattern = re.compile(
        rf"(?:the\s+)?(?:correct\s+)?(?:answer|choice|option)\s*(?:is|:|\*|\.|\s)*\s*\(?([{choices_str}])\)?",
        re.IGNORECASE,
    )
    matches = explicit_pattern.findall(text)
    if matches:
        return matches[-1].upper()

    # e.g. "(A)", "[B]" appearing anywhere in the text
    bracket_pattern = re.compile(rf"[\(\[]\s*([{choices_str}])\s*[\)\]]")
    bracket_matches = bracket_pattern.findall(text)
    if bracket_matches:
        return bracket_matches[-1].upper()

    # strip first-person phrases so "I choose A" isn't mistaken for option I
    sanitized_text = re.sub(
        r"\bI\s+(?:think|believe|choose|guess|assume|would|conclude|found)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    standalone_pattern = re.compile(rf"\b([{choices_str}])\b")
    standalone_matches = standalone_pattern.findall(sanitized_text)
    if standalone_matches:
        return standalone_matches[-1].upper()

    if index2ans:
        for opt, ans_text in sorted(
            index2ans.items(), key=lambda x: len(str(x[1])), reverse=True
        ):
            if ans_text and str(ans_text).strip().lower() in text.lower():
                return opt.upper()

    return ""


def parse_open_response(prediction: str) -> str:
    if not prediction:
        return ""
    text = prediction.strip()

    patterns = [
        r"(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is|:|\*|=)\s*(.+)$",
        r"(?:therefore|thus|hence),?\s*(.+)$",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            break
    return text.split("\n")[0].strip()


def eval_open_match(prediction: str, ground_truth: Any) -> float:
    """Evaluate open-ended response against ground truth via exact or numerical matching."""
    extracted_pred = parse_open_response(prediction)

    if isinstance(ground_truth, str) and ground_truth.strip().startswith("["):
        try:
            candidates = [str(x) for x in ast.literal_eval(ground_truth)]
        except Exception:
            candidates = [ground_truth.strip()]
    elif isinstance(ground_truth, (list, tuple)):
        candidates = [str(x) for x in ground_truth]
    else:
        candidates = [str(ground_truth).strip()]

    num_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    for cand in candidates:
        cand_str = cand.strip()
        pred_nums = num_pattern.findall(extracted_pred.replace(",", ""))
        gt_nums = num_pattern.findall(cand_str.replace(",", ""))

        if gt_nums and re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cand_str.replace(",", "")):
            if pred_nums:
                try:
                    val_gt = float(gt_nums[0])
                    val_pred = float(pred_nums[-1])
                    if math.isclose(val_gt, val_pred, rel_tol=1e-2, abs_tol=1e-3):
                        return 1.0
                except ValueError:
                    pass
            continue

        norm_cand = re.sub(r"[^a-z0-9]", "", cand_str.lower())
        norm_pred = re.sub(r"[^a-z0-9]", "", extracted_pred.lower())
        if norm_cand and norm_cand == norm_pred:
            return 1.0

        cand_escaped = re.escape(cand_str.lower())
        if cand_str and re.search(rf"\b{cand_escaped}\b", extracted_pred.lower()):
            return 1.0

    return 0.0


class MMBenchEvaluator:
    """Implements MMBench official CircularEval mechanism."""

    def __init__(self, option_keys: List[str] = OPTION_KEYS[:4]):
        self.option_keys = option_keys

    def score_sample(self, prediction: str, ground_truth: str) -> float:
        parsed = parse_choice_response(prediction, self.option_keys)
        return float(parsed == ground_truth.strip().upper())

    def score(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        """Compatible with BaseBenchmarkDataset interface."""
        return {"accuracy": self.score_sample(prediction, str(ground_truth))}

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate single-run accuracy (Acc) and circular consistency accuracy (Acc+)."""
        if not results:
            return {"accuracy": 0.0, "acc": 0.0, "acc_plus": 0.0, "num_questions": 0}

        grouped = defaultdict(list)
        for r in results:
            qid = r.get("question_id", r.get("index", r.get("id")))
            correct = self.score_sample(r["prediction"], r["ground_truth"])
            grouped[qid].append(correct)

        total_trials = sum(len(v) for v in grouped.values())
        acc = sum(sum(v) for v in grouped.values()) / total_trials if total_trials else 0.0

        # Acc+ counts a question correct only if every circular shift of it was answered correctly
        acc_plus = sum(1 for v in grouped.values() if all(c == 1.0 for c in v)) / len(grouped) if grouped else 0.0

        return {
            "accuracy": acc,
            "acc": acc,
            "acc_plus": acc_plus,
            "num_questions": len(grouped),
        }

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compatible with BaseBenchmarkDataset interface."""
        return self.evaluate(results)


class MMMUEvaluator:
    """Implements MMMU official micro/macro accuracy and open/closed grading."""

    def __init__(self, option_keys: List[str] = OPTION_KEYS[:4]):
        self.option_keys = option_keys
        self._choice_only = re.compile(rf"^[{''.join(option_keys)}]$")

    def is_mcq(self, ground_truth: Any) -> bool:
        return bool(self._choice_only.fullmatch(str(ground_truth).strip().upper()))

    def score_sample(self, prediction: str, ground_truth: Any) -> float:
        if self.is_mcq(ground_truth):
            parsed = parse_choice_response(prediction, self.option_keys)
            return float(parsed == str(ground_truth).strip().upper())
        return eval_open_match(prediction, ground_truth)

    def score(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        """Compatible with BaseBenchmarkDataset interface."""
        return {"accuracy": self.score_sample(prediction, ground_truth)}

    def evaluate(
        self, results: List[Dict[str, Any]], domain_key: str = "subject"
    ) -> Dict[str, Any]:
        """Calculate Micro-Accuracy and Macro-Accuracy (by domain/subject)."""
        if not results:
            return {"accuracy": 0.0, "overall_micro": 0.0, "overall_macro": 0.0, "by_domain": {}}

        grouped = defaultdict(list)
        for r in results:
            score = self.score_sample(r["prediction"], r["ground_truth"])
            dom = r.get(domain_key, "general")
            grouped[dom].append(score)

        by_domain = {dom: sum(scores) / len(scores) for dom, scores in grouped.items() if scores}
        micro_acc = sum(sum(scores) for scores in grouped.values()) / len(results) if results else 0.0
        macro_acc = sum(by_domain.values()) / len(by_domain) if by_domain else 0.0

        return {
            "accuracy": micro_acc,
            "overall_micro": micro_acc,
            "overall_macro": macro_acc,
            "by_domain": by_domain,
        }

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compatible with BaseBenchmarkDataset interface."""
        return self.evaluate(results)


class MMMUProEvaluator:
    """Implements MMMU-Pro 10-option (A-J) evaluator."""

    def __init__(self):
        self.option_keys = OPTION_KEYS

    def score_sample(
        self,
        prediction: str,
        ground_truth: str,
        index2ans: Optional[Dict[str, str]] = None,
    ) -> float:
        parsed = parse_choice_response(prediction, self.option_keys, index2ans)
        return float(parsed == ground_truth.strip().upper())

    def score(self, prediction: str, ground_truth: Any, **kwargs) -> Dict[str, float]:
        """Compatible with BaseBenchmarkDataset interface."""
        return {"accuracy": self.score_sample(prediction, str(ground_truth), kwargs.get("index2ans"))}

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {"accuracy": 0.0}
        scores = [
            self.score_sample(r["prediction"], r["ground_truth"], r.get("index2ans"))
            for r in results
        ]
        return {"accuracy": sum(scores) / len(scores)}

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compatible with BaseBenchmarkDataset interface."""
        return self.evaluate(results)


class Accuracy:
    """Accuracy metric shared across benchmarks.

    Supports both multiple-choice (letter) grading and free-text exact-match
    grading against one or more accepted answers.
    """

    def __init__(self, option_keys: List[str] = OPTION_KEYS):
        self.option_keys = option_keys
        self._choice_pattern = re.compile(rf"\b([{''.join(option_keys)}])\b")
        self._choice_only_pattern = re.compile(rf"^[{''.join(option_keys)}]$")

    def score(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        """Compute the accuracy of a single prediction against its ground truth"""
        if self.is_choice(ground_truth):
            predicted_letter = self.extract_choice(prediction)
            correct = float(predicted_letter == str(ground_truth).strip().upper())
        else:
            accepted_answers = self.parse_answers(ground_truth)
            normalized_prediction = self.normalize(prediction)
            correct = float(any(normalized_prediction == self.normalize(ans) for ans in accepted_answers))
            if correct == 0.0:
                correct = eval_open_match(prediction, ground_truth)
        return {"accuracy": correct}

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate per-sample accuracy scores into a final accuracy"""
        if not results:
            return {"accuracy": 0.0}
        total_accuracy = sum(r["metrics"]["accuracy"] for r in results)
        return {"accuracy": total_accuracy / len(results)}

    def is_choice(self, ground_truth: Any) -> bool:
        return bool(self._choice_only_pattern.fullmatch(str(ground_truth).strip().upper()))

    def extract_choice(self, prediction: str) -> str:
        return parse_choice_response(prediction, self.option_keys)

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @staticmethod
    def parse_answers(ground_truth: Any) -> List[str]:
        text = str(ground_truth).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        return [text]

