import os
import json
import argparse
import models
import benchmarks
import baselines
from utils import CaseInsensitiveChoice
from tqdm import tqdm

def load_model(args):
    model = models.build_model(args.model, model_path=args.model_path, device=args.device)

    if args.baseline is not None:
        model = baselines.build_baseline(args.baseline, model=model)

    return model

def load_benchmark(args):
    kwargs = {}
    if args.data_path:
        kwargs["data_path"] = args.data_path
    if args.split:
        kwargs["split"] = args.split
    if args.config_name:
        kwargs["config_name"] = args.config_name

    return benchmarks.build_dataset(args.benchmark, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Benchmark Evaluation")

    parser.add_argument("--model", type=CaseInsensitiveChoice(models.MODEL_REGISTRY),
                        required=True, metavar="|".join(sorted(models.MODEL_REGISTRY)),
                        help="Model architecture to evaluate")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HuggingFace ID for the model checkpoint")

    parser.add_argument("--benchmark", type=CaseInsensitiveChoice(benchmarks.DATASET_REGISTRY),
                        required=True, metavar="|".join(sorted(benchmarks.DATASET_REGISTRY)),
                        help="Benchmark dataset to run")

    parser.add_argument("--data_path", type=str, default=None,
                        help="HuggingFace dataset ID or local dataset directory "
                             "(defaults to the benchmark's own dataset ID)")

    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split (defaults to the benchmark's own split)")

    parser.add_argument("--config_name", type=str, default=None,
                        help="Dataset config/subset name (defaults to the benchmark's own config)")

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Cache directory for downloaded datasets (sets HF_DATASETS_CACHE)")

    parser.add_argument("--baseline", type=CaseInsensitiveChoice(baselines.BASELINE_REGISTRY),
                        default=None, metavar="|".join(sorted(baselines.BASELINE_REGISTRY)),
                        help="Baseline or acceleration technique to apply")

    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save evaluation results")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for model inference (default: 1)")

    parser.add_argument("--max_new_tokens", type=int, default=8192,
                        help="Max new tokens to generate per sample")

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def format_sample(result, index):
    """Render one finished sample as a single console line"""
    accuracy = result["metrics"].get("accuracy")
    mark = "-" if accuracy is None else ("✓" if accuracy >= 1.0 else "✗")
    prediction = " ".join(str(result["prediction"]).split())
    if len(prediction) > 80:
        prediction = prediction[:77] + "..."
    return (f"[{index:>5}] {mark} id={result['id']} "
            f"gt={result['ground_truth']} pred={prediction!r}")


def make_progress_reporter():
    """Build the per-sample callback passed to run_evaluation"""
    seen = 0

    def report(result, running):
        nonlocal seen
        seen += 1
        tqdm.write(format_sample(result, seen))

    return report


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.data_dir:
        os.makedirs(args.data_dir, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = args.data_dir

    print(f"Loading model '{args.model}'...")
    model = load_model(args)

    print(f"Loading benchmark '{args.benchmark}'...")
    dataset = load_benchmark(args)

    print("Starting evaluation...")
    evaluation = dataset.run_evaluation(
        model, on_sample=make_progress_reporter(), batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )

    details = evaluation["details"]
    correct = sum(1 for r in details if r["metrics"].get("accuracy", 0) >= 1.0)
    print(f"Correct: {correct}/{len(details)}")
    print(f"Evaluation Metrics: {evaluation['summary']}")

    # Save results and configuration
    output_filename = f"{args.model}_{args.benchmark}" + (f"_{args.baseline}" if args.baseline else "")
    out_file = os.path.join(args.output_dir, f"{output_filename}.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "metrics": evaluation["summary"],
            "results": evaluation["details"]
        }, f, indent=4)

    print(f"Results successfully saved to {out_file}")


if __name__ == "__main__":
    main()
