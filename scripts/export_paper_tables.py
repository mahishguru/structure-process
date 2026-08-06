"""Flatten the cross-pipeline summaries into tidy CSVs and paper-ready markdown.

Every number quoted in results/paper_results.md must come from here, so tables
and figures can be rebuilt without opening a JSON by hand.
"""
import argparse
import csv
import json
from pathlib import Path

PIPELINES = ("conventional", "genai", "gnn")
COMPOSITION_HEADS = (
    ("knn", "kNN retrieval"),
    ("xgb_classifier", "XGBoost classifier"),
    ("catboost_classifier", "CatBoost classifier"),
    ("ft_transformer", "FT-Transformer"),
    ("catboost_knn_fusion", "CatBoost + kNN fusion"),
    ("balanced_fusion", "Condition-balanced fusion"),
    ("prior_adjusted_balanced_fusion", "Prior-adjusted balanced fusion"),
    ("xgb_regression", "XGBoost regression"),
    ("catboost_regression", "CatBoost regression"),
)
PROCESS_HEADS = (
    ("knn", "kNN retrieval"),
    ("xgb", "XGBoost"),
    ("catboost", "CatBoost"),
    ("ft_transformer", "FT-Transformer"),
    ("gaussian_process", "Gaussian process"),
)
COMPOSITION_FIELDS = (
    "element_wape_present", "element_mae_macro",
    "alloy_top1", "alloy_top3", "class_nll",
)
PROCESS_FIELDS = ("mae", "mape", "wape", "r2", "nll", "coverage90")


def load(split: str) -> dict:
    return {
        pipeline: json.loads(Path(
            f"runs/summary_cross_pipeline_heads_{pipeline}_{split}.json"
        ).read_text())
        for pipeline in PIPELINES
    }


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def export(summaries: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    composition, per_element, process = [], [], []
    for pipeline, summary in summaries.items():
        for head, label in COMPOSITION_HEADS:
            block = summary["composition_heads"][head]
            composition.append({
                "pipeline": pipeline, "head": head, "label": label,
                **{key: block.get(key) for key in COMPOSITION_FIELDS},
            })
            for element, scores in block["per_element"].items():
                per_element.append({
                    "pipeline": pipeline, "head": head, "element": element,
                    "mae": scores["mae"],
                    "wape_present": scores["wape_present"],
                })
        for head, label in PROCESS_HEADS:
            block = summary["process_heads"][head]
            for target in ("T_ext", "v_ext"):
                process.append({
                    "pipeline": pipeline, "head": head, "label": label,
                    "target": target,
                    **{key: block[target].get(key) for key in PROCESS_FIELDS},
                })

    write_csv(out_dir / "composition_heads.csv",
              ["pipeline", "head", "label", *COMPOSITION_FIELDS], composition)
    write_csv(out_dir / "composition_per_element.csv",
              ["pipeline", "head", "element", "mae", "wape_present"],
              per_element)
    write_csv(out_dir / "process_heads.csv",
              ["pipeline", "head", "label", "target", *PROCESS_FIELDS], process)


def markdown(summaries: dict) -> str:
    lines = ["### Table 3a", "",
             "| head | conv WAPE | conv MAE | conv top-1 | genai WAPE | "
             "genai MAE | genai top-1 | gnn WAPE | gnn MAE | gnn top-1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for head, label in COMPOSITION_HEADS:
        cells = []
        for pipeline in PIPELINES:
            block = summaries[pipeline]["composition_heads"][head]
            top1 = block.get("alloy_top1")
            cells += [
                f"{100 * block['element_wape_present']:.1f}%",
                f"{block['element_mae_macro']:.3f}",
                "-" if top1 is None else f"{top1:.3f}",
            ]
        lines.append("| " + " | ".join([label, *cells]) + " |")

    lines += ["", "### Table 3b", "",
              "| head | conv T MAE | conv T MAPE | conv T R2 | conv v MAE | "
              "conv v MAPE | conv v R2 | genai T MAE | genai T MAPE | "
              "genai T R2 | genai v MAE | genai v MAPE | genai v R2 | "
              "gnn T MAE | gnn T MAPE | gnn T R2 | gnn v MAE | gnn v MAPE | "
              "gnn v R2 |",
              "|" + "---|" * 19]
    for head, label in PROCESS_HEADS:
        cells = []
        for pipeline in PIPELINES:
            block = summaries[pipeline]["process_heads"][head]
            for target, digits in (("T_ext", 1), ("v_ext", 2)):
                scores = block[target]
                cells += [
                    f"{scores['mae']:.{digits}f}",
                    f"{100 * scores['mape']:.1f}%",
                    f"{scores['r2']:.3f}",
                ]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="random_seed0")
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args()
    summaries = load(args.split)
    export(summaries, Path(args.out_dir))
    print()
    print(markdown(summaries))


if __name__ == "__main__":
    main()
