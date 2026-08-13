#!/usr/bin/env python
"""Wait for the seed-0 chain, then re-export tables and refresh paper_results.md.

Run after scripts/run_cross_pipeline_random_seed0.sh completes. It regenerates
results/tables/*.csv and updates the numeric tokens in results/paper_results.md
that are derived from the seed-0 summaries, so the results doc tracks the
current (7-element) run without hand-editing.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PIPELINES = ("conventional", "genai", "gnn")


def wait_for_chain(timeout_s: int = 12 * 3600) -> None:
    import time
    deadline = time.time() + timeout_s
    expected = [RUNS / f"summary_cross_pipeline_heads_{p}_random_seed0.json"
                for p in PIPELINES]
    while time.time() < deadline:
        if all(p.exists() for p in expected):
            return
        time.sleep(30)
    raise TimeoutError("seed-0 summaries did not all appear in time")


def recompute_check() -> None:
    out = ROOT / "results" / "_recomputed.json"
    subprocess.run([sys.executable, "scripts/metrics_from_predictions.py",
                    "--out", str(out)], cwd=ROOT, check=True)
    rec = json.loads(out.read_text())
    n = bad = 0
    for pipe in rec:
        ref = json.loads(
            (RUNS / f"summary_cross_pipeline_heads_{pipe}_random_seed0.json")
            .read_text())

        def walk(a, b):
            nonlocal n, bad
            if isinstance(a, dict):
                for k in a:
                    walk(a[k], b[k])
            elif isinstance(a, (int, float)):
                n += 1
                bad += abs(float(a) - float(b)) > 1e-9

        for group in ("composition_heads", "process_heads"):
            walk(rec[pipe][group], ref[group])
    print(f"[assimilate] store recompute: {n} scalars, {bad} mismatches")
    if bad:
        raise SystemExit("prediction store does not match summaries")


def refresh_calibration_line(md: str) -> str:
    gp = {p: json.loads(
        (RUNS / f"summary_cross_pipeline_heads_{p}_random_seed0.json")
        .read_text())["process_heads"]["gaussian_process"] for p in PIPELINES}
    tc, tg, tn = (gp[p]["T_ext"]["coverage90"] for p in PIPELINES)
    vc, vg, vn = (gp[p]["v_ext"]["coverage90"] for p in PIPELINES)
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("GP 90% coverage (target 0.90): T_ext "):
            lines[i] = (f"GP 90% coverage (target 0.90): T_ext {tc:.3f} "
                        f"conventional / {tg:.3f} genai /")
            lines[i + 1] = (f"{tn:.3f} gnn; v_ext {vc:.3f} conventional / "
                            f"{vg:.3f} genai / {vn:.3f} gnn.")
            break
    return "\n".join(lines) + "\n"


def main() -> None:
    print("[assimilate] waiting for seed-0 chain", flush=True)
    wait_for_chain()
    print("[assimilate] chain done; re-exporting tables and updating "
          "paper_results.md", flush=True)
    subprocess.run([sys.executable, "scripts/export_paper_tables.py",
                    "--update-results"], cwd=ROOT, check=True)
    recompute_check()
    md_path = ROOT / "results" / "paper_results.md"
    md = refresh_calibration_line(md_path.read_text())
    md_path.write_text(md)
    print("[assimilate] updated Table 3a/3b rows and the GP coverage line")
    print("[assimilate] done", flush=True)


if __name__ == "__main__":
    main()
