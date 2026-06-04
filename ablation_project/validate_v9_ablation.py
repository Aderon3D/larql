"""
validate_v9_ablation.py
=======================
Post-ablation validation for v9_counting_ablation.vlp.

Rigor level: exploratory (qualitative gate threshold comparison, no cross-validation).

Two checks:
  1. SUPPRESSION CHECK: Re-run EXPLAIN INFER on the original counting/math prompts.
     For each ablated feature, record its post-ablation gate score.
     Expected: ablated features fire at gate < GATE_THRESHOLD (ideally 0.0).

  2. COHERENCE CHECK: Run EXPLAIN INFER on neutral / control prompts.
     Check that total activation mass on non-ablated features is not catastrophically
     reduced (i.e. the model still processes language).

Results are written to ablation_project/results/v9_validation/report.json
and a human-readable summary to ablation_project/results/v9_validation/summary.txt.
"""

import subprocess
import json
import os
import re
import time

# ==========================================================
#  Config
# ==========================================================
VINDEX      = os.environ.get("VINDEX", "ablation_project/indices/gemma4_deep_weights.vindex")
VLP_PATH    = "ablation_project/patches/v9_counting_ablation.vlp"
OUTPUT_DIR  = "ablation_project/results/v9_validation"
TOP_K       = 85
GATE_THRESHOLD = 1.0   # same threshold used during profiling

LARQL_CMD_BASE = ["cargo", "run", "--release", "--bin", "larql", "--"]

# Probes: (prompt, category)
MATH_PROBES = [
    ("The act of reciting numbers in order is called", "math"),
    ("Addition, subtraction, and multiplication are forms of", "math"),
    ("If you have five apples and eat two, how many remain", "math"),
    ("The square root of sixteen is", "math"),
]

NEUTRAL_PROBES = [
    ("The capital of France is", "neutral"),
    ("Water freezes at zero degrees", "neutral"),
    ("The sky appears blue because of", "neutral"),
    ("A haiku is a form of poetry that", "neutral"),
]

# ==========================================================
#  Helpers
# ==========================================================

def run_explain_infer(vindex: str, prompt: str, top_k: int = TOP_K) -> list:
    lql = f"USE '{vindex}'; EXPLAIN INFER '{prompt}' TOP {top_k};"
    cmd = LARQL_CMD_BASE + ["lql", lql]
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [ERROR] EXPLAIN INFER failed (exit {result.returncode}):")
        print(result.stderr[-2000:].encode("ascii", "ignore").decode("ascii"))
        return []
    print(f"  Done in {elapsed:.1f}s")
    return result.stdout.splitlines()


def extract_activations(lines: list) -> dict:
    """Parse EXPLAIN INFER output into {(layer, feature): gate_score}."""
    activations = {}
    pattern = re.compile(
        r"L\s*(?P<layer>\d+):.*F\s*(?P<feature>\d+)\s+gate=(?P<gate>[+-]?\d+\.?\d*)"
    )
    for line in lines:
        m = pattern.search(line)
        if m:
            layer   = int(m.group("layer"))
            feat    = int(m.group("feature"))
            gate    = abs(float(m.group("gate")))
            key     = (layer, feat)
            activations[key] = max(activations.get(key, 0.0), gate)
    return activations


def load_ablated_features(vlp_path: str) -> set:
    """Return the set of (layer, feature) tuples that were ablated."""
    with open(vlp_path, "r", encoding="utf-8") as f:
        patch = json.load(f)
    ablated = set()
    for op in patch.get("operations", []):
        if op.get("op") == "delete":
            ablated.add((op["layer"], op["feature"]))
    return ablated


# ==========================================================
#  Main
# ==========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  V9 Ablation Validation")
    print(f"  VIndex: {VINDEX}")
    print(f"  VLP:    {VLP_PATH}")
    print("=" * 70)

    # Load ablated features
    ablated = load_ablated_features(VLP_PATH)
    print(f"\nLoaded {len(ablated)} ablated (layer, feature) pairs from patch.\n")

    # --- SUPPRESSION CHECK ---
    print("=" * 70)
    print("  1. SUPPRESSION CHECK (math/counting probes)")
    print("=" * 70)

    suppression_results = []
    for prompt, category in MATH_PROBES:
        print(f"\n  Prompt: {prompt!r}")
        lines = run_explain_infer(VINDEX, prompt)
        acts  = extract_activations(lines)

        total_features   = len(acts)
        ablated_firing   = {k: v for k, v in acts.items() if k in ablated}
        survived_ablated = {k: v for k, v in ablated_firing.items() if v >= GATE_THRESHOLD}

        print(f"  Total features in output:         {total_features}")
        print(f"  Ablated features still firing:    {len(ablated_firing)}")
        print(f"  ... above gate threshold ({GATE_THRESHOLD}): {len(survived_ablated)}")

        if survived_ablated:
            top = sorted(survived_ablated.items(), key=lambda x: -x[1])[:5]
            print(f"  Top surviving ablated features (unexpected):")
            for (l, f), g in top:
                print(f"    L{l} F{f}  gate={g:.3f}")
        else:
            print("  All ablated features suppressed below threshold. [PASS]")

        suppression_results.append({
            "prompt": prompt,
            "category": category,
            "total_features": total_features,
            "ablated_still_firing": len(ablated_firing),
            "ablated_above_threshold": len(survived_ablated),
            "pass": len(survived_ablated) == 0,
            "top_survivors": [
                {"layer": k[0], "feature": k[1], "gate": v}
                for k, v in sorted(survived_ablated.items(), key=lambda x: -x[1])[:10]
            ],
        })

    # --- COHERENCE CHECK ---
    print("\n" + "=" * 70)
    print("  2. COHERENCE CHECK (neutral probes)")
    print("=" * 70)

    coherence_results = []
    for prompt, category in NEUTRAL_PROBES:
        print(f"\n  Prompt: {prompt!r}")
        lines = run_explain_infer(VINDEX, prompt)
        acts  = extract_activations(lines)

        total_features  = len(acts)
        total_gate_mass = sum(acts.values())
        ablated_in_out  = {k: v for k, v in acts.items() if k in ablated}

        print(f"  Total features in output:   {total_features}")
        print(f"  Total gate mass:            {total_gate_mass:.1f}")
        print(f"  Ablated features present:   {len(ablated_in_out)}")

        # Coherence pass: model still outputs >= 10 features with gate > threshold
        active = sum(1 for v in acts.values() if v >= GATE_THRESHOLD)
        coherent = active >= 10
        print(f"  Features >= gate threshold: {active}  ->  {'COHERENT [PASS]' if coherent else 'INCOHERENT [FAIL]'}")

        coherence_results.append({
            "prompt": prompt,
            "category": category,
            "total_features": total_features,
            "total_gate_mass": total_gate_mass,
            "active_above_threshold": active,
            "ablated_features_in_output": len(ablated_in_out),
            "pass": coherent,
        })

    # --- SUMMARY ---
    suppression_pass = all(r["pass"] for r in suppression_results)
    coherence_pass   = all(r["pass"] for r in coherence_results)

    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Suppression check: {'PASS' if suppression_pass else 'PARTIAL/FAIL'}")
    print(f"  Coherence check:   {'PASS' if coherence_pass else 'FAIL'}")
    print(f"  Overall:           {'SUCCESS' if suppression_pass and coherence_pass else 'NEEDS REVIEW'}")

    # Save report
    report = {
        "vlp_path": VLP_PATH,
        "ablated_feature_count": len(ablated),
        "gate_threshold": GATE_THRESHOLD,
        "suppression_pass": suppression_pass,
        "coherence_pass": coherence_pass,
        "suppression_results": suppression_results,
        "coherence_results": coherence_results,
    }
    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved: {report_path}")

    # Human-readable summary
    summary_lines = [
        "V9 Ablation Validation Summary",
        "=" * 50,
        f"Patch: {VLP_PATH}",
        f"Ablated features: {len(ablated)}",
        f"Gate threshold: {GATE_THRESHOLD}",
        "",
        "SUPPRESSION CHECK (math/counting prompts):",
    ]
    for r in suppression_results:
        status = "PASS" if r["pass"] else f"FAIL ({r['ablated_above_threshold']} survived)"
        summary_lines.append(f"  [{status}] {r['prompt'][:60]}")
    summary_lines.append("")
    summary_lines.append("COHERENCE CHECK (neutral prompts):")
    for r in coherence_results:
        status = "PASS" if r["pass"] else "FAIL"
        summary_lines.append(
            f"  [{status}] {r['prompt'][:60]}  "
            f"(active={r['active_above_threshold']}, mass={r['total_gate_mass']:.0f})"
        )
    summary_lines.append("")
    summary_lines.append(f"OVERALL: {'SUCCESS' if suppression_pass and coherence_pass else 'NEEDS REVIEW'}")

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  Summary saved:     {summary_path}")


if __name__ == "__main__":
    main()
