"""
Step 2: Contrastive EXPLAIN INFER — isolate System 2 counting/abstract math circuits.

Target vindex: ablation_project/indices/gemma4_deep_weights.vindex
  - Layers 20..41, extract_level=all
  - GGUF fallback: loads FFN weights for layers 20..41 from Q8_0 (~6.9 GB f32, safe on 32 GB)

Strategy — contrastive honeypot pairs:
  TARGET  (abstract counting / math):
    "The act of reciting numbers in order is called"  → counting
    "Addition, subtraction, and multiplication are forms of"  → arithmetic / math

  CONTROL (similar surface structure, different domain):
    "The act of reciting poems in order is called"  → recitation / poetry
    "Painting, sculpting, and drawing are forms of"  → art / visual arts

Method:
  1. Run EXPLAIN INFER on each prompt → capture per-feature gate activations for layers 20..41
  2. For each target, compute target_features = features where gate > GATE_THRESHOLD
  3. Control-subtract: drop any feature that also fires in the matched control at >= CONTROL_FRACTION
  4. The residual is the abstract counting/math circuit
  5. Export candidate features as a .vlp patch for ablate.rs

Rigor level: exploratory (qualitative gate threshold, no cross-validation).
"""

import subprocess
import json
import os
import sys
import time
import re

# ==========================================================
#  Config
# ==========================================================
LARQL_BIN = os.environ.get("LARQL_BIN", "cargo run --release --bin larql --")
VINDEX = os.environ.get(
    "VINDEX",
    "ablation_project/indices/gemma4_deep_weights.vindex",
)
OUTPUT_DIR = "ablation_project/results/v9_deep_contrastive"
VLP_OUT    = "ablation_project/patches/v9_counting_ablation.vlp"

GATE_THRESHOLD    = 1.0   # minimum gate score to count a feature as "firing"
CONTROL_FRACTION  = 0.5   # if control gate >= GATE_THRESHOLD * CONTROL_FRACTION, suppress
TOP_K             = 85    # EXPLAIN INFER top_k

# Contrastive pairs: (target_prompt, control_prompt, label)
PAIRS = [
    (
        "The act of reciting numbers in order is called",
        "The act of reciting poems in order is called",
        "counting_vs_poetry",
    ),
    (
        "Addition, subtraction, and multiplication are forms of",
        "Painting, sculpting, and drawing are forms of",
        "math_vs_art",
    ),
]


# ==========================================================
#  Helpers
# ==========================================================
LARQL_CMD_BASE = ["cargo", "run", "--release", "--bin", "larql", "--"]

def run_explain_infer(vindex: str, prompt: str, top_k: int = TOP_K) -> list:
    """
    Runs:
      larql lql "USE '<vindex>'; EXPLAIN INFER '<prompt>' TOP <top_k>;"
    and returns the stdout lines.
    """
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
        clean_err = result.stderr[-2000:].encode('ascii', 'ignore').decode('ascii')
        print(clean_err)
        return []

    print(f"  Done in {elapsed:.1f}s")
    return result.stdout.splitlines()


def extract_activations(lines: list) -> dict[tuple, float]:
    """
    Parse EXPLAIN INFER lines into {(layer, feature): gate_score} dict.
    Lines look like:
      L38:                F6244  gate=+18.4  -> igl             [igl, ...]
    """
    activations: dict[tuple, float] = {}
    pattern = re.compile(r"L\s*(?P<layer>\d+):.*F\s*(?P<feature>\d+)\s+gate=(?P<gate>[+-]?\d+\.?\d*)")

    for line in lines:
        m = pattern.search(line)
        if m:
            layer = int(m.group("layer"))
            feat = int(m.group("feature"))
            gate = abs(float(m.group("gate")))
            key = (layer, feat)
            activations[key] = max(activations.get(key, 0.0), gate)

    return activations


def set_subtract(
    target: dict[tuple, float],
    control: dict[tuple, float],
    gate_thresh: float,
    control_frac: float,
) -> dict[tuple, float]:
    """
    Return features that fire in target at >= gate_thresh but NOT in control
    at >= gate_thresh * control_frac.
    """
    suppression_thresh = gate_thresh * control_frac
    result = {}
    for key, score in target.items():
        if score < gate_thresh:
            continue
        control_score = control.get(key, 0.0)
        if control_score >= suppression_thresh:
            continue  # shared with control, suppress
        result[key] = score
    return result


def format_vlp(features: dict[tuple, float], label: str) -> str:
    """Format surviving features as a JSON-compatible VLP patch."""
    operations = []
    for (layer, feat), score in sorted(features.items(), key=lambda x: -x[1]):
        operations.append({
            "op": "delete",
            "layer": layer,
            "feature": feat,
            "reason": f"contrastive_gate_score_{score:.3f}"
        })
    patch = {
        "version": 1,
        "base_model": "google/gemma-4-e4b-it",
        "base_checksum": None,
        "created_at": "2026-05-21T23:55:00Z",
        "description": f"System 2 counting circuit ablation for label {label}",
        "author": "Antigravity",
        "tags": ["counting", "ablation", label],
        "operations": operations
    }
    return json.dumps(patch, indent=2)


# ==========================================================
#  Main
# ==========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VLP_OUT), exist_ok=True)

    print("=" * 70)
    print(f"  Contrastive EXPLAIN INFER - System 2 counting circuit isolation")
    print(f"  VIndex: {VINDEX}")
    print(f"  Gate threshold: {GATE_THRESHOLD}  Control fraction: {CONTROL_FRACTION}")
    print("=" * 70)

    all_target_activations: dict[tuple, float] = {}
    all_control_activations: dict[tuple, float] = {}
    pair_results = []

    for target_prompt, control_prompt, label in PAIRS:
        print(f"\n{'-'*70}")
        print(f"  Pair: {label}")
        print(f"  TARGET:  {target_prompt}")
        print(f"  CONTROL: {control_prompt}")
        print(f"{'-'*70}")

        print(f"\n  Running EXPLAIN INFER on TARGET...")
        target_raw = run_explain_infer(VINDEX, target_prompt)
        target_acts = extract_activations(target_raw)
        print(f"  TARGET activations extracted: {len(target_acts)} (feature, layer) pairs")

        print(f"\n  Running EXPLAIN INFER on CONTROL...")
        control_raw = run_explain_infer(VINDEX, control_prompt)
        control_acts = extract_activations(control_raw)
        print(f"  CONTROL activations extracted: {len(control_acts)} (feature, layer) pairs")

        # Accumulate across pairs
        for k, v in target_acts.items():
            all_target_activations[k] = max(all_target_activations.get(k, 0.0), v)
        for k, v in control_acts.items():
            all_control_activations[k] = max(all_control_activations.get(k, 0.0), v)

        # Per-pair subtraction
        unique = set_subtract(target_acts, control_acts, GATE_THRESHOLD, CONTROL_FRACTION)
        print(f"\n  After set subtraction: {len(unique)} unique features (gate >= {GATE_THRESHOLD})")

        # Show top 20
        top = sorted(unique.items(), key=lambda x: -x[1])[:20]
        print(f"  {'Layer':<8} {'Feature':<10} {'Gate':<10}")
        for (layer, feat), score in top:
            print(f"  L{layer:<7d} F{feat:<9d} {score:<10.3f}")

        # Save per-pair raw output
        pair_out = {
            "label": label,
            "target_prompt": target_prompt,
            "control_prompt": control_prompt,
            "target_activations": {f"{k[0]}:{k[1]}": v for k, v in target_acts.items()},
            "control_activations": {f"{k[0]}:{k[1]}": v for k, v in control_acts.items()},
            "unique_features": {f"{k[0]}:{k[1]}": v for k, v in unique.items()},
        }
        pair_path = os.path.join(OUTPUT_DIR, f"{label}_raw.json")
        with open(pair_path, "w", encoding="utf-8") as f:
            json.dump(pair_out, f, indent=2)
        print(f"  Saved: {pair_path}")

        pair_results.append({
            "label": label,
            "target_activations": len(target_acts),
            "control_activations": len(control_acts),
            "unique_features": len(unique),
            "top_features": [{"layer": k[0], "feature": k[1], "gate": v}
                             for k, v in top],
        })

    # Cross-pair intersection: features that survive BOTH pairs
    print(f"\n{'='*70}")
    print(f"  Cross-pair intersection (features surviving both contrastive pairs)")
    print(f"{'='*70}")

    combined_unique = set_subtract(
        all_target_activations, all_control_activations,
        GATE_THRESHOLD, CONTROL_FRACTION,
    )
    print(f"\n  Combined unique features: {len(combined_unique)}")

    # Layer distribution
    by_layer: dict[int, list] = {}
    for (layer, feat), score in combined_unique.items():
        by_layer.setdefault(layer, []).append((feat, score))

    print(f"\n  Layer distribution:")
    print(f"  {'Layer':<8} {'Count':<8} {'Max gate':<12} {'Top feature'}")
    for layer in sorted(by_layer.keys()):
        feats = sorted(by_layer[layer], key=lambda x: -x[1])
        top_feat, top_gate = feats[0]
        print(f"  L{layer:<7d} {len(feats):<8d} {top_gate:<12.3f} F{top_feat}")

    # Export VLP patch
    vlp_content = format_vlp(combined_unique, "counting_and_math_circuits")
    with open(VLP_OUT, "w", encoding="utf-8") as f:
        f.write(vlp_content)
    print(f"\n  VLP patch saved: {VLP_OUT}  ({len(combined_unique)} features)")

    # Summary JSON
    # ──────────────────────────────────────────────────────
    summary = {
        "vindex": VINDEX,
        "gate_threshold": GATE_THRESHOLD,
        "control_fraction": CONTROL_FRACTION,
        "top_k": TOP_K,
        "pairs": pair_results,
        "combined_unique_features": len(combined_unique),
        "layer_distribution": {
            str(l): len(fs) for l, fs in sorted(by_layer.items())
        },
        "vlp_patch": VLP_OUT,
    }
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {summary_path}")

    print(f"\n{'='*70}")
    print(f"  DONE - {len(combined_unique)} candidate features for ablation")
    print(f"  Next: cargo run --release --bin larql -- ablate '{VINDEX}' '{VLP_OUT}'")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
