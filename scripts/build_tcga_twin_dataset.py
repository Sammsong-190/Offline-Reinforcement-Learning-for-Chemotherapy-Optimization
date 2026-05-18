#!/usr/bin/env python3
"""Build offline datasets from TCGA clinical TSV × chemo twin rollouts."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.generate import collect_trajectory, save_dataset  # noqa: E402
from env.tcga_mapper import tcga_behavior_policy  # noqa: E402
from env.tcga_twins import load_tcga_twins_from_clinical, train_eval_split_twins  # noqa: E402


DEFAULT_CLINICAL = Path(
    "/Users/song666/Desktop/clinical.cart.2026-05-16/clinical.tsv"
)


def generate_dataset_from_twins(
    twins: list,
    seed: int,
    episodes_per_patient: int = 2,
):
    rng = np.random.default_rng(seed)
    transitions = []

    for twin in twins:
        stage = twin["meta"]["stage"]
        policy = tcga_behavior_policy(stage, rng)

        for _ in range(episodes_per_patient):
            traj = collect_trajectory(
                policy=policy,
                x0=twin["x0"],
                randomize_patient=False,
                state_noise_sigma=0.02,
                patient_ctx=twin["patient_ctx"],
                rng=rng,
            )

            for t in traj:
                t["cohort"] = twin["patient_ctx"]["cohort"]

            transitions.extend(traj)

    return transitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clinical",
        type=Path,
        default=DEFAULT_CLINICAL,
        help="Path to TCGA clinical.tsv",
    )
    parser.add_argument(
        "--out-train",
        type=Path,
        default=ROOT / "data/raw/tcga_twin_train.npz",
        help="Output path for training dataset",
    )
    parser.add_argument(
        "--out-eval",
        type=Path,
        default=ROOT / "data/raw/tcga_twin_eval.npz",
        help="Output path for eval dataset",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.72,
        help="Patient-level fraction for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for train split and train rollouts base",
    )
    parser.add_argument(
        "--episodes-train",
        type=int,
        default=26,
        help="Episodes per patient (train); higher when trajectories are short after patient_ctx termination",
    )
    parser.add_argument(
        "--episodes-eval",
        type=int,
        default=10,
        help="Episodes per patient (eval)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=2026,
        help="RNG seed for eval rollouts",
    )
    args = parser.parse_args()

    clinical_path = args.clinical
    if not clinical_path.exists():
        print(f"[ERROR] Missing clinical file: {clinical_path}")
        return 1

    twins = load_tcga_twins_from_clinical(clinical_path)

    print(f"[INFO] Twins with age+stage: {len(twins)} (after case dedup & filter).")

    if not twins:
        print("[ERROR] No twins; check clinical.tsv columns and parsing.")
        return 1

    train_twins, eval_twins = train_eval_split_twins(
        twins, args.train_frac, args.seed
    )

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_eval.parent.mkdir(parents=True, exist_ok=True)

    train_data = generate_dataset_from_twins(
        train_twins,
        seed=args.seed,
        episodes_per_patient=args.episodes_train,
    )
    eval_data = generate_dataset_from_twins(
        eval_twins,
        seed=args.eval_seed,
        episodes_per_patient=args.episodes_eval,
    )

    save_dataset(train_data, str(args.out_train))
    save_dataset(eval_data, str(args.out_eval))

    c_mean = np.mean([t.get("c", 0.0) for t in train_data])
    print(
        f"[INFO] Train cost violation rate (c=1 frac): {100.0 * c_mean:.2f}% "
        f"(ideal rough range ~5–25%)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
