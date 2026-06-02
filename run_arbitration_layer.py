#!/usr/bin/env python3
"""Command-line runner for DAMSAL arbitration layer."""
from __future__ import annotations

import argparse
from pathlib import Path

from arbitration_layer import ArbitrationWeights, run_arbitration, default_strategy_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run degradation-aware multi-strategy arbitration layer.")
    parser.add_argument("--clean_csv", required=False, help="Clean baseline CSV/XLSX output from HVAC solver.")
    parser.add_argument("--degraded_csv", required=False, help="Degraded baseline CSV/XLSX output from HVAC solver.")
    parser.add_argument("--strategy_bank_csv", default=None, help="Optional custom strategy bank CSV.")
    parser.add_argument("--out_dir", default="arbitration_outputs", help="Output directory.")
    parser.add_argument("--write_default_strategy_bank", action="store_true", help="Only write the default strategy bank CSV and exit.")
    parser.add_argument("--w_energy", type=float, default=0.45)
    parser.add_argument("--w_comfort", type=float, default=0.20)
    parser.add_argument("--w_degradation", type=float, default=0.15)
    parser.add_argument("--w_maintenance", type=float, default=0.10)
    parser.add_argument("--w_robustness", type=float, default=0.07)
    parser.add_argument("--w_switching", type=float, default=0.03)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.write_default_strategy_bank:
        out = out_dir / "default_strategy_bank.csv"
        default_strategy_dataframe().to_csv(out, index=False)
        print(f"Wrote {out}")
        return

    if not args.clean_csv or not args.degraded_csv:
        raise SystemExit("You must provide --clean_csv and --degraded_csv, or use --write_default_strategy_bank.")

    weights = ArbitrationWeights(
        energy=args.w_energy,
        comfort=args.w_comfort,
        degradation=args.w_degradation,
        maintenance=args.w_maintenance,
        robustness=args.w_robustness,
        switching=args.w_switching,
    )
    paths = run_arbitration(
        clean_path=args.clean_csv,
        degraded_path=args.degraded_csv,
        out_dir=out_dir,
        strategy_bank_path=args.strategy_bank_csv,
        weights=weights,
    )
    print("DAMSAL arbitration completed. Outputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
