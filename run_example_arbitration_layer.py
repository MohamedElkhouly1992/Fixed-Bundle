"""Self-contained example for DAMSAL arbitration layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from arbitration_layer import run_arbitration, default_strategy_dataframe

np.random.seed(42)
out_dir = Path("example_arbitration_outputs")
out_dir.mkdir(exist_ok=True)

# Example clean/degraded daily outputs for one year.
dates = pd.date_range("2024-01-01", periods=365, freq="D")
doy = np.arange(365)
season = 0.5 + 0.5 * np.sin(2 * np.pi * (doy - 120) / 365)
clean_energy = 3500 + 2200 * season + 300 * np.random.randn(365)
clean_energy = np.maximum(clean_energy, 1000)
severity = np.clip(0.15 + 0.55 * season + 0.1 * np.random.randn(365), 0, 1)
excess = clean_energy * (0.08 + 0.22 * severity)
degraded_energy = clean_energy + excess

clean = pd.DataFrame({"Date/Time": dates, "energy_kwh_day": clean_energy})
degraded = pd.DataFrame({"Date/Time": dates, "energy_kwh_day": degraded_energy, "MDI": severity})
clean_path = out_dir / "example_clean_baseline.csv"
deg_path = out_dir / "example_degraded_scenario.csv"
strategy_path = out_dir / "default_strategy_bank.csv"
clean.to_csv(clean_path, index=False)
degraded.to_csv(deg_path, index=False)
default_strategy_dataframe().to_csv(strategy_path, index=False)
paths = run_arbitration(clean_path, deg_path, out_dir=out_dir, strategy_bank_path=strategy_path)
print("Example complete:")
for k, v in paths.items():
    print(f"{k}: {v}")
