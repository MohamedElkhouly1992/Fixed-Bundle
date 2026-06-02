"""
DAMSAL: Degradation-Aware Multi-Strategy Arbitration Layer
-----------------------------------------------------------
Post-layer for HVAC v3 dynamic reduced-order solver outputs.

Purpose
=======
Evaluate multiple degradation-mitigation/control candidates (S0-S6) using
clean and degraded solver outputs, then select the best strategy per day and
summarize whether S3 is actually superior or not.

This module intentionally works as a post-layer: it does not change the core
HVAC solver equations. It consumes clean/degraded time-series outputs and
produces arbitration results.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math
import json

import numpy as np
import pandas as pd


ENERGY_CANDIDATES = [
    "energy_kwh_day",
    "energy_kwh_period",
    "total_energy_kwh",
    "total_hvac_kwh",
    "E_total_kWh",
    "energy_kwh",
    "Energy_kWh",
    "Total Energy",
    "Total Energy kWh",
]

DATE_CANDIDATES = ["Date/Time", "datetime", "date", "Date", "time", "timestamp"]
COMFORT_CANDIDATES = ["comfort_deviation", "comfort_penalty", "unmet_hours", "PMV_penalty"]
MDI_CANDIDATES = ["MDI", "mean_degradation_index", "degradation_index", "Severity", "severity"]
BHI_CANDIDATES = ["BHI", "building_health_index", "health_index"]


@dataclass
class ArbitrationWeights:
    energy: float = 0.45
    comfort: float = 0.20
    degradation: float = 0.15
    maintenance: float = 0.10
    robustness: float = 0.07
    switching: float = 0.03

    def normalized(self) -> "ArbitrationWeights":
        values = np.array([self.energy, self.comfort, self.degradation, self.maintenance, self.robustness, self.switching], dtype=float)
        s = float(np.sum(values))
        if s <= 0:
            return self
        return ArbitrationWeights(*(values / s))


@dataclass
class StrategyDefinition:
    strategy: str
    name: str
    description: str
    base_recovery: float
    severity_gain: float
    max_recovery: float
    comfort_penalty_coeff: float
    degradation_slowdown: float
    maintenance_cost: float
    robustness_penalty_coeff: float
    switching_penalty_coeff: float
    preferred_min_severity: float = 0.0
    preferred_max_severity: float = 1.0

    def recovery_fraction(self, severity_score: np.ndarray) -> np.ndarray:
        raw = self.base_recovery + self.severity_gain * severity_score
        # Penalize if strategy is used far outside its preferred severity interval.
        below = np.maximum(self.preferred_min_severity - severity_score, 0)
        above = np.maximum(severity_score - self.preferred_max_severity, 0)
        mismatch = below + above
        raw = raw * np.maximum(0.50, 1.0 - 0.60 * mismatch)
        return np.clip(raw, 0.0, self.max_recovery)


DEFAULT_STRATEGIES: List[StrategyDefinition] = [
    StrategyDefinition(
        strategy="S0",
        name="No corrective control",
        description="Degraded operation without additional control or maintenance.",
        base_recovery=0.00,
        severity_gain=0.00,
        max_recovery=0.00,
        comfort_penalty_coeff=0.00,
        degradation_slowdown=0.00,
        maintenance_cost=0.00,
        robustness_penalty_coeff=0.10,
        switching_penalty_coeff=0.00,
        preferred_min_severity=0.0,
        preferred_max_severity=1.0,
    ),
    StrategyDefinition(
        strategy="S1",
        name="Rule-based passive correction",
        description="Simple rule-based setpoint and runtime correction; low complexity and low risk.",
        base_recovery=0.15,
        severity_gain=0.08,
        max_recovery=0.28,
        comfort_penalty_coeff=0.08,
        degradation_slowdown=0.05,
        maintenance_cost=0.00,
        robustness_penalty_coeff=0.08,
        switching_penalty_coeff=0.03,
        preferred_min_severity=0.0,
        preferred_max_severity=0.40,
    ),
    StrategyDefinition(
        strategy="S2",
        name="Schedule-aware active control",
        description="Monthly/occupancy/HVAC-availability correction with active scheduling.",
        base_recovery=0.25,
        severity_gain=0.14,
        max_recovery=0.45,
        comfort_penalty_coeff=0.12,
        degradation_slowdown=0.10,
        maintenance_cost=0.00,
        robustness_penalty_coeff=0.09,
        switching_penalty_coeff=0.06,
        preferred_min_severity=0.10,
        preferred_max_severity=0.65,
    ),
    StrategyDefinition(
        strategy="S3",
        name="Predictive degradation-aware control",
        description="Rolling-horizon predictive control using the dynamic ROM digital twin.",
        base_recovery=0.35,
        severity_gain=0.18,
        max_recovery=0.62,
        comfort_penalty_coeff=0.18,
        degradation_slowdown=0.20,
        maintenance_cost=0.00,
        robustness_penalty_coeff=0.12,
        switching_penalty_coeff=0.09,
        preferred_min_severity=0.20,
        preferred_max_severity=0.85,
    ),
    StrategyDefinition(
        strategy="S4",
        name="Maintenance-triggered control",
        description="Control plus maintenance trigger when degradation becomes physically dominant.",
        base_recovery=0.22,
        severity_gain=0.55,
        max_recovery=0.78,
        comfort_penalty_coeff=0.10,
        degradation_slowdown=0.55,
        maintenance_cost=0.22,
        robustness_penalty_coeff=0.07,
        switching_penalty_coeff=0.05,
        preferred_min_severity=0.55,
        preferred_max_severity=1.0,
    ),
    StrategyDefinition(
        strategy="S5",
        name="Robust uncertainty-aware control",
        description="Conservative control optimized against weather/occupancy/degradation uncertainty.",
        base_recovery=0.30,
        severity_gain=0.12,
        max_recovery=0.52,
        comfort_penalty_coeff=0.11,
        degradation_slowdown=0.18,
        maintenance_cost=0.00,
        robustness_penalty_coeff=0.03,
        switching_penalty_coeff=0.07,
        preferred_min_severity=0.15,
        preferred_max_severity=0.90,
    ),
    StrategyDefinition(
        strategy="S6",
        name="Hybrid control-maintenance strategy",
        description="Combined predictive control, fan/runtime correction, and maintenance decision.",
        base_recovery=0.38,
        severity_gain=0.38,
        max_recovery=0.82,
        comfort_penalty_coeff=0.15,
        degradation_slowdown=0.65,
        maintenance_cost=0.28,
        robustness_penalty_coeff=0.05,
        switching_penalty_coeff=0.11,
        preferred_min_severity=0.45,
        preferred_max_severity=1.0,
    ),
]


def default_strategy_dataframe() -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in DEFAULT_STRATEGIES])


def load_strategy_bank(path: Optional[str | Path] = None) -> List[StrategyDefinition]:
    if path is None:
        return DEFAULT_STRATEGIES
    df = pd.read_csv(path)
    required = set(asdict(DEFAULT_STRATEGIES[0]).keys())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Strategy bank CSV is missing required columns: {sorted(missing)}")
    strategies = []
    for _, row in df.iterrows():
        kwargs = {k: row[k] for k in required}
        # Cast numeric fields safely.
        for k in required - {"strategy", "name", "description"}:
            kwargs[k] = float(kwargs[k])
        strategies.append(StrategyDefinition(**kwargs))
    return strategies


def find_first_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]
    # Fuzzy includes.
    for candidate in candidates:
        key = candidate.strip().lower()
        for col in df.columns:
            if key in str(col).strip().lower():
                return col
    return None


def read_timeseries(path: str | Path, sheet: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        if sheet is None:
            xls = pd.ExcelFile(path)
            # Prefer daily_data then first sheet.
            sheet = "daily_data" if "daily_data" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_csv(path)
    df = df.dropna(how="all").copy()
    return df


def _prepare_arbitration_timeseries(
    df: pd.DataFrame,
    energy_col: str,
    date_col: Optional[str],
    out_energy_col: str,
) -> pd.DataFrame:
    """Prepare a compact daily/periodic series for arbitration.

    This function prevents many-to-many date merges. If an uploaded file has
    duplicate Date/Time values, or hourly timestamps that collapse to repeated
    dates after parsing, it aggregates to one row per date before merging.
    """
    out = pd.DataFrame({out_energy_col: pd.to_numeric(df[energy_col], errors="coerce")})
    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        if dt.notna().sum() > 0:
            out["Date/Time"] = dt.dt.floor("D")
            out = out.dropna(subset=["Date/Time", out_energy_col])
            # Aggregate duplicate dates to avoid catastrophic many-to-many merges.
            out = out.groupby("Date/Time", as_index=False)[out_energy_col].sum()
            return out
    out = out.dropna(subset=[out_energy_col]).reset_index(drop=True)
    out["Date/Time"] = np.arange(len(out))
    return out[["Date/Time", out_energy_col]]


def align_clean_degraded(clean: pd.DataFrame, degraded: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    clean = clean.copy()
    degraded = degraded.copy()
    if date_col is None:
        c_date = find_first_column(clean, DATE_CANDIDATES)
        d_date = find_first_column(degraded, DATE_CANDIDATES)
    else:
        c_date = d_date = date_col

    clean_energy_col = find_first_column(clean, ENERGY_CANDIDATES)
    degraded_energy_col = find_first_column(degraded, ENERGY_CANDIDATES)
    if clean_energy_col is None:
        raise ValueError(f"Cannot identify clean energy column. Available columns: {list(clean.columns)}")
    if degraded_energy_col is None:
        raise ValueError(f"Cannot identify degraded energy column. Available columns: {list(degraded.columns)}")

    clean_small = _prepare_arbitration_timeseries(clean, clean_energy_col, c_date, "clean_energy_kwh")
    degraded_small = _prepare_arbitration_timeseries(degraded, degraded_energy_col, d_date, "degraded_energy_kwh")

    # Prefer safe one-to-one date alignment after aggregation.
    if "Date/Time" in clean_small.columns and "Date/Time" in degraded_small.columns:
        merged = pd.merge(degraded_small, clean_small, on="Date/Time", how="inner", validate="one_to_one")
        if len(merged) >= min(len(clean_small), len(degraded_small)) * 0.80:
            return merged.sort_values("Date/Time").reset_index(drop=True)

    # Fallback: row-order alignment. This is safer than a many-to-many date merge.
    n = min(len(clean_small), len(degraded_small))
    merged = pd.concat([
        degraded_small[["degraded_energy_kwh"]].iloc[:n].reset_index(drop=True),
        clean_small[["clean_energy_kwh"]].iloc[:n].reset_index(drop=True),
    ], axis=1)
    if c_date is not None:
        raw_dates = pd.to_datetime(clean[c_date], errors="coerce")
        if raw_dates.notna().sum() > 0:
            merged["Date/Time"] = raw_dates.dt.floor("D").dropna().iloc[:n].values
        else:
            merged["Date/Time"] = np.arange(n)
    elif d_date is not None:
        raw_dates = pd.to_datetime(degraded[d_date], errors="coerce")
        if raw_dates.notna().sum() > 0:
            merged["Date/Time"] = raw_dates.dt.floor("D").dropna().iloc[:n].values
        else:
            merged["Date/Time"] = np.arange(n)
    else:
        merged["Date/Time"] = np.arange(n)
    return merged

def infer_severity_score(df: pd.DataFrame, excess_col: str = "excess_energy_kwh") -> np.ndarray:
    col = find_first_column(df, MDI_CANDIDATES)
    if col is not None:
        x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        # Normalize if values look like percentages or arbitrary MDI scale.
        mx = np.nanmax(x) if len(x) else 1.0
        if mx > 1.5:
            x = x / max(mx, 1.0)
        return np.clip(x, 0, 1)
    clean = np.maximum(pd.to_numeric(df.get("clean_energy_kwh", 0), errors="coerce").fillna(0).to_numpy(dtype=float), 1e-9)
    excess = np.maximum(pd.to_numeric(df.get(excess_col, 0), errors="coerce").fillna(0).to_numpy(dtype=float), 0)
    # Excess ratio of 30% or above is treated as high severity.
    return np.clip(excess / (0.30 * clean), 0, 1)


def infer_uncertainty_score(df: pd.DataFrame) -> np.ndarray:
    # If weather/occupancy mismatch indicators are absent, use a mild constant uncertainty.
    n = len(df)
    score = np.full(n, 0.20, dtype=float)
    for key in ["weather_uncertainty", "schedule_uncertainty", "occupancy_uncertainty"]:
        col = find_first_column(df, [key])
        if col is not None:
            x = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
            if np.nanmax(x) > 1.5:
                x = x / max(np.nanmax(x), 1.0)
            score = np.maximum(score, np.clip(x, 0, 1))
    return score


def evaluate_strategies(
    base: pd.DataFrame,
    strategies: Optional[List[StrategyDefinition]] = None,
    weights: ArbitrationWeights = ArbitrationWeights(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategies = strategies or DEFAULT_STRATEGIES
    weights = weights.normalized()
    df = base.copy()
    df["clean_energy_kwh"] = pd.to_numeric(df["clean_energy_kwh"], errors="coerce").fillna(0)
    df["degraded_energy_kwh"] = pd.to_numeric(df["degraded_energy_kwh"], errors="coerce").fillna(0)
    df["excess_energy_kwh"] = np.maximum(df["degraded_energy_kwh"] - df["clean_energy_kwh"], 0.0)
    severity = infer_severity_score(df)
    uncertainty = infer_uncertainty_score(df)
    clean = df["clean_energy_kwh"].to_numpy(dtype=float)
    degraded = df["degraded_energy_kwh"].to_numpy(dtype=float)
    excess = df["excess_energy_kwh"].to_numpy(dtype=float)
    clean_safe = np.maximum(clean, 1e-9)

    rows = []
    for s in strategies:
        recovery = s.recovery_fraction(severity)
        recovered_energy = recovery * excess
        controlled = degraded - recovered_energy
        controlled = np.maximum(controlled, clean * 0.90)  # avoid nonphysical over-recovery below clean baseline by >10%.
        recovered_energy = degraded - controlled
        excess_after = np.maximum(controlled - clean, 0.0)

        energy_term = excess_after / clean_safe
        comfort_term = s.comfort_penalty_coeff * recovery * (1 + 0.30 * severity)
        degradation_term = (1.0 - s.degradation_slowdown) * severity
        maintenance_term = np.full_like(severity, s.maintenance_cost)
        robustness_term = s.robustness_penalty_coeff * uncertainty
        switching_term = np.full_like(severity, s.switching_penalty_coeff)

        objective = (
            weights.energy * energy_term
            + weights.comfort * comfort_term
            + weights.degradation * degradation_term
            + weights.maintenance * maintenance_term
            + weights.robustness * robustness_term
            + weights.switching * switching_term
        )
        tmp = pd.DataFrame({
            "Date/Time": df["Date/Time"].values,
            "strategy": s.strategy,
            "strategy_name": s.name,
            "clean_energy_kwh": clean,
            "degraded_energy_kwh": degraded,
            "controlled_energy_kwh": controlled,
            "excess_before_kwh": excess,
            "excess_after_kwh": excess_after,
            "recovered_energy_kwh": recovered_energy,
            "recovery_fraction": recovery,
            "severity_score": severity,
            "uncertainty_score": uncertainty,
            "energy_term": energy_term,
            "comfort_penalty": comfort_term,
            "degradation_penalty": degradation_term,
            "maintenance_cost": maintenance_term,
            "robustness_penalty": robustness_term,
            "switching_penalty": switching_term,
            "objective": objective,
        })
        rows.append(tmp)

    long = pd.concat(rows, ignore_index=True)
    idx = long.groupby("Date/Time")["objective"].idxmin()
    daily_best = long.loc[idx].sort_values("Date/Time").reset_index(drop=True)
    daily_best = daily_best.rename(columns={"strategy": "selected_strategy", "strategy_name": "selected_strategy_name"})

    summary = long.groupby(["strategy", "strategy_name"], as_index=False).agg(
        clean_energy_kwh=("clean_energy_kwh", "sum"),
        degraded_energy_kwh=("degraded_energy_kwh", "sum"),
        controlled_energy_kwh=("controlled_energy_kwh", "sum"),
        excess_before_kwh=("excess_before_kwh", "sum"),
        excess_after_kwh=("excess_after_kwh", "sum"),
        recovered_energy_kwh=("recovered_energy_kwh", "sum"),
        mean_recovery_fraction=("recovery_fraction", "mean"),
        mean_comfort_penalty=("comfort_penalty", "mean"),
        mean_degradation_penalty=("degradation_penalty", "mean"),
        mean_maintenance_cost=("maintenance_cost", "mean"),
        mean_robustness_penalty=("robustness_penalty", "mean"),
        mean_objective=("objective", "mean"),
    )
    summary["energy_recovery_ratio_pct"] = np.where(
        summary["excess_before_kwh"] > 1e-9,
        100 * summary["recovered_energy_kwh"] / summary["excess_before_kwh"],
        0,
    )
    summary["total_energy_saving_vs_degraded_pct"] = np.where(
        summary["degraded_energy_kwh"] > 1e-9,
        100 * (summary["degraded_energy_kwh"] - summary["controlled_energy_kwh"]) / summary["degraded_energy_kwh"],
        0,
    )
    summary = summary.sort_values("mean_objective").reset_index(drop=True)
    return long, daily_best, summary


def pareto_front(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "controlled_energy_kwh",
        "mean_comfort_penalty",
        "mean_degradation_penalty",
        "mean_maintenance_cost",
        "mean_robustness_penalty",
    ]
    values = summary[metrics].to_numpy(dtype=float)
    n = len(summary)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                dominated[i] = True
                break
    out = summary.copy()
    out["pareto_status"] = np.where(dominated, "dominated", "non_dominated")
    return out


def s3_superiority(summary: pd.DataFrame) -> Dict[str, float | str]:
    if "S3" not in set(summary["strategy"]):
        return {"status": "S3 not available"}
    s3_obj = float(summary.loc[summary["strategy"] == "S3", "mean_objective"].iloc[0])
    alternatives = summary.loc[summary["strategy"] != "S3"].copy()
    if alternatives.empty:
        return {"status": "No alternatives"}
    best_alt = alternatives.sort_values("mean_objective").iloc[0]
    best_alt_obj = float(best_alt["mean_objective"])
    if abs(best_alt_obj) < 1e-12:
        ssi = 0.0
    else:
        ssi = 100 * (best_alt_obj - s3_obj) / abs(best_alt_obj)
    status = "S3 superior" if ssi > 0 else "S3 not superior"
    return {
        "status": status,
        "S3_mean_objective": s3_obj,
        "best_alternative": str(best_alt["strategy"]),
        "best_alternative_mean_objective": best_alt_obj,
        "SSI_S3_pct": ssi,
    }


def run_arbitration(
    clean_path: str | Path,
    degraded_path: str | Path,
    out_dir: str | Path = ".",
    strategy_bank_path: Optional[str | Path] = None,
    weights: ArbitrationWeights = ArbitrationWeights(),
    clean_sheet: Optional[str] = None,
    degraded_sheet: Optional[str] = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = read_timeseries(clean_path, clean_sheet)
    degraded = read_timeseries(degraded_path, degraded_sheet)
    base = align_clean_degraded(clean, degraded)
    strategies = load_strategy_bank(strategy_bank_path)
    long, daily_best, summary = evaluate_strategies(base, strategies=strategies, weights=weights)
    pareto = pareto_front(summary)
    s3 = s3_superiority(summary)
    s3_df = pd.DataFrame([s3])

    paths = {
        "daily_all_strategies": out_dir / "arbitration_daily_all_strategies.csv",
        "daily_selected": out_dir / "arbitration_daily_selected_strategy.csv",
        "summary": out_dir / "arbitration_strategy_summary.csv",
        "pareto": out_dir / "arbitration_pareto_summary.csv",
        "s3_superiority": out_dir / "arbitration_s3_superiority.csv",
        "workbook": out_dir / "arbitration_results.xlsx",
        "metadata": out_dir / "arbitration_metadata.json",
    }
    long.to_csv(paths["daily_all_strategies"], index=False)
    daily_best.to_csv(paths["daily_selected"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pareto.to_csv(paths["pareto"], index=False)
    s3_df.to_csv(paths["s3_superiority"], index=False)
    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        pd.DataFrame([asdict(weights.normalized())]).to_excel(writer, sheet_name="weights", index=False)
        default_strategy_dataframe().to_excel(writer, sheet_name="default_strategy_bank", index=False)
        summary.to_excel(writer, sheet_name="strategy_summary", index=False)
        pareto.to_excel(writer, sheet_name="pareto_summary", index=False)
        s3_df.to_excel(writer, sheet_name="s3_superiority", index=False)
        daily_best.to_excel(writer, sheet_name="daily_selected_strategy", index=False)
        long.head(10000).to_excel(writer, sheet_name="daily_all_strategies_head", index=False)
    metadata = {
        "clean_path": str(clean_path),
        "degraded_path": str(degraded_path),
        "strategy_bank_path": str(strategy_bank_path) if strategy_bank_path else None,
        "weights": asdict(weights.normalized()),
        "records_compared": int(len(base)),
        "s3_superiority": s3,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths
