# DAMSAL: Degradation-Aware Multi-Strategy Arbitration Layer

## Purpose

DAMSAL is a post-layer for the HVAC v3 dynamic reduced-order digital twin. It does not assume that S3 is always the best strategy. Instead, it evaluates S0-S6 candidate strategies under the same clean/degraded time-series conditions and selects the best strategy using multi-objective scoring and Pareto analysis.

## Candidate strategy bank

- **S0**: no corrective control.
- **S1**: rule-based passive correction.
- **S2**: schedule-aware active control.
- **S3**: predictive degradation-aware control.
- **S4**: maintenance-triggered control.
- **S5**: robust uncertainty-aware control.
- **S6**: hybrid control-maintenance strategy.

## Core metrics

The layer calculates degradation-induced excess energy:

\[
E_{excess,t}=E_{degraded,t}-E_{clean,t}
\]

and evaluates strategy performance using:

\[
J_i = w_E E_{excess,i} + w_C C_i + w_D D_i + w_M M_i + w_R R_i + w_S S_i
\]

where energy, comfort, degradation, maintenance, robustness, and switching penalties are evaluated for each candidate strategy.

## S3 superiority index

S3 is compared against the best alternative using:

\[
SSI_{S3}=\frac{J_{best\ alternative}-J_{S3}}{J_{best\ alternative}}\times100
\]

If this value is positive, S3 is superior to the nearest competing strategy. If it is negative, another strategy is better.

## How to run

Write the default strategy bank:

```bash
python run_arbitration_layer.py --write_default_strategy_bank --out_dir arbitration_outputs
```

Run arbitration:

```bash
python run_arbitration_layer.py \
  --clean_csv clean_baseline_daily.csv \
  --degraded_csv degraded_scenario_daily.csv \
  --strategy_bank_csv arbitration_outputs/default_strategy_bank.csv \
  --out_dir arbitration_outputs
```

Run the standalone Streamlit arbitration app:

```bash
streamlit run streamlit_arbitration_app.py
```

## Outputs

- `arbitration_daily_all_strategies.csv`
- `arbitration_daily_selected_strategy.csv`
- `arbitration_strategy_summary.csv`
- `arbitration_pareto_summary.csv`
- `arbitration_s3_superiority.csv`
- `arbitration_results.xlsx`

## Scientific framing

DAMSAL converts S3 from an assumed best case into a testable candidate strategy. The contribution is therefore not only a single strategy, but a dynamic decision layer that identifies when predictive control is worthwhile, when simpler strategies are sufficient, and when maintenance-integrated control is required.
