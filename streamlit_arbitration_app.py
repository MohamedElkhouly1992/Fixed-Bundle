"""Streamlit app for DAMSAL arbitration layer."""
from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from arbitration_layer import ArbitrationWeights, run_arbitration, default_strategy_dataframe, read_timeseries, align_clean_degraded, evaluate_strategies, pareto_front, s3_superiority

st.set_page_config(page_title="DAMSAL HVAC Arbitration Layer", layout="wide")
st.title("DAMSAL — Degradation-Aware Multi-Strategy Arbitration Layer")
st.caption("Use clean/degraded outputs from the dynamic HVAC v3 solver to test whether S3 is actually better than alternative control layers.")

with st.expander("Scientific role", expanded=True):
    st.markdown(
        """
This post-layer does **not** change the core solver. It compares candidate strategies S0–S6 using the same clean and degraded time series, then ranks them by energy recovery, comfort, degradation, maintenance cost, robustness, and switching penalty.
        """
    )

clean_file = st.file_uploader("Upload clean baseline CSV/XLSX", type=["csv", "xlsx", "xls"])
degraded_file = st.file_uploader("Upload degraded scenario CSV/XLSX", type=["csv", "xlsx", "xls"])
strategy_file = st.file_uploader("Optional custom strategy bank CSV", type=["csv"])

st.subheader("Objective weights")
cols = st.columns(6)
w_energy = cols[0].number_input("Energy", 0.0, 1.0, 0.45, 0.01)
w_comfort = cols[1].number_input("Comfort", 0.0, 1.0, 0.20, 0.01)
w_deg = cols[2].number_input("Degradation", 0.0, 1.0, 0.15, 0.01)
w_maint = cols[3].number_input("Maintenance", 0.0, 1.0, 0.10, 0.01)
w_rob = cols[4].number_input("Robustness", 0.0, 1.0, 0.07, 0.01)
w_switch = cols[5].number_input("Switching", 0.0, 1.0, 0.03, 0.01)

st.download_button(
    "Download default strategy bank template",
    default_strategy_dataframe().to_csv(index=False).encode("utf-8"),
    file_name="default_strategy_bank.csv",
    mime="text/csv",
)

if clean_file and degraded_file and st.button("Run arbitration", type="primary"):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clean_path = tmp_path / clean_file.name
        deg_path = tmp_path / degraded_file.name
        clean_path.write_bytes(clean_file.getvalue())
        deg_path.write_bytes(degraded_file.getvalue())
        strategy_path = None
        if strategy_file:
            strategy_path = tmp_path / strategy_file.name
            strategy_path.write_bytes(strategy_file.getvalue())
        out_dir = tmp_path / "outputs"
        weights = ArbitrationWeights(w_energy, w_comfort, w_deg, w_maint, w_rob, w_switch)
        paths = run_arbitration(clean_path, deg_path, out_dir, strategy_path, weights)
        summary = pd.read_csv(paths["summary"])
        pareto = pd.read_csv(paths["pareto"])
        s3 = pd.read_csv(paths["s3_superiority"])
        daily = pd.read_csv(paths["daily_selected"])

        st.subheader("Strategy summary")
        st.dataframe(summary, use_container_width=True)
        st.subheader("Pareto status")
        st.dataframe(pareto, use_container_width=True)
        st.subheader("S3 superiority")
        st.dataframe(s3, use_container_width=True)
        st.subheader("Daily selected strategy")
        st.dataframe(daily.head(200), use_container_width=True)

        st.download_button(
            "Download arbitration workbook",
            paths["workbook"].read_bytes(),
            file_name="arbitration_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download daily selected strategy CSV",
            paths["daily_selected"].read_bytes(),
            file_name="arbitration_daily_selected_strategy.csv",
            mime="text/csv",
        )
