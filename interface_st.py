"""
interface_st.py
---------------
Streamlit web interface for ADCO.
Run with: streamlit run interface_st.py
"""

import streamlit as st
import pandas as pd

import data_loader
import CO
import ADCO_EM
import ADCO_confidence
import ADCO_one
from ADCO_aggregation import AGGREGATION_METHODS

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADCO",
    page_icon="🌿",
    layout="wide"
)

st.title("ADCO")
st.caption("Attribute Grouping & Ensemble Decision Trees")

# ── ZONE 1 : Dataset ──────────────────────────────────────────────────────
st.subheader("1 — Dataset")

col1, col2 = st.columns([1, 1])

with col1:
    source_type = st.radio("Source", ["CSV file", "sklearn dataset"],
                           horizontal=True)

with col2:
    if source_type == "CSV file":
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        df = pd.read_csv(uploaded) if uploaded else None
    else:
        sklearn_name = st.selectbox(
            "Dataset", list(data_loader.SKLEARN_DATASETS.keys())
        )
        df = data_loader.load_dataset(sklearn_name)

if df is not None:
    st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── ZONE 2 : Parameters ───────────────────────────────────────────────
    st.subheader("2 — Parameters")

    col_a, col_b, col_c = st.columns([1, 1, 1])

    with col_a:
        target_col = st.selectbox("Target column", df.columns.tolist(),
                                  index=len(df.columns) - 1)

    with col_b:
        threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.8, 0.05)

    with col_c:
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)

    st.write("**Aggregation methods**")
    method_cols = st.columns(len(AGGREGATION_METHODS) + 1)
    selected_methods = []
    for i, method in enumerate(AGGREGATION_METHODS.keys()):
        if method_cols[i].checkbox(method, value=True):
            selected_methods.append(method)
    use_single = method_cols[len(AGGREGATION_METHODS)].checkbox(
        "single tree", value=True
    )

    # ── Run ───────────────────────────────────────────────────────────────
    if st.button("▶  Run", type="primary"):
        if not selected_methods and not use_single:
            st.error("Please select at least one method.")
        else:
            with st.spinner("Running..."):
                # Split
                df_train, df_test = data_loader.split_dataset(
                    df, target_col, test_size=test_size
                )

                # Grouping
                groups, _ = CO.attribute_grouping(
                    df_train, target_col, threshold=threshold
                )

                # Train ensemble
                trees = ADCO_EM.train_trees(df_train, groups, target_col)
                GR    = ADCO_EM.build_global_rules(trees)

                # Train single tree
                tree_one  = ADCO_one.train(df_train, target_col)

                # ── ZONE 3 : Results ──────────────────────────────────────
                st.subheader("3 — Results")
                st.caption(
                    f"Train: {len(df_train)} samples  |  "
                    f"Test: {len(df_test)} samples"
                )

                left_col, right_col = st.columns([1, 2])

                # Left : groups and rules
                with left_col:
                    st.markdown("**Groups**")
                    for i, g in enumerate(groups, 1):
                        st.markdown(f"**G{i}** : `{g}`")

                    st.markdown("---")
                    st.markdown("**Rules**")
                    for rule in GR:
                        conds = " AND ".join(
                            f"{c['feature']} {c['op']} {c['threshold']}"
                            for c in rule["conditions"]
                        )
                        st.markdown(
                            f"`[{rule['group']}]` IF {conds if conds else '(always)'}  \n"
                            f"→ **{rule['class']}** "
                            f"[{rule['confidence']:.0%}, s={rule['support']}]"
                        )

                # Right : comparative table
                with right_col:
                    st.markdown("**Comparative Results**")

                    # Build results
                    all_results = {}
                    for method in selected_methods:
                        all_results[method] = ADCO_confidence.predict_ensemble(
                            GR, df_test, target_col, method=method
                        )
                    if use_single:
                        all_results["single"] = ADCO_one.predict_single(
                            tree_one, df_test, target_col
                        )

                    methods = list(all_results.keys())
                    n = len(df_test)

                    # Build DataFrame for display
                    table_data = {"#": list(range(1, n + 1)),
                                  "True class": [
                                      all_results[methods[0]][i]["true_class"]
                                      for i in range(n)
                                  ]}

                    for method in methods:
                        table_data[method] = [
                            all_results[method][i]["predicted_class"] or "—"
                            for i in range(n)
                        ]

                    table_df = pd.DataFrame(table_data)

                    # Accuracy row
                    acc_row = {"#": "Accuracy", "True class": ""}
                    for method in methods:
                        if method == "single":
                            acc = ADCO_one.accuracy_single(all_results[method])
                        else:
                            acc = ADCO_confidence.accuracy_ensemble(
                                all_results[method]
                            )
                        acc_row[method] = f"{acc:.2%}"
                    acc_df = pd.DataFrame([acc_row])

                    # Style function : green correct, red incorrect
                    def style_cell(val, col, method):
                        if col not in methods:
                            return ""
                        true_vals = table_df["True class"].tolist()
                        idx = table_df[col].tolist().index(val) \
                            if val in table_df[col].tolist() else -1
                        if idx == -1:
                            return ""
                        correct = val == true_vals[idx]
                        if correct:
                            return "background-color: #c8e6c9; color: #1b5e20"
                        return "background-color: #ffcdd2; color: #b71c1c"

                    # Apply styling
                    def highlight(row):
                        styles = [""] * len(row)
                        for i, col in enumerate(row.index):
                            if col in methods:
                                true_class = table_df.loc[
                                    row.name, "True class"
                                ] if row.name < len(table_df) else None
                                pred = row[col]
                                if true_class and pred != "—":
                                    if pred == true_class:
                                        styles[i] = (
                                            "background-color: #c8e6c9;"
                                            "color: #1b5e20"
                                        )
                                    else:
                                        styles[i] = (
                                            "background-color: #ffcdd2;"
                                            "color: #b71c1c"
                                        )
                        return styles

                    styled = table_df.style.apply(highlight, axis=1)
                    st.dataframe(styled, use_container_width=True, height=400)

                    # Accuracy summary
                    st.markdown("**Accuracy summary**")
                    acc_summary = {}
                    for method in methods:
                        if method == "single":
                            acc = ADCO_one.accuracy_single(all_results[method])
                        else:
                            acc = ADCO_confidence.accuracy_ensemble(
                                all_results[method]
                            )
                        acc_summary[method] = f"{acc:.2%}"

                    st.dataframe(
                        pd.DataFrame([acc_summary], index=["Accuracy"]),
                        use_container_width=True
                    )
