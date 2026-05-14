"""
interface_st
------------
Web interface (Streamlit).

Code implementation was assisted by Claude (Anthropic, 2026) under the author's direction. All architectural decisions, scientific choices (attribute grouping strategy, confidence degree definition, aggregation methods), code and algorithmic corrections and validations were made by the author.
"""

import random
import streamlit as st
import pandas as pd
import numpy as np

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

# ── Initialize session state ──────────────────────────────────────────────
for key in ["df", "df_train", "df_test", "GR", "tree_one",
            "groups", "target_col", "selected_methods",
            "use_single", "results_ready"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "obs_idx" not in st.session_state:
    st.session_state["obs_idx"] = 0
if "results_ready" not in st.session_state:
    st.session_state["results_ready"] = False

# ── ZONE 1 : Dataset ──────────────────────────────────────────────────────
st.subheader("1 — Dataset")

col1, col2 = st.columns([1, 1])

with col1:
    source_type = st.radio("Source", ["CSV file", "sklearn dataset"],
                           horizontal=True)

with col2:
    if source_type == "CSV file":
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded:
            st.session_state["df"] = pd.read_csv(uploaded)
    else:
        sklearn_name = st.selectbox(
            "Dataset", list(data_loader.SKLEARN_DATASETS.keys())
        )
        if st.button("Load"):
            st.session_state["df"] = data_loader.load_dataset(sklearn_name)
            st.session_state["results_ready"] = False

df = st.session_state["df"]

if df is not None:
    st.success(
        f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns"
    )

    # ── ZONE 2 : Parameters (appears only after dataset loaded) ───────────
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
        if method_cols[i].checkbox(method, value=True, key=f"cb_{method}"):
            selected_methods.append(method)
    use_single = method_cols[len(AGGREGATION_METHODS)].checkbox(
        "single tree", value=True, key="cb_single"
    )

    # Run button — inside Zone 2 like Tkinter
    if st.button("▶  Run", type="primary"):
        if not selected_methods and not use_single:
            st.error("Please select at least one method.")
        else:
            with st.spinner("Running..."):
                df_train, df_test = data_loader.split_dataset(
                    df, target_col, test_size=test_size
                )
                groups, _ = CO.attribute_grouping(
                    df_train, target_col, threshold=threshold
                )
                trees    = ADCO_EM.train_trees(df_train, groups, target_col)
                GR       = ADCO_EM.build_global_rules(trees)
                tree_one = ADCO_one.train(df_train, target_col)

                # Store in session state
                st.session_state["df_train"]        = df_train
                st.session_state["df_test"]         = df_test
                st.session_state["GR"]              = GR
                st.session_state["tree_one"]        = tree_one
                st.session_state["groups"]          = groups
                st.session_state["target_col"]      = target_col
                st.session_state["selected_methods"] = selected_methods
                st.session_state["use_single"]      = use_single
                st.session_state["results_ready"]   = True
                st.session_state["obs_idx"]         = 0

    # ── ZONE 3 : Results (only if Run was clicked) ────────────────────────
    if st.session_state["results_ready"]:

        df_train   = st.session_state["df_train"]
        df_test    = st.session_state["df_test"]
        GR         = st.session_state["GR"]
        tree_one   = st.session_state["tree_one"]
        groups     = st.session_state["groups"]
        target_col = st.session_state["target_col"]
        sel_methods = st.session_state["selected_methods"]
        use_single  = st.session_state["use_single"]

        st.subheader("3 — Results")
        st.caption(
            f"Train: {len(df_train)} samples  |  "
            f"Test: {len(df_test)} samples"
        )

        tab1, tab2, tab3 = st.tabs([
            "📋 Groups & Rules",
            "📊 Comparative Results",
            "🔍 Observation Test"
        ])

        # ── Tab 1 : Groups & Rules ────────────────────────────────────────
        with tab1:
            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.markdown("**Groups**")
                for i, g in enumerate(groups, 1):
                    st.markdown(f"**G{i}** : `{g}`")

            with right_col:
                st.markdown("**Rules**")
                for rule in GR:
                    conds = " AND ".join(
                        f"{c['feature']} {c['op']} {c['threshold']}"
                        for c in rule["conditions"]
                    )
                    st.markdown(
                        f"`[{rule['group']}]` "
                        f"IF {conds if conds else '(always)'}  \n"
                        f"→ **{rule['class']}** "
                        f"[conf={rule['confidence']:.0%},"
                        f" support={rule['support']}]"
                    )

        # ── Tab 2 : Comparative Results ───────────────────────────────────
        with tab2:
            all_results = {}
            for method in sel_methods:
                all_results[method] = ADCO_confidence.predict_ensemble(
                    GR, df_test, target_col, method=method
                )
            if use_single:
                all_results["single"] = ADCO_one.predict_single(
                    tree_one, df_test, target_col
                )

            methods = list(all_results.keys())
            n = len(df_test)

            # Build DataFrame
            table_data = {
                "#": list(range(1, n + 1)),
                "True class": [
                    all_results[methods[0]][i]["true_class"]
                    for i in range(n)
                ]
            }
            for method in methods:
                table_data[method] = [
                    all_results[method][i]["predicted_class"] or "—"
                    for i in range(n)
                ]

            table_df = pd.DataFrame(table_data)

            # Style : green correct, red incorrect
            def highlight(row):
                styles = [""] * len(row)
                for i, col in enumerate(row.index):
                    if col in methods:
                        true_val = table_df.loc[row.name, "True class"] \
                            if row.name < len(table_df) else None
                        pred = row[col]
                        if true_val and pred != "—":
                            if pred == true_val:
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

            st.dataframe(
                table_df.style.apply(highlight, axis=1),
                use_container_width=True,
                height=420
            )

            # Accuracy summary
            st.markdown("**Accuracy summary**")
            acc_summary = {}
            for method in methods:
                if method == "single":
                    acc = ADCO_one.accuracy_single(all_results[method])
                else:
                    acc = ADCO_confidence.accuracy_ensemble(all_results[method])
                acc_summary[method] = f"{acc:.2%}"

            st.dataframe(
                pd.DataFrame([acc_summary], index=["Accuracy"]),
                use_container_width=True
            )

        # ── Tab 3 : Observation Test ──────────────────────────────────────
        with tab3:
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

            with ctrl_col1:
                obs_idx = st.number_input(
                    "Observation index",
                    min_value=0,
                    max_value=len(df_test) - 1,
                    value=st.session_state["obs_idx"],
                    step=1,
                    key="obs_input"
                )
                st.session_state["obs_idx"] = obs_idx

            with ctrl_col2:
                st.write("")
                st.write("")
                if st.button("🎲 Random"):
                    st.session_state["obs_idx"] = random.randint(
                        0, len(df_test) - 1
                    )
                    st.rerun()

            observation = df_test.iloc[st.session_state["obs_idx"]]
            true_class  = observation[target_col]

            # ── Observation values ────────────────────────────────────────
            st.markdown(
                f"**Observation #{st.session_state['obs_idx']}**"
            )
            feature_cols = [c for c in df_test.columns if c != target_col]
            obs_data = {col: [observation[col]] for col in feature_cols}
            obs_data[f"✦ {target_col} (true)"] = [true_class]
            st.dataframe(pd.DataFrame(obs_data), use_container_width=True)

            # ── Predictions ───────────────────────────────────────────────
            st.markdown("**Predictions**")

            pred_rows = []
            for method in sel_methods:
                predicted, scores, _ = ADCO_confidence.predict_one_ensemble(
                    GR, observation, method=method
                )
                correct = predicted == true_class
                score_str = "  ".join(
                    f"{cls}: {s:.3f}"
                    for cls, s in sorted(scores.items())
                )
                pred_rows.append({
                    "Method":          method,
                    "Predicted class": str(predicted or "—"),
                    "Correct?":        "✔" if correct else "✘",
                    "Scores per class": score_str
                })

            if use_single:
                X_obs = observation[tree_one["group"]]\
                    .values.reshape(1, -1)
                predicted_s = tree_one["tree"].predict(X_obs)[0]
                correct_s   = predicted_s == true_class
                pred_rows.append({
                    "Method":          "single tree",
                    "Predicted class": str(predicted_s),
                    "Correct?":        "✔" if correct_s else "✘",
                    "Scores per class": "—"
                })

            pred_df = pd.DataFrame(pred_rows)

            def highlight_pred(row):
                styles = [""] * len(row)
                correct = row["Correct?"] == "✔"
                for i in range(len(row)):
                    if row.index[i] in ["Predicted class", "Correct?"]:
                        styles[i] = (
                            "background-color: #c8e6c9; color: #1b5e20"
                            if correct else
                            "background-color: #ffcdd2; color: #b71c1c"
                        )
                return styles

            st.dataframe(
                pred_df.style.apply(highlight_pred, axis=1),
                use_container_width=True
            )
