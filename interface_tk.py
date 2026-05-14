"""
interface_tk.py
---------------
Tkinter graphical interface for ADCO.
Displays attribute groups on the left and a comparative results table on the right.
Supports multiple aggregation methods via checkboxes.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

import data_loader
import CO
import ADCO_EM
import ADCO_confidence
import ADCO_one
from ADCO_aggregation import AGGREGATION_METHODS


# ── Color palette ─────────────────────────────────────────────────────────
BG          = "#f0f2f5"
DARK        = "#1a1a2e"
WHITE       = "#ffffff"
BORDER      = "#dde1e7"
GREEN       = "#c8e6c9"
GREEN_TXT   = "#1b5e20"
RED         = "#ffcdd2"
RED_TXT     = "#b71c1c"
BLUE        = "#64b5f6"
ORANGE      = "#ffb74d"
GREY_TXT    = "#b0bec5"


class ADCOApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ADCO — Attribute Grouping & Decision Trees")
        self.resizable(True, True)
        self.configure(bg=BG)
        self.geometry("1200x800")

        self._df       = None
        self._df_train = None
        self._df_test  = None
        self._GR       = None
        self._trees    = None
        self._groups   = None

        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title
        tk.Label(self, text="ADCO",
                 font=("Helvetica", 20, "bold"),
                 bg=BG, fg=DARK).pack(pady=(14, 2))
        tk.Label(self, text="Attribute Grouping & Ensemble Decision Trees",
                 font=("Helvetica", 10), bg=BG, fg="#555").pack(pady=(0, 10))

        # ── ZONE 1 : Dataset source ───────────────────────────────────────
        z1 = tk.LabelFrame(self, text="  1 — Dataset  ",
                           font=("Helvetica", 9, "bold"),
                           bg=WHITE, fg=DARK, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        z1.pack(fill="x", padx=20, pady=4)

        self._source_var = tk.StringVar(value="file")

        # CSV file
        tk.Radiobutton(z1, text="CSV file",
                       variable=self._source_var, value="file",
                       command=self._toggle_source,
                       bg=WHITE, font=("Helvetica", 10)
                       ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))

        self._path_var = tk.StringVar()
        self._path_entry = tk.Entry(z1, textvariable=self._path_var,
                                    width=44, font=("Helvetica", 10),
                                    relief="solid", bd=1)
        self._path_entry.grid(row=0, column=1, padx=(0, 6), pady=(8, 2))

        self._browse_btn = tk.Button(z1, text="Browse…",
                                     command=self._browse,
                                     font=("Helvetica", 10), bg="#e8eaf0",
                                     relief="flat", cursor="hand2")
        self._browse_btn.grid(row=0, column=2, padx=(0, 6), pady=(8, 2))

        # sklearn dataset
        tk.Radiobutton(z1, text="sklearn dataset",
                       variable=self._source_var, value="sklearn",
                       command=self._toggle_source,
                       bg=WHITE, font=("Helvetica", 10)
                       ).grid(row=1, column=0, sticky="w", padx=12, pady=(2, 8))

        self._sklearn_var = tk.StringVar()
        self._sklearn_combo = ttk.Combobox(
            z1, textvariable=self._sklearn_var,
            values=list(data_loader.SKLEARN_DATASETS.keys()),
            state="disabled", width=20, font=("Helvetica", 10)
        )
        self._sklearn_combo.grid(row=1, column=1, sticky="w",
                                 padx=(0, 6), pady=(2, 8))

        self._load_btn = tk.Button(z1, text="Load",
                                   command=self._load_dataset,
                                   font=("Helvetica", 10), bg=DARK, fg=WHITE,
                                   relief="flat", cursor="hand2", padx=12)
        self._load_btn.grid(row=1, column=2, padx=(0, 6), pady=(2, 8))

        # Dataset info label
        self._info_var = tk.StringVar(value="")
        tk.Label(z1, textvariable=self._info_var,
                 font=("Helvetica", 9), bg=WHITE, fg="#888"
                 ).grid(row=0, column=3, padx=12)

        # ── ZONE 2 : Parameters (hidden until dataset loaded) ─────────────
        self._z2 = tk.LabelFrame(self, text="  2 — Parameters  ",
                                  font=("Helvetica", 9, "bold"),
                                  bg=WHITE, fg=DARK, relief="flat",
                                  highlightbackground=BORDER, highlightthickness=1)

        pad = {"padx": 12, "pady": 5}

        # Target column
        tk.Label(self._z2, text="Target column",
                 font=("Helvetica", 10, "bold"),
                 bg=WHITE).grid(row=0, column=0, sticky="w", **pad)

        self._label_var = tk.StringVar()
        self._label_combo = ttk.Combobox(
            self._z2, textvariable=self._label_var,
            state="readonly", width=26, font=("Helvetica", 10)
        )
        self._label_combo.grid(row=0, column=1, sticky="w", **pad)

        # Threshold
        tk.Label(self._z2, text="Correlation threshold (0–1)",
                 font=("Helvetica", 10, "bold"),
                 bg=WHITE).grid(row=1, column=0, sticky="w", **pad)
        self._thresh_var = tk.StringVar(value="0.8")
        tk.Entry(self._z2, textvariable=self._thresh_var,
                 width=8, font=("Helvetica", 10), relief="solid", bd=1
                 ).grid(row=1, column=1, sticky="w", **pad)

        # Test size
        tk.Label(self._z2, text="Test set size (0–1)",
                 font=("Helvetica", 10, "bold"),
                 bg=WHITE).grid(row=2, column=0, sticky="w", **pad)
        self._testsize_var = tk.StringVar(value="0.2")
        tk.Entry(self._z2, textvariable=self._testsize_var,
                 width=8, font=("Helvetica", 10), relief="solid", bd=1
                 ).grid(row=2, column=1, sticky="w", **pad)

        # Aggregation methods checkboxes
        tk.Label(self._z2, text="Aggregation methods",
                 font=("Helvetica", 10, "bold"),
                 bg=WHITE).grid(row=3, column=0, sticky="w", **pad)

        cb_frame = tk.Frame(self._z2, bg=WHITE)
        cb_frame.grid(row=3, column=1, sticky="w", **pad)

        self._method_vars = {}
        for i, method in enumerate(AGGREGATION_METHODS.keys()):
            var = tk.BooleanVar(value=True)
            self._method_vars[method] = var
            tk.Checkbutton(cb_frame, text=method, variable=var,
                           bg=WHITE, font=("Helvetica", 10)
                           ).grid(row=0, column=i, padx=6)

        # Single tree checkbox
        self._single_var = tk.BooleanVar(value=True)
        tk.Checkbutton(cb_frame, text="single tree",
                       variable=self._single_var,
                       bg=WHITE, font=("Helvetica", 10)
                       ).grid(row=0, column=len(AGGREGATION_METHODS), padx=6)

        # Run button — inside zone 2, appears with it
        tk.Button(self._z2, text="▶  Run", command=self._run,
                  font=("Helvetica", 11, "bold"), bg=DARK, fg=WHITE,
                  relief="flat", cursor="hand2", padx=20, pady=6
                  ).grid(row=4, column=0, columnspan=2,
                         pady=(8, 10), padx=12, sticky="w")

        # ── ZONE 3 : Results ──────────────────────────────────────────────
        self._z3 = tk.LabelFrame(self, text="  3 — Results  ",
                                  font=("Helvetica", 9, "bold"),
                                  bg=WHITE, fg=DARK, relief="flat",
                                  highlightbackground=BORDER, highlightthickness=1)

        # Notebook (tabs)
        style = ttk.Style()
        style.configure("TNotebook", background=WHITE)
        style.configure("TNotebook.Tab", font=("Helvetica", 10, "bold"),
                        padding=[12, 4])

        self._notebook = ttk.Notebook(self._z3)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Tab 1 : Groups & Rules ────────────────────────────────────────
        tab1 = tk.Frame(self._notebook, bg=WHITE)
        self._notebook.add(tab1, text="Groups & Rules")

        left = tk.Frame(tab1, bg=WHITE)
        left.pack(fill="both", expand=True, padx=4, pady=4)

        sc_left = tk.Scrollbar(left)
        sc_left.pack(side="right", fill="y")
        self._groups_text = tk.Text(left, font=("Courier", 9),
                                    yscrollcommand=sc_left.set,
                                    bg=DARK, fg="#e8f4f8",
                                    relief="flat", padx=8, pady=6,
                                    wrap="none")
        self._groups_text.pack(fill="both", expand=True)
        sc_left.config(command=self._groups_text.yview)
        self._groups_text.tag_config("header", foreground=BLUE,
                                     font=("Courier", 9, "bold"))
        self._groups_text.tag_config("group",  foreground="#81c784",
                                     font=("Courier", 9, "bold"))
        self._groups_text.tag_config("rule",   foreground="#fff9c4")
        self._groups_text.tag_config("conf",   foreground=ORANGE)

        # ── Tab 2 : Comparative Results ───────────────────────────────────
        tab2 = tk.Frame(self._notebook, bg=WHITE)
        self._notebook.add(tab2, text="Comparative Results")

        table_container = tk.Frame(tab2, bg=WHITE)
        table_container.pack(fill="both", expand=True)

        sc_y = tk.Scrollbar(table_container, orient="vertical")
        sc_y.pack(side="right", fill="y")
        sc_x = tk.Scrollbar(table_container, orient="horizontal")
        sc_x.pack(side="bottom", fill="x")

        self._canvas = tk.Canvas(table_container, bg=WHITE,
                                 yscrollcommand=sc_y.set,
                                 xscrollcommand=sc_x.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        sc_y.config(command=self._canvas.yview)
        sc_x.config(command=self._canvas.xview)

        self._table_frame = tk.Frame(self._canvas, bg=WHITE)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._table_frame, anchor="nw"
        )
        self._table_frame.bind("<Configure>", self._on_frame_configure)

        # ── Tab 3 : Single Observation Test ──────────────────────────────
        tab3 = tk.Frame(self._notebook, bg=WHITE)
        self._notebook.add(tab3, text="Observation Test")

        # Controls
        ctrl = tk.Frame(tab3, bg=WHITE)
        ctrl.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(ctrl, text="Observation index :",
                 font=("Helvetica", 10, "bold"),
                 bg=WHITE).grid(row=0, column=0, sticky="w", padx=(0, 6))

        self._obs_idx_var = tk.StringVar(value="0")
        self._obs_entry = tk.Entry(ctrl, textvariable=self._obs_idx_var,
                                   width=8, font=("Helvetica", 10),
                                   relief="solid", bd=1)
        self._obs_entry.grid(row=0, column=1, padx=(0, 10))

        tk.Button(ctrl, text="Random",
                  command=self._random_observation,
                  font=("Helvetica", 10), bg="#e8eaf0",
                  relief="flat", cursor="hand2"
                  ).grid(row=0, column=2, padx=(0, 10))

        tk.Button(ctrl, text="Test this observation",
                  command=self._test_observation,
                  font=("Helvetica", 10, "bold"), bg=DARK, fg=WHITE,
                  relief="flat", cursor="hand2", padx=12
                  ).grid(row=0, column=3)

        # Observation display (scrollable)
        obs_container = tk.Frame(tab3, bg=WHITE)
        obs_container.pack(fill="both", expand=True, padx=12, pady=4)

        sc_obs_y = tk.Scrollbar(obs_container, orient="vertical")
        sc_obs_y.pack(side="right", fill="y")
        sc_obs_x = tk.Scrollbar(obs_container, orient="horizontal")
        sc_obs_x.pack(side="bottom", fill="x")

        self._obs_canvas = tk.Canvas(obs_container, bg=WHITE,
                                     yscrollcommand=sc_obs_y.set,
                                     xscrollcommand=sc_obs_x.set)
        self._obs_canvas.pack(side="left", fill="both", expand=True)
        sc_obs_y.config(command=self._obs_canvas.yview)
        sc_obs_x.config(command=self._obs_canvas.xview)

        self._obs_frame = tk.Frame(self._obs_canvas, bg=WHITE)
        self._obs_canvas.create_window((0, 0), window=self._obs_frame,
                                       anchor="nw")
        self._obs_frame.bind("<Configure>",
                             lambda e: self._obs_canvas.config(
                                 scrollregion=self._obs_canvas.bbox("all")
                             ))

        self._toggle_source()

    # ── Toggle source ─────────────────────────────────────────────────────
    def _toggle_source(self):
        if self._source_var.get() == "file":
            self._path_entry.config(state="normal")
            self._browse_btn.config(state="normal")
            self._sklearn_combo.config(state="disabled")
        else:
            self._path_entry.config(state="disabled")
            self._browse_btn.config(state="disabled")
            self._sklearn_combo.config(state="readonly")

    # ── Browse CSV ────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Choose a CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self._path_var.set(path)

    # ── Load dataset ──────────────────────────────────────────────────────
    def _load_dataset(self):
        source = (self._path_var.get().strip()
                  if self._source_var.get() == "file"
                  else self._sklearn_var.get().strip())

        if not source:
            messagebox.showwarning("Warning", "Please select a file or dataset.")
            return

        try:
            self._df = data_loader.load_dataset(source)
            cols = list(self._df.columns)
            self._label_combo["values"] = cols
            self._label_var.set(cols[-1])
            self._info_var.set(
                f"{self._df.shape[0]} rows × {self._df.shape[1]} columns"
            )
            # Show zones 2 and 3
            self._z2.pack(fill="x", padx=20, pady=4)
            self._z3.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        except Exception as e:
            messagebox.showerror("Error", f"Could not load dataset:\n{e}")

    # ── Run ───────────────────────────────────────────────────────────────
    def _run(self):
        # Validate inputs
        target_col = self._label_var.get()
        if not target_col:
            messagebox.showwarning("Warning", "Please select the target column.")
            return

        try:
            threshold = float(self._thresh_var.get())
            assert 0.0 <= threshold <= 1.0
        except Exception:
            messagebox.showerror("Error", "Threshold must be between 0 and 1.")
            return

        try:
            test_size = float(self._testsize_var.get())
            assert 0.0 < test_size < 1.0
        except Exception:
            messagebox.showerror("Error", "Test size must be between 0 and 1.")
            return

        selected_methods = [m for m, v in self._method_vars.items() if v.get()]
        use_single = self._single_var.get()

        if not selected_methods and not use_single:
            messagebox.showwarning("Warning", "Please select at least one method.")
            return

        # Split
        self._df_train, self._df_test = data_loader.split_dataset(
            self._df, target_col, test_size=test_size
        )

        # Grouping
        self._groups, _ = CO.attribute_grouping(
            self._df_train, target_col, threshold=threshold
        )

        # Train ensemble trees
        self._trees = ADCO_EM.train_trees(
            self._df_train, self._groups, target_col
        )
        self._GR = ADCO_EM.build_global_rules(self._trees)

        # Train single tree
        self._tree_one = ADCO_one.train(self._df_train, target_col)
        self._rules_one = ADCO_one.extract_rules_single(self._tree_one)

        # ── Left panel : groups and rules ─────────────────────────────────
        self._groups_text.delete("1.0", tk.END)
        self._groups_text.insert(tk.END,
            f"Train: {len(self._df_train)}  Test: {len(self._df_test)}\n\n",
            "header"
        )
        self._groups_text.insert(tk.END,
            f"── Groups (threshold={threshold}) ──\n\n", "header"
        )
        for i, g in enumerate(self._groups, 1):
            self._groups_text.insert(tk.END, f"G{i} : {g}\n", "group")

        self._groups_text.insert(tk.END, "\n── Rules ──\n\n", "header")
        for rule in self._GR:
            conds = " AND ".join(
                f"{c['feature']} {c['op']} {c['threshold']}"
                for c in rule["conditions"]
            )
            self._groups_text.insert(
                tk.END,
                f"[{rule['group']}] IF {conds if conds else '(always)'}\n",
                "rule"
            )
            self._groups_text.insert(
                tk.END,
                f"  → {rule['class']}  [{rule['confidence']:.0%},"
                f" s={rule['support']}]\n",
                "conf"
            )

        # ── Right panel : comparative table ───────────────────────────────
        for w in self._table_frame.winfo_children():
            w.destroy()

        # Build results per method
        all_results = {}
        for method in selected_methods:
            all_results[method] = ADCO_confidence.predict_ensemble(
                self._GR, self._df_test, target_col, method=method
            )
        if use_single:
            all_results["single"] = ADCO_one.predict_single(
                self._tree_one, self._df_test, target_col
            )

        methods = list(all_results.keys())
        n = len(self._df_test)

        # Header row
        headers = ["#", "True class"] + methods
        for col, h in enumerate(headers):
            tk.Label(self._table_frame,
                     text=h, font=("Helvetica", 9, "bold"),
                     bg=DARK, fg=WHITE,
                     padx=8, pady=4, relief="flat"
                     ).grid(row=0, column=col, sticky="nsew", padx=1, pady=1)

        # Data rows
        for row_idx in range(n):
            bg_row = WHITE if row_idx % 2 == 0 else "#f7f7f7"

            # Index
            tk.Label(self._table_frame,
                     text=str(row_idx + 1),
                     font=("Helvetica", 9), bg=bg_row, fg="#888",
                     padx=6, pady=3
                     ).grid(row=row_idx + 1, column=0,
                            sticky="nsew", padx=1, pady=1)

            # True class
            true_class = all_results[methods[0]][row_idx]["true_class"]
            tk.Label(self._table_frame,
                     text=str(true_class),
                     font=("Helvetica", 9, "bold"), bg=bg_row, fg=DARK,
                     padx=8, pady=3
                     ).grid(row=row_idx + 1, column=1,
                            sticky="nsew", padx=1, pady=1)

            # Predicted class per method
            for col_idx, method in enumerate(methods):
                r = all_results[method][row_idx]
                pred  = r["predicted_class"] or "—"
                correct = r["correct"]
                cell_bg  = GREEN if correct else RED
                cell_fg  = GREEN_TXT if correct else RED_TXT

                tk.Label(self._table_frame,
                         text=str(pred),
                         font=("Helvetica", 9), bg=cell_bg, fg=cell_fg,
                         padx=8, pady=3
                         ).grid(row=row_idx + 1, column=col_idx + 2,
                                sticky="nsew", padx=1, pady=1)

        # Accuracy row
        acc_row = n + 1
        tk.Label(self._table_frame,
                 text="Accuracy", font=("Helvetica", 9, "bold"),
                 bg=DARK, fg=WHITE, padx=8, pady=4
                 ).grid(row=acc_row, column=0, columnspan=2,
                        sticky="nsew", padx=1, pady=1)

        for col_idx, method in enumerate(methods):
            if method == "single":
                acc = ADCO_one.accuracy_single(all_results[method])
            else:
                acc = ADCO_confidence.accuracy_ensemble(all_results[method])
            tk.Label(self._table_frame,
                     text=f"{acc:.2%}",
                     font=("Helvetica", 9, "bold"),
                     bg=DARK, fg=ORANGE,
                     padx=8, pady=4
                     ).grid(row=acc_row, column=col_idx + 2,
                            sticky="nsew", padx=1, pady=1)

        self._canvas.update_idletasks()
        self._canvas.config(
            scrollregion=self._canvas.bbox("all")
        )

    # ── Canvas resize ─────────────────────────────────────────────────────
    def _on_frame_configure(self, event):
        self._canvas.config(scrollregion=self._canvas.bbox("all"))


# ─────────────────────────────────────────────────────────────────────────────
    # ── Random observation ────────────────────────────────────────────────
    def _random_observation(self):
        if self._df_test is None:
            messagebox.showwarning("Warning", "Please run the analysis first.")
            return
        import random
        idx = random.randint(0, len(self._df_test) - 1)
        self._obs_idx_var.set(str(idx))
        self._test_observation()

    # ── Test a single observation ─────────────────────────────────────────
    def _test_observation(self):
        if self._df_test is None:
            messagebox.showwarning("Warning", "Please run the analysis first.")
            return

        try:
            idx = int(self._obs_idx_var.get())
            if not (0 <= idx < len(self._df_test)):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Error",
                f"Index must be between 0 and {len(self._df_test) - 1}."
            )
            return

        target_col = self._label_var.get()
        observation = self._df_test.iloc[idx]
        true_class  = observation[target_col]
        selected_methods = [m for m, v in self._method_vars.items() if v.get()]
        use_single = self._single_var.get()

        for w in self._obs_frame.winfo_children():
            w.destroy()

        pad = {"padx": 2, "pady": 2}

        # ── Observation values ────────────────────────────────────────────
        tk.Label(self._obs_frame,
                 text=f"Observation #{idx}",
                 font=("Helvetica", 10, "bold"),
                 bg=WHITE, fg=DARK).grid(
            row=0, column=0, columnspan=100, sticky="w", padx=8, pady=(8, 4)
        )

        feature_cols = [c for c in self._df_test.columns if c != target_col]
        all_cols = feature_cols + [target_col]

        for col_i, col in enumerate(all_cols):
            is_target = col == target_col
            tk.Label(self._obs_frame,
                     text=col,
                     font=("Helvetica", 9, "bold"),
                     bg=DARK, fg=WHITE if not is_target else ORANGE,
                     padx=8, pady=4
                     ).grid(row=1, column=col_i, sticky="nsew", **pad)

        for col_i, col in enumerate(all_cols):
            val = observation[col]
            is_target = col == target_col
            tk.Label(self._obs_frame,
                     text=str(val),
                     font=("Helvetica", 9, "bold" if is_target else "normal"),
                     bg=WHITE, fg=DARK if not is_target else ORANGE,
                     padx=8, pady=4
                     ).grid(row=2, column=col_i, sticky="nsew", **pad)

        # ── Predictions ───────────────────────────────────────────────────
        tk.Label(self._obs_frame,
                 text="Predictions",
                 font=("Helvetica", 10, "bold"),
                 bg=WHITE, fg=DARK).grid(
            row=3, column=0, columnspan=100,
            sticky="w", padx=8, pady=(16, 4)
        )

        for col_i, h in enumerate(["Method", "Predicted class", "Correct?", "Scores"]):
            tk.Label(self._obs_frame,
                     text=h, font=("Helvetica", 9, "bold"),
                     bg=DARK, fg=WHITE, padx=8, pady=4
                     ).grid(row=4, column=col_i, sticky="nsew", **pad)

        row_i = 5
        for method in selected_methods:
            predicted, scores, _ = ADCO_confidence.predict_one_ensemble(
                self._GR, observation, method=method
            )
            correct  = predicted == true_class
            cell_bg  = GREEN if correct else RED
            cell_fg  = GREEN_TXT if correct else RED_TXT
            score_str = "  ".join(
                f"{cls}: {s:.3f}" for cls, s in sorted(scores.items())
            )
            tk.Label(self._obs_frame, text=method,
                     font=("Helvetica", 9, "bold"),
                     bg=WHITE, fg=DARK, padx=8, pady=4
                     ).grid(row=row_i, column=0, sticky="nsew", **pad)
            tk.Label(self._obs_frame, text=str(predicted or "—"),
                     font=("Helvetica", 9), bg=cell_bg, fg=cell_fg,
                     padx=8, pady=4
                     ).grid(row=row_i, column=1, sticky="nsew", **pad)
            tk.Label(self._obs_frame,
                     text="✔" if correct else "✘",
                     font=("Helvetica", 9, "bold"),
                     bg=cell_bg, fg=cell_fg, padx=8, pady=4
                     ).grid(row=row_i, column=2, sticky="nsew", **pad)
            tk.Label(self._obs_frame, text=score_str,
                     font=("Courier", 9), bg=WHITE, fg="#555",
                     padx=8, pady=4
                     ).grid(row=row_i, column=3, sticky="nsew", **pad)
            row_i += 1

        if use_single:
            X_obs = observation[self._tree_one["group"]].values.reshape(1, -1)
            predicted_s = self._tree_one["tree"].predict(X_obs)[0]
            correct_s = predicted_s == true_class
            cell_bg   = GREEN if correct_s else RED
            cell_fg   = GREEN_TXT if correct_s else RED_TXT

            tk.Label(self._obs_frame, text="single tree",
                     font=("Helvetica", 9, "bold"),
                     bg=WHITE, fg=DARK, padx=8, pady=4
                     ).grid(row=row_i, column=0, sticky="nsew", **pad)
            tk.Label(self._obs_frame, text=str(predicted_s),
                     font=("Helvetica", 9), bg=cell_bg, fg=cell_fg,
                     padx=8, pady=4
                     ).grid(row=row_i, column=1, sticky="nsew", **pad)
            tk.Label(self._obs_frame,
                     text="✔" if correct_s else "✘",
                     font=("Helvetica", 9, "bold"),
                     bg=cell_bg, fg=cell_fg, padx=8, pady=4
                     ).grid(row=row_i, column=2, sticky="nsew", **pad)
            tk.Label(self._obs_frame, text="—",
                     font=("Courier", 9), bg=WHITE, fg="#555",
                     padx=8, pady=4
                     ).grid(row=row_i, column=3, sticky="nsew", **pad)

        self._obs_canvas.update_idletasks()
        self._obs_canvas.config(scrollregion=self._obs_canvas.bbox("all"))
        self._notebook.select(2)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ADCOApp()
    app.mainloop()
