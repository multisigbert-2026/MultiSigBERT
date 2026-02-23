import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torchtuples as tt

from pycox.models import CoxTime, CoxPH
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv

from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.utils import concordance_index


def parse_embedding(x):
    """
    Parse one embedding that can be:
    - a numpy array
    - a Python list
    - a string representation like "[ 1.2e-3 4.5e-2 ... ]" (possibly with newlines)
    - a string with commas
    Returns a 1D np.ndarray of floats.
    """
    if isinstance(x, np.ndarray):
        return x.astype(float).ravel()
    if isinstance(x, list) or isinstance(x, tuple):
        return np.asarray(x, dtype=float).ravel()
    if isinstance(x, str):
        s = x.strip()

        # Remove surrounding quotes if they exist (rare but happens)
        if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ["'", '"']):
            s = s[1:-1].strip()

        # Remove brackets if present
        s = s.replace("[", " ").replace("]", " ")

        # Normalize separators: commas -> spaces, newlines/tabs -> spaces
        s = s.replace(",", " ")
        s = re.sub(r"\s+", " ", s).strip()

        # Now parse floats
        if s == "":
            raise ValueError("Empty embedding string after cleaning.")
        return np.fromstring(s, sep=" ", dtype=float).ravel()

    raise TypeError(f"Unsupported embedding type: {type(x)}")


def expand_embedding_column(
    df,
    embedding_column,
    parse_embedding_fn,
    seq_structured_columns=None,
    reports_only=False,
    drop_original=True,
    verbose=True
):
    """
    Parse and expand an embedding column into numerical features and
    construct the list of time-dependent covariates.
    """

    if embedding_column not in df.columns:
        raise ValueError(f"Column '{embedding_column}' not found in DataFrame.")

    df_work = df.copy()

    # ------------------------------------------------------------------
    # Parse embeddings
    # ------------------------------------------------------------------
    emb_parsed = df_work[embedding_column].apply(parse_embedding_fn)

    dims = emb_parsed.apply(len)
    if dims.nunique() != 1:
        raise ValueError(
            f"Inconsistent embedding dimensions found: "
            f"{sorted(dims.unique())[:10]}"
        )

    q_emb = int(dims.iloc[0])

    emb_matrix = np.vstack(emb_parsed.values)
    emb_col_names = [f"emb_{j}" for j in range(q_emb)]

    df_emb = pd.DataFrame(
        emb_matrix,
        columns=emb_col_names,
        index=df_work.index
    )

    # ------------------------------------------------------------------
    # Build expanded DataFrame
    # ------------------------------------------------------------------
    if drop_original:
        df_expanded = df_work.drop(columns=[embedding_column])
    else:
        df_expanded = df_work

    # Add embedding columns
    df_expanded = pd.concat([df_expanded, df_emb], axis=1)

    # ------------------------------------------------------------------
    # Structured columns check (DO NOT TOUCH THEM)
    # ------------------------------------------------------------------
    structured_cols = []

    if not reports_only and seq_structured_columns:
        structured_cols = list(seq_structured_columns)

        missing_cols = [
            c for c in structured_cols if c not in df_expanded.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Structured columns missing from DataFrame: {missing_cols}"
            )

    # ------------------------------------------------------------------
    # Time-dependent covariates
    # ------------------------------------------------------------------
    if reports_only:
        time_covariates_name = emb_col_names
        q_total = q_emb
    else:
        time_covariates_name = emb_col_names + structured_cols
        q_total = q_emb + len(structured_cols)

    if verbose:
        print(f"Embedding dimension: {q_emb}")
        print(f"Structured dimension: {len(structured_cols)}")
        print(f"Total covariate dimension: {q_total}")

    return df_expanded, time_covariates_name, q_total




def interpolate_structured_sequences(
    df,
    structured_columns,
    patient_id_col,
    time_col,
    interpolation_type="linear",
    verbose=True
):
    """
    Interpolate and standardize structured sequential covariates per patient.

    This function performs:
    - Numeric coercion of structured variables
    - Intra-patient interpolation (linear or zero-fill)
    - Edge-value extension (bfill/ffill for linear mode)
    - Fallback fill with zeros if entire patient path is missing
    - Global standardization
    - Optional missing indicator encoding

    Parameters
    ----------
    df : pandas.DataFrame
        Input longitudinal dataset.
    structured_columns : list
        List of structured sequential variable names.
    patient_id_col : str
        Column identifying patients.
    time_col : str
        Temporal ordering column.
    interpolation_type : str, optional
        "linear" or "zeros".
    verbose : bool, optional
        If True, prints diagnostic information.

    Returns
    -------
    df_processed : pandas.DataFrame
        DataFrame with interpolated and standardized structured variables.
    structured_columns_out : list
        Final list of structured covariate names (including indicators if any).
    """

    df_processed = df.copy()
    structured_columns_out = structured_columns.copy()

    # ------------------------------------------------------------------
    # Check column existence
    # ------------------------------------------------------------------
    missing_cols = [c for c in structured_columns if c not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Missing structured variables in df: {missing_cols}")

    # ------------------------------------------------------------------
    # Ensure numeric format
    # ------------------------------------------------------------------
    df_processed[structured_columns] = df_processed[structured_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    if interpolation_type == "linear":

        # Sort within patient by time
        sorted_idx = df_processed.sort_values(
            [patient_id_col, time_col]
        ).index

        block = df_processed.loc[sorted_idx, structured_columns]

        block = (
            block.groupby(df_processed.loc[sorted_idx, patient_id_col], group_keys=False)
                 .apply(lambda g: g.interpolate(method="linear").bfill().ffill())
        )

        df_processed.loc[sorted_idx, structured_columns] = block.values

        # Fallback: if entire trajectory missing → fill with 0
        df_processed[structured_columns] = (
            df_processed
            .groupby(patient_id_col, group_keys=False)[structured_columns]
            .apply(lambda g: g.fillna(0.0))
        )

    elif interpolation_type == "zeros":
        df_processed[structured_columns] = df_processed[structured_columns].fillna(0.0)

    else:
        raise ValueError(f"Unknown interpolation mode: {interpolation_type}")

    # ------------------------------------------------------------------
    # Standardization (global)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    df_processed[structured_columns] = scaler.fit_transform(
        df_processed[structured_columns]
    )


    if verbose:
        print(f"Number of patients after interpolation: "
              f"{df_processed[patient_id_col].nunique()}")
        print(f"Number of structured components: {len(structured_columns_out)}")

    return df_processed, structured_columns_out





#######################################################
#                                                     #
#                Naive Mean DeepSurv                  #
#                                                     #
#######################################################


def make_train_test_agg(
    df,
    var_id="ID",
    n_group=10,
    size_test=0.5,
    random_state=177,
    verbose=False
):
    """
    Split an aggregated patient-level DataFrame (one row per ID)
    into one training set and multiple independent test groups.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated dataset (one row per patient).
    var_id : str
        Patient identifier column.
    n_group : int
        Number of independent test groups.
    size_test : float
        Proportion of patients assigned to the test set.
    random_state : int
        Random seed.
    verbose : bool
        Whether to print split statistics.

    Returns
    -------
    df_train : pd.DataFrame
    test_groups : list of pd.DataFrame
    """

    if df[var_id].duplicated().any():
        raise ValueError("Input DataFrame must contain one row per patient.")

    unique_ids = df[var_id].values

    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=size_test,
        random_state=random_state
    )

    df_train = df[df[var_id].isin(train_ids)]
    df_test = df[df[var_id].isin(test_ids)]

    test_id_splits = np.array_split(test_ids, n_group)

    test_groups = [
        df_test[df_test[var_id].isin(split)]
        for split in test_id_splits
    ]

    if verbose:
        print(f"Train patients: {df_train[var_id].nunique()}")
        for i, group in enumerate(test_groups, 1):
            print(f"Test group {i}: {group[var_id].nunique()} patients")

    return df_train, test_groups




def aggregate_patient_mean(df, seq_struc_col=None, id_col="ID", time_col="R", event_col="DEATH_L"):
    """
    Aggregate longitudinal data into patient-level static representation
    using mean pooling of embedding coordinates.
    """
    if seq_struc_col:
        cols = [c for c in df.columns if c.startswith("emb_")] + seq_struc_col 
    else:
        cols = [c for c in df.columns if c.startswith("emb_")]
        
    df_agg = (
        df
        .groupby(id_col)
        .agg(
            {**{col: "mean" for col in cols},
             time_col: "max",        # survival time
             event_col: "max"       # event indicator
        }
        )
        .reset_index()
    )
    return df_agg




# -----------------------------
# 2) Helpers
# -----------------------------
def build_xy_from_df(df, time_col="T_days", event_col="DEATH_L", drop_cols=None):
    """
    Build (X, (durations, events)) from a patient-level aggregated dataframe.
    Assumes all remaining columns (after dropping) are numeric covariates.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
        Duration column.
    event_col : str
        Event indicator column (0/1).
    drop_cols : list[str] or None
        Additional columns to drop from X (IDs, dates, etc.).

    Returns
    -------
    X : np.ndarray (float32)
    y : tuple[np.ndarray, np.ndarray]
        (durations, events)
    x_cols : list[str]
        Covariate column names used in X.
    """
    if drop_cols is None:
        drop_cols = []

    base_drop = [time_col, event_col] + drop_cols
    base_drop = [c for c in base_drop if c in df.columns]

    x_cols = [c for c in df.columns if c not in base_drop]

    # Keep only numeric columns (safety if df contains stray object columns)
    X_df = df[x_cols].copy()
    non_numeric = [c for c in x_cols if not pd.api.types.is_numeric_dtype(X_df[c])]
    if len(non_numeric) > 0:
        # Drop non-numeric silently (or raise if you prefer strictness)
        X_df = X_df.drop(columns=non_numeric)
        x_cols = [c for c in x_cols if c not in non_numeric]

    X = X_df.values.astype(np.float32)
    durations = df[time_col].values
    events = df[event_col].values

    return X, (durations, events), x_cols


def fit_deepsurv_coxph(X_train, y_train, lr=1e-3, num_nodes=(64, 32),
                       batch_norm=True, dropout=0.3, batch_size=256,
                       epochs=200, seed=42):
    """
    Fit a CoxPH (DeepSurv) model in pycox without validation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_features = X_train.shape[1]
    net = tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=list(num_nodes),
        out_features=1,
        batch_norm=batch_norm,
        dropout=dropout,
        output_bias=False
    )

    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(lr)

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=False
    )

    # Needed for survival curves, not needed for risk-only C-index,
    # but kept to mirror the official example style.
    model.compute_baseline_hazards()
    return model


def cindex_lifelines_from_risk(y, risk):
    """
    Compute C-index using lifelines with the correct sign convention.
    lifelines expects: higher score -> longer survival,
    while CoxPH outputs: higher score -> higher risk -> shorter survival.
    Hence the minus sign.
    """
    durations, events = y
    return concordance_index(durations, -risk.reshape(-1), events)



def global_mean_deepsurv(
    df_train,
    test_groups,
    time_col="R",
    event_col="DEATH_L",
    seq_struc_col=None,
    lr=1e-1,
    num_nodes=(64, 32, 16),
    batch_norm=True,
    dropout=0.75,
    batch_size=64,
    epochs=200,
    seed=177,
    verbose=True,
    ibs_grid_size=100,
):
    """
    Train and evaluate a DeepSurv (CoxPH) model on aggregated patient-level data.

    By default, uses only embedding coordinates as covariates (columns starting with "emb_").
    If `seq_struc_col` is provided, these structured covariates are appended to the embedding
    coordinates and used jointly as model inputs.

    Metrics:
    - C-index (lifelines) on train
    - C-index (lifelines) on each independent test set + mean/std
    - Integrated Brier Score (pycox EvalSurv) on each independent test set + mean/std

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe (patient-level aggregated).
    test_groups : list[pd.DataFrame]
        List of independent test dataframes (patient-level aggregated), typically length 10.
    time_col : str, default="R"
        Duration column name.
    event_col : str, default="DEATH_L"
        Event indicator column name (0/1).
    seq_struc_col : list[str] or None, default=None
        Optional list of additional structured covariate column names to include alongside
        the embedding coordinates.
    lr : float, default=1e-1
        Learning rate.
    num_nodes : tuple[int], default=(64, 32, 16)
        Hidden layer widths.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    dropout : float, default=0.75
        Dropout probability.
    batch_size : int, default=64
        Mini-batch size.
    epochs : int, default=200
        Number of training epochs.
    seed : int, default=177
        Random seed for reproducibility.
    verbose : bool, default=True
        If True, prints per-split and summary metrics.
    ibs_grid_size : int, default=100
        Number of points in the time grid used to compute IBS per test set.

    Returns
    -------
    results : dict
        Dictionary with:
        - "model": trained pycox CoxPH model
        - "scaler": fitted StandardScaler
        - "x_cols": covariate names used in X
        - "c_index_train": float
        - "c_index_tests": list[float]
        - "mean_c_index_test": float
        - "std_c_index_test": float
        - "ibs_tests": list[float]
        - "mean_ibs_test": float
        - "std_ibs_test": float
    """
    # -----------------------------
    # Reproducibility
    # -----------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------------
    # Select covariate columns
    # -----------------------------
    emb_cols = [c for c in df_train.columns if c.startswith("emb_")]

    if seq_struc_col:
        missing = [c for c in seq_struc_col if c not in df_train.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing columns in df_train: {missing}")
        cols = emb_cols + list(seq_struc_col)
    else:
        cols = emb_cols

    if len(cols) == 0:
        raise ValueError("No covariate columns selected. Check that 'emb_' columns exist.")

    # -----------------------------
    # Build X/y (train)
    # -----------------------------
    X_train_raw = df_train[cols].values
    y_train = (
        df_train[time_col].values,
        df_train[event_col].values,
    )
    x_cols = cols

    # -----------------------------
    # Build X/y (tests)
    # -----------------------------
    X_test_raw_list = []
    y_test_list = []

    for df_test in test_groups:
        missing_test = [c for c in cols if c not in df_test.columns]
        if len(missing_test) > 0:
            raise ValueError(f"Missing columns in a test group: {missing_test}")

        X_test_raw_list.append(df_test[cols].values)
        y_test_list.append((
            df_test[time_col].values,
            df_test[event_col].values,
        ))

    # -----------------------------
    # Standardize (fit on train only) + float32 for torch
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_list = [scaler.transform(X_t).astype(np.float32) for X_t in X_test_raw_list]

    # -----------------------------
    # Train DeepSurv (CoxPH)
    # -----------------------------
    model = fit_deepsurv_coxph(
        X_train=X_train,
        y_train=y_train,
        lr=lr,
        num_nodes=num_nodes,
        batch_norm=batch_norm,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
    )

    # -----------------------------
    # Evaluation: C-index (lifelines)
    # -----------------------------
    risk_train = model.predict(X_train)
    c_index_train = cindex_lifelines_from_risk(y_train, risk_train)

    if verbose:
        print(f"C-index train: {c_index_train:.4f}")

    c_index_tests = []
    ibs_tests = []

    for i, (X_test, y_test) in enumerate(zip(X_test_list, y_test_list), start=1):
        durations_test, events_test = y_test

        # ---- C-index (lifelines) ----
        risk_test = model.predict(X_test)
        c_index = cindex_lifelines_from_risk(y_test, risk_test)
        c_index_tests.append(c_index)

        # ---- IBS (pycox EvalSurv) ----
        # surv is a pd.DataFrame with index = times, columns = individuals
        surv = model.predict_surv_df(X_test)

        # Time grid strictly inside the predicted index range and test duration range
        t_min = float(np.min(durations_test))
        t_max = float(np.max(durations_test))

        # Guard against degenerate grids
        if t_max <= t_min:
            ibs = np.nan
        else:
            time_grid = np.linspace(t_min, t_max, ibs_grid_size)

            # Ensure grid within survival function index to avoid EvalSurv interpolation issues
            surv_index_min = float(surv.index.min())
            surv_index_max = float(surv.index.max())
            time_grid = time_grid[(time_grid >= surv_index_min) & (time_grid <= surv_index_max)]

            if time_grid.size < 2:
                ibs = np.nan
            else:
                ev = EvalSurv(surv, durations_test, events_test, censor_surv="km")
                ibs = float(ev.integrated_brier_score(time_grid))

        ibs_tests.append(ibs)

        if verbose:
            print(f"Test set {i}: C-index = {c_index:.4f} | IBS = {ibs:.4f}")

    mean_c_index_test = float(np.mean(c_index_tests)) if len(c_index_tests) > 0 else np.nan
    std_c_index_test = float(np.std(c_index_tests)) if len(c_index_tests) > 0 else np.nan

    mean_ibs_test = float(np.nanmean(ibs_tests)) if len(ibs_tests) > 0 else np.nan
    std_ibs_test = float(np.nanstd(ibs_tests)) if len(ibs_tests) > 0 else np.nan

    if verbose:
        print("\nSummary over independent test sets")
        print(f"Mean C-index test: {mean_c_index_test:.4f}")
        print(f"SD C-index test: {std_c_index_test:.4f}")
        print(f"Mean IBS test: {mean_ibs_test:.4f}")
        print(f"SD IBS test: {std_ibs_test:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "x_cols": x_cols,
        "c_index_train": float(c_index_train),
        "c_index_tests": c_index_tests,
        "mean_c_index_test": mean_c_index_test,
        "std_c_index_test": std_c_index_test,
        "ibs_tests": ibs_tests,
        "mean_ibs_test": mean_ibs_test,
        "std_ibs_test": std_ibs_test,
    }


#######################################################
#                                                     #
#                       CoxTime                       #
#                                                     #
#######################################################


class SimpleDataFrameMapper(BaseEstimator, TransformerMixin):
    """
    Minimal re-implementation of sklearn_pandas.DataFrameMapper.

    Parameters
    ----------
    features : list of tuples
        Each tuple must be of the form:
            (columns, transformer)

        - columns: str or list of str
        - transformer: sklearn-compatible transformer or None (passthrough)

    return_df : bool, default=False
        If True, returns a pandas DataFrame.
        Otherwise returns a numpy array.

    Notes
    -----
    - Transformers are cloned at fit time.
    - Output columns are concatenated in the order provided.
    """

    def __init__(self, features, return_df=False):
        self.features = features
        self.return_df = return_df

    def fit(self, X, y=None):
        self.transformers_ = []
        self.output_columns_ = []

        for cols, transformer in self.features:
            cols = [cols] if isinstance(cols, str) else list(cols)

            if transformer is None:
                self.transformers_.append((cols, None))
                self.output_columns_.extend(cols)
                continue

            tr = clone(transformer)
            tr.fit(X[cols], y)
            self.transformers_.append((cols, tr))

            # Infer output dimension
            if hasattr(tr, "get_feature_names_out"):
                names = tr.get_feature_names_out(cols)
                self.output_columns_.extend(names)
            else:
                # fallback: assume same number of columns
                n_out = tr.transform(X[cols].iloc[:1]).shape[1]
                if n_out == len(cols):
                    self.output_columns_.extend(cols)
                else:
                    self.output_columns_.extend(
                        [f"{cols[0]}_{i}" for i in range(n_out)]
                    )

        return self

    def transform(self, X):
        outputs = []
        for cols, transformer in self.transformers_:
            if transformer is None:
                out = X[cols].values
            else:
                out = transformer.transform(X[cols])
            if out.ndim == 1:
                out = out.reshape(-1, 1)
            outputs.append(out)
        X_out = np.hstack(outputs)
        if self.return_df:
            return pd.DataFrame(
                X_out,
                columns=self.output_columns_,
                index=X.index
            )
        return X_out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)



def prepare_coxtime_data(
    df_train,
    df_test,
    time_covariates_name,
    duration_col,
    event_col
):
    """
    Standardize covariates and prepare CoxTime targets.
    """

    cols_standardize = time_covariates_name
    standardize = [([col], StandardScaler()) for col in cols_standardize]

    x_mapper = SimpleDataFrameMapper(standardize)

    x_train = x_mapper.fit_transform(df_train).astype("float32")
    x_test = x_mapper.transform(df_test).astype("float32")

    labtrans = CoxTime.label_transform()

    get_target = lambda df: (df[duration_col].values, df[event_col].values)

    y_train = labtrans.fit_transform(*get_target(df_train))
    durations_test, events_test = get_target(df_test)

    return x_train, x_test, y_train, durations_test, events_test, labtrans


def train_single_coxtime(
    x_train,
    y_train,
    labtrans,
    num_nodes=[32, 16],
    batch_norm=True,
    dropout=0.2,
    lr=0.01,
    batch_size=256
):
    """
    Train a single CoxTime neural network.
    """

    in_features = x_train.shape[1]

    net = MLPVanillaCoxTime(
        in_features,
        num_nodes,
        batch_norm,
        dropout
    )

    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

    model.optimizer.set_lr(lr)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=100,
        verbose=False
    )

    model.compute_baseline_hazards()

    return model


def evaluate_single_coxtime(
    model,
    x_test,
    durations_test,
    events_test
):
    """
    Compute C-index (time-dependent) and IBS.
    """

    surv = model.predict_surv_df(x_test)

    ev = EvalSurv(
        surv,
        durations_test,
        events_test,
        censor_surv="km"
    )

    c_index = ev.concordance_td()

    time_grid = np.linspace(
        durations_test.min(),
        durations_test.max(),
        100
    )

    ibs = ev.integrated_brier_score(time_grid)

    return c_index, ibs


def apply_coxtime(
    df_train,
    test_groups,
    time_covariates_name,
    reports_only=False,
    duration_col="R",
    event_col="DEATH_L",
    verbose=True
):
    """
    Train CoxTime once on df_train and evaluate on multiple independent test groups.

    Returns
    -------
    c_index_train : float
    c_index_test_list : list
    ibs_list : list
    """

    if reports_only:
        time_covariates_name = [c for c in df_train.columns if c.startswith("emb_")]
        
    # ------------------------------------------------------------------
    # Prepare train data (only once)
    # ------------------------------------------------------------------
    x_train, _, y_train, _, _, labtrans = prepare_coxtime_data(
        df_train,
        df_train,  # dummy second argument (not used here)
        time_covariates_name,
        duration_col,
        event_col
    )

    durations_train = df_train[duration_col].values
    events_train = df_train[event_col].values

    # ------------------------------------------------------------------
    # Train model once
    # ------------------------------------------------------------------
    model = train_single_coxtime(
        x_train,
        y_train,
        labtrans
    )

    # ------------------------------------------------------------------
    # Evaluate on TRAIN (once)
    # ------------------------------------------------------------------
    surv_train = model.predict_surv_df(x_train)

    ev_train = EvalSurv(
        surv_train,
        durations_train,
        events_train,
        censor_surv="km"
    )

    c_index_train = ev_train.concordance_td()

    if verbose:
        print(f"C-index Train: {c_index_train:.4f}\n")

    # ------------------------------------------------------------------
    # Evaluate on TEST groups
    # ------------------------------------------------------------------
    c_index_test_list = []
    ibs_list = []

    iterator = tqdm(test_groups, desc="CoxTime evaluation", total=len(test_groups)) \
        if verbose else test_groups

    for i, df_test in enumerate(iterator):

        # Prepare test data
        _, x_test, _, durations_test, events_test, _ = prepare_coxtime_data(
            df_train,
            df_test,
            time_covariates_name,
            duration_col,
            event_col
        )

        # Evaluate
        c_index_test, ibs = evaluate_single_coxtime(
            model,
            x_test,
            durations_test,
            events_test
        )

        c_index_test_list.append(c_index_test)
        ibs_list.append(ibs)

        if verbose:
            print(f"Test group {i+1}")
            print(f"C-index Test: {c_index_test:.4f}")
            print(f"IBS: {ibs:.4f}\n")

    return c_index_train, c_index_test_list, ibs_list


#######################################################
#                                                     #
#                   Cox Time Varying                  #
#                                                     #
#######################################################




# ---------------------------------------------------------
# 1) Utilities
# ---------------------------------------------------------
def _select_covariate_columns(df, seq_struc_col=None, emb_prefix="emb_"):
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    if seq_struc_col:
        missing = [c for c in seq_struc_col if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")
        cov_cols = emb_cols + list(seq_struc_col)
    else:
        cov_cols = emb_cols

    if len(cov_cols) == 0:
        raise ValueError(f"No covariate columns found with prefix '{emb_prefix}'.")
    return cov_cols


def build_start_stop_dataframe(
    df_long,
    id_col="ID",
    obs_time_col="days_since_start",
    duration_col="R",
    event_col="DEATH_L",
    covariate_cols=None,
):
    if covariate_cols is None:
        raise ValueError("covariate_cols must be provided.")

    needed = [id_col, obs_time_col, duration_col, event_col] + list(covariate_cols)
    missing = [c for c in needed if c not in df_long.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_long = df_long[needed].copy().sort_values([id_col, obs_time_col])

    rows = []
    for pid, g in df_long.groupby(id_col, sort=False):
        g = g.dropna(subset=[obs_time_col, duration_col, event_col]).copy()
        if g.empty:
            continue

        R = float(g[duration_col].iloc[0])
        E = int(g[event_col].iloc[0])

        # One row per unique obs time, keep last if duplicates
        g = g.groupby(obs_time_col, as_index=False).tail(1)
        g = g.sort_values(obs_time_col).reset_index(drop=True)

        times = np.clip(g[obs_time_col].astype(float).values, 0.0, R)
        if times.size == 0:
            continue

        for k in range(len(times)):
            start = 0.0 if k == 0 else float(times[k])
            stop = float(min(times[k + 1], R)) if k < len(times) - 1 else float(R)
            if stop <= start:
                continue

            cov_vals = g.loc[k, covariate_cols].to_dict()
            row = {
                id_col: pid,
                "start": start,
                "stop": stop,
                event_col: 1 if (E == 1 and k == len(times) - 1) else 0,
            }
            row.update(cov_vals)
            rows.append(row)

    df_tv = pd.DataFrame(rows)
    if df_tv.empty:
        raise ValueError("Start/stop dataframe is empty. Check your input data and columns.")
    return df_tv


def build_patient_level_last_obs(
    df_long,
    id_col="ID",
    obs_time_col="days_since_start",
    duration_col="R",
    event_col="DEATH_L",
    covariate_cols=None,
):
    if covariate_cols is None:
        raise ValueError("covariate_cols must be provided.")

    needed = [id_col, obs_time_col, duration_col, event_col] + list(covariate_cols)
    missing = [c for c in needed if c not in df_long.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    g = (
        df_long[needed]
        .sort_values([id_col, obs_time_col])
        .groupby(id_col, as_index=False)
        .tail(1)
        .copy()
    )
    return g[[id_col, duration_col, event_col] + list(covariate_cols)].reset_index(drop=True)


def cindex_from_ctv_on_last_obs(ctv, df_last, duration_col="R", event_col="DEATH_L", covariate_cols=None):
    if covariate_cols is None:
        raise ValueError("covariate_cols must be provided.")
    risk = ctv.predict_partial_hazard(df_last[covariate_cols]).values.reshape(-1)
    T = df_last[duration_col].values
    E = df_last[event_col].values
    return concordance_index(T, -risk, E)


# -----------------------------
# IPCW Brier score + IBS
# -----------------------------
def _km_censoring_survival(durations, events):
    """
    Estimate censoring survival G(t) using Kaplan-Meier on censoring indicator:
    censor_event = 1 if censored (i.e., event==0), else 0
    """
    km = KaplanMeierFitter()
    censor_event = 1 - events
    km.fit(durations, event_observed=censor_event)
    return km


def _G_at(km, t, eps=1e-12):
    """
    Return G(t) with safeguards. `lifelines` returns a Series for predict.
    """
    g = float(km.predict(t))
    return max(g, eps)


def brier_score_ipcw_from_surv(
    surv_probs,        # shape (n,), S_i(t)
    durations,         # shape (n,)
    events,            # shape (n,)
    km_cens,           # fitted KM censoring survival
    t,
    eps=1e-12,
):
    """
    IPCW Brier score at time t.

    BS(t) = 1/n * sum_i [ I(T_i <= t, E_i=1)*(0 - S_i(t))^2 / G(T_i)
                        + I(T_i > t)*(1 - S_i(t))^2 / G(t) ]
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    S = np.asarray(surv_probs, dtype=float)

    # Masks
    died_by_t = (durations <= t) & (events == 1)
    alive_past_t = durations > t

    # Weights
    # G(T_i) for those who died by t
    w1 = np.zeros_like(durations, dtype=float)
    if np.any(died_by_t):
        # vectorized via list-comprehension to avoid lifelines overhead in loops for each sample
        g_T = np.array([_G_at(km_cens, Ti, eps=eps) for Ti in durations[died_by_t]], dtype=float)
        w1[died_by_t] = 1.0 / g_T

    # G(t) for those alive past t
    Gt = _G_at(km_cens, t, eps=eps)
    w2 = np.zeros_like(durations, dtype=float)
    w2[alive_past_t] = 1.0 / Gt

    # Terms
    term1 = died_by_t.astype(float) * w1 * (S ** 2)                 # (0 - S)^2 = S^2
    term2 = alive_past_t.astype(float) * w2 * ((1.0 - S) ** 2)

    return float(np.mean(term1 + term2))


def integrated_brier_score_ipcw(
    S_mat,             # shape (n, m) survival probabilities at grid times
    durations,
    events,
    time_grid,         # shape (m,)
    km_cens,
):
    """
    Integrated Brier score over time_grid using trapezoidal integration,
    normalized by (t_max - t_min), matching common IBS conventions.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    if time_grid.size < 2:
        return np.nan

    bs = np.zeros_like(time_grid, dtype=float)
    for j, t in enumerate(time_grid):
        bs[j] = brier_score_ipcw_from_surv(S_mat[:, j], durations, events, km_cens, t)

    integral = np.trapz(bs, time_grid)
    denom = (time_grid[-1] - time_grid[0])
    return float(integral / denom) if denom > 0 else np.nan


# ---------------------------------------------------------
# 2) Main routine: CoxTimeVaryingFitter + C-index + IBS
# ---------------------------------------------------------

def global_coxtimevaryingfitter(
    df_train,
    test_groups,
    id_col="ID",
    obs_time_col="days_since_start",
    time_col="R",
    event_col="DEATH_L",
    seq_struc_col=None,
    emb_prefix="emb_",
    penalizer=1e-4,
    l1_ratio=0.5,
    ibs_grid_size=100,
    verbose=True,
):
    """
    Train CoxTimeVaryingFitter on train and evaluate:
    - C-index (last observation) on train + each test group
    - IBS (IPCW Brier) on each test group using a *CoxPH-style* survival approximation
      from the last-observation risk score (static landmarking).

    Important:
    lifelines.CoxTimeVaryingFitter does not provide predict_survival_function().
    For IBS, we use:
        S_i(t) = exp( - H0(t) * exp(eta_i) )
    where eta_i is the (log) partial hazard from last-observation covariates and H0(t)
    is the baseline cumulative hazard estimated by the fitted TV-Cox model.
    """
    covariate_cols = _select_covariate_columns(df_train, seq_struc_col=seq_struc_col, emb_prefix=emb_prefix)

    # ---------
    # Scaling
    # ---------
    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_train_scaled[covariate_cols] = scaler.fit_transform(df_train_scaled[covariate_cols]).astype(np.float32)

    test_groups_scaled = []
    for df_test in test_groups:
        df_t = df_test.copy()
        df_t[covariate_cols] = scaler.transform(df_t[covariate_cols]).astype(np.float32)
        test_groups_scaled.append(df_t)

    # ---------
    # Start/stop + fit
    # ---------
    df_train_tv = build_start_stop_dataframe(
        df_long=df_train_scaled,
        id_col=id_col,
        obs_time_col=obs_time_col,
        duration_col=time_col,
        event_col=event_col,
        covariate_cols=covariate_cols,
    )

    ctv = CoxTimeVaryingFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    ctv.fit(
        df_train_tv,
        id_col=id_col,
        start_col="start",
        stop_col="stop",
        event_col=event_col,
        show_progress=False,
    )

    # Baseline cumulative hazard (Series indexed by time)
    # If it's a DataFrame with one column, squeeze to Series.
    base_ch = ctv.baseline_cumulative_hazard_
    if hasattr(base_ch, "shape") and len(base_ch.shape) == 2 and base_ch.shape[1] == 1:
        base_ch = base_ch.iloc[:, 0]
    # Ensure increasing index
    base_ch = base_ch.sort_index()

    # ---------
    # Train C-index (last obs)
    # ---------
    df_train_last = build_patient_level_last_obs(
        df_long=df_train_scaled,
        id_col=id_col,
        obs_time_col=obs_time_col,
        duration_col=time_col,
        event_col=event_col,
        covariate_cols=covariate_cols,
    )
    c_index_train = cindex_from_ctv_on_last_obs(
        ctv, df_train_last, duration_col=time_col, event_col=event_col, covariate_cols=covariate_cols
    )
    if verbose:
        print(f"C-index train: {c_index_train:.4f}")

    # ---------
    # Test metrics: C-index + IBS
    # ---------
    c_index_tests = []
    ibs_tests = []

    for i, df_test_scaled in enumerate(test_groups_scaled, start=1):
        df_test_last = build_patient_level_last_obs(
            df_long=df_test_scaled,
            id_col=id_col,
            obs_time_col=obs_time_col,
            duration_col=time_col,
            event_col=event_col,
            covariate_cols=covariate_cols,
        )

        durations_test = df_test_last[time_col].values.astype(float)
        events_test = df_test_last[event_col].values.astype(int)

        # ---- C-index ----
        cidx = cindex_from_ctv_on_last_obs(
            ctv, df_test_last, duration_col=time_col, event_col=event_col, covariate_cols=covariate_cols
        )
        c_index_tests.append(cidx)

        # ---- IBS (IPCW Brier) ----
        km_cens = _km_censoring_survival(durations_test, events_test)

        t_min = float(np.min(durations_test))
        t_max = float(np.max(durations_test))

        if t_max <= t_min:
            ibs = np.nan
        else:
            time_grid = np.linspace(t_min, t_max, ibs_grid_size)

            # Compute baseline cumulative hazard on the grid by interpolation
            # (baseline_cumulative_hazard_ is defined on event times)
            H0 = np.interp(time_grid, base_ch.index.values.astype(float), base_ch.values.astype(float))
            H0 = np.maximum(H0, 0.0)  # numeric safety

            # log-risk (eta) from last obs covariates
            eta = np.log(ctv.predict_partial_hazard(df_test_last[covariate_cols]).values.reshape(-1))
            # exp(eta) = partial hazard
            ph = np.exp(eta).reshape(-1, 1)  # (n,1)

            # Survival matrix S_mat (n, m): S_i(t) = exp(-H0(t) * exp(eta_i))
            S_mat = np.exp(-ph * H0.reshape(1, -1))

            ibs = integrated_brier_score_ipcw(
                S_mat=S_mat,
                durations=durations_test,
                events=events_test,
                time_grid=time_grid,
                km_cens=km_cens,
            )

        ibs_tests.append(ibs)

        if verbose:
            print(f"Test set {i}: C-index = {cidx:.4f} | IBS = {ibs:.4f}")

    mean_c = float(np.mean(c_index_tests)) if c_index_tests else np.nan
    std_c = float(np.std(c_index_tests)) if c_index_tests else np.nan

    mean_ibs = float(np.nanmean(ibs_tests)) if ibs_tests else np.nan
    std_ibs = float(np.nanstd(ibs_tests)) if ibs_tests else np.nan

    if verbose:
        print("\nSummary over independent test sets")
        print(f"Mean C-index test: {mean_c:.4f}")
        print(f"SD C-index test: {std_c:.4f}")
        print(f"Mean IBS test: {mean_ibs:.4f}")
        print(f"SD IBS test: {std_ibs:.4f}")

    return {
        "model": ctv,
        "scaler": scaler,
        "covariate_cols": covariate_cols,
        "c_index_train": float(c_index_train),
        "c_index_tests": c_index_tests,
        "mean_c_index_test": mean_c,
        "std_c_index_test": std_c,
        "ibs_tests": ibs_tests,
        "mean_ibs_test": mean_ibs,
        "std_ibs_test": std_ibs,
    }