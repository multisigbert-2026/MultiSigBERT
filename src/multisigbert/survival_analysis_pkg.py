########################################################################################
#                                                                                      #
#                                                                                      #
#                                        COX MODEL                                     #
#                                                                                      #
#                                                                                      #
#       https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html        #
#                                                                                      #
#            https://scikit-survival.readthedocs.io/en/stable/install.html             #
#                                                                                      #
#                             pip install scikit-survival                              #
#                                                                                      #
#                                                                                      #
#                                      Cox - skglm                                     #
#                                                                                      #
#   contrib.scikit-learn.org/skglm/stable/auto_examples/plot_survival_analysis.html    #
#                                                                                      #
########################################################################################

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from tqdm import tqdm

import time

from sklearn.model_selection import KFold
from sklearn.linear_model import HuberRegressor


from scipy.stats import ttest_ind

################### START       ---     SKGLM ###################
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index, k_fold_cross_validation
from skglm.datafits import Cox
from skglm.penalties import L1
from skglm.solvers import ProxNewton
from skglm.utils.jit_compilation import compiled_clone
###################  END        ---     SKGLM ###################


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, _check_feature_names_in

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom one-hot encoder inspired by scikit-survival's OneHotEncoder.
    Encodes categorical columns into 0/1 indicator columns (drop-first is not applied).

    Parameters
    ----------
    allow_drop : bool, default=True
        Whether to allow dropping columns with a single unique category.

    Attributes
    ----------
    feature_names_ : list
        List of categorical column names that were encoded.
    categories_ : dict
        Mapping of column name to original category values (order preserved).
    encoded_columns_ : list
        Final list of columns after encoding.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : array-like
        Names of the input features seen during fit.
    """

    def __init__(self, *, allow_drop=True):
        self.allow_drop = allow_drop

    def fit(self, X, y=None):
        """Fit encoder to categorical structure of X."""
        self.fit_transform(X)
        return self

    def _encode(self, X, columns_to_encode):
        # Optionally drop columns with a single category
        if self.allow_drop:
            columns_to_encode = [col for col in columns_to_encode if X[col].nunique() > 1]
        return pd.get_dummies(X, columns=columns_to_encode, drop_first=False)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to X and return encoded DataFrame."""
        #_check_n_features(self, X, reset=True)
        self.feature_names_in_ = X.columns
        self.n_features_in_ = X.shape[1]

        columns_to_encode = X.select_dtypes(include=["object", "category"]).columns
        Xt = self._encode(X, columns_to_encode)

        # Store metadata
        self.feature_names_ = list(columns_to_encode)
        self.categories_ = {
            col: X[col].astype("category").cat.categories
            for col in columns_to_encode
        }
        self.encoded_columns_ = Xt.columns

        return Xt

    def transform(self, X):
        """Transform new data using the fitted encoder."""
        check_is_fitted(self, "encoded_columns_")
        # _check_n_features(self, X, reset=True)  # REMOVE THIS

        Xt = X.copy()
        for col, cats in self.categories_.items():
            Xt[col] = Xt[col].astype("category").cat.set_categories(cats)

        Xt_encoded = self._encode(Xt, self.feature_names_)
        return Xt_encoded.loc[:, self.encoded_columns_]

    def get_feature_names_out(self, input_features=None):
        """Return names of the output features after encoding."""
        check_is_fitted(self, "encoded_columns_")
        input_features = _check_feature_names_in(self, input_features)
        return self.encoded_columns_.values.copy()



def preprocess_cox(
    df,
    var_DEATH='DEATH',
    date_death="date_death",
    debut_etude="date_start",
    fin_etude="date_end",
    var_id="ID",
    return_id=False,
    var_known=None,
    retire_duration_known=False
):
    """
    Prepares the input DataFrame for Cox model training by computing event durations,
    filtering invalid entries, and extracting signature-based features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing patient-level and time-series information.
    var_DEATH : str, default='DEATH'
        Name of the binary event column (1 = event occurred, 0 = censored).
    date_death : str, default='date_death'
        Name of the column containing the event (death) date.
    debut_etude : str, default='date_start'
        Column indicating the start date of follow-up for each patient.
    fin_etude : str, default='date_end'
        Column indicating the end date of follow-up (for censored patients).
    var_id : str, default='ID'
        Column identifying patients.
    return_id : bool, default=False
        If True, also returns the list of patient IDs retained after filtering.
    var_known : str or None, optional
        Name of the column indicating the known observation window (used to truncate duration).
    retire_duration_known : bool, default=False
        If True, subtracts 'duration_known' from 'duree' to focus on prediction beyond last known data.

    Returns
    -------
    df_filtered : pd.DataFrame
        Filtered DataFrame with columns ['duree', var_DEATH, *signature_features].
    features : list of str
        Names of signature features used as input to the Cox model.
    id_list : list of str, optional
        List of patient IDs retained (only if return_id is True).
    """

    # Ensure datetime columns are parsed correctly
    for col in [debut_etude, date_death, fin_etude]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"Converting '{col}' to datetime format...")
            df[col] = pd.to_datetime(df[col])

    # Compute duration (duree): from start to death if available, else to end of follow-up
    df['duree'] = np.where(
        df[date_death].notna(),
        (df[date_death] - df[debut_etude]).dt.days,
        (df[fin_etude] - df[debut_etude]).dt.days
    )

    # Optionally subtract the known observation window from duration
    if retire_duration_known and var_known is not None and var_known in df.columns:
        print("\n--- Duration Truncation Based on Known Observation Window ---")
        n_patients_before = df[var_id].nunique()
        mean_known_duration = df.groupby(var_id)['duree'].max().mean()
        print(f"Number of patients before duration cut: {n_patients_before}")
        print(f"Mean duration in study per patient: {mean_known_duration:.2f} days")

        if df.columns.duplicated().sum() > 0:
            duplicated_cols = df.columns[df.columns.duplicated(keep='first')]
            if var_known in duplicated_cols.values:
                print(f"Warning: duplicated column '{var_known}' detected — removing duplicates.")
                df = df.loc[:, ~df.columns.duplicated()]

        df['duree'] = df['duree'] - df[var_known].values
        df = df[df['duree'] >= 0]

        n_patients_after = df[var_id].nunique()
        mean_pred_duration = df.groupby(var_id)['duree'].max().mean()
        print(f"Number of patients after duration cut: {n_patients_after}")
        print(f"Mean predicted duration per patient: {mean_pred_duration:.2f} days")

    # Identify signature features
    features = [col for col in df.columns if col.startswith('sig_')]

    # Drop rows with missing values in key columns
    df_clean = df.dropna(subset=['duree', var_DEATH] + features)

    # Filter out negative durations
    df_clean = df_clean[df_clean['duree'] >= 0]

    # Build final filtered DataFrame
    cols = [var_id, 'duree', var_DEATH] + features
    if var_known is not None and var_known in df.columns:
        cols.insert(3, var_known)
    df_filtered = df_clean[cols]

    # Return outputs
    id_list = df_clean[var_id].unique()
    if return_id:
        return df_filtered, features, id_list
    else:
        return df_filtered, features




def feat_event_extract(
    df_OG,
    features,
    var_id="ID",
    var_DEATH="DEATH",
    var_duree="duree"
):
    """
    Extracts and preprocesses features and event information for Cox model training.

    Parameters
    ----------
    df_OG : pd.DataFrame
        DataFrame containing raw patient data.
    features : list
        List of feature column names to be used as covariates.
    var_id : str, default='ID'
        Column name identifying patients.
    var_DEATH : str, default='DEATH'
        Column name indicating event occurrence (1 = event, 0 = censored). Will be converted to boolean.
    var_duree : str, default='duree'
        Column name containing durations. Negative durations are filtered out.

    Returns
    -------
    Xt : scipy.sparse.csr_matrix
        Sparse matrix containing encoded covariates using OneHotEncoder.
    y : np.ndarray
        Structured array of shape (n_samples,) with fields 'event' (bool) and 'time' (float).
    id_list : np.ndarray
        Array of patient identifiers retained in the final dataset.

    Notes
    -----
    - Filters out rows with negative durations.
    - Converts the event column to boolean type.
    - Sorts the dataset by ascending duration.
    - Applies one-hot encoding to the specified features.
    """

    df = df_OG.copy()

    # Convert the DEATH column to boolean (True for event, False for censoring)
    df[var_DEATH] = df[var_DEATH].astype(bool)

    # Check for negative durations
    negative_durations = df[df[var_duree] < 0]
    if len(negative_durations) > 0:
        print(f"Number of negative durations: {len(negative_durations)}")
        print("Warning: negative duration values detected!\n", negative_durations[[var_DEATH, var_duree]])

    # Remove rows with negative durations
    df = df[df[var_duree] >= 0]

    # Print summary statistics on the event distribution
    n_deaths = df[var_DEATH].sum()
    total = len(df)
    print(f"Number of events (deaths): {n_deaths} out of {total} ({100 * n_deaths / total:.2f}%)")

    # Sort by increasing duration
    df = df.sort_values(by=var_duree).reset_index(drop=True)

    # Create the structured array y with 'event' and 'time'
    y = np.array(
        [(event, duration) for event, duration in zip(df[var_DEATH], df[var_duree])],
        dtype=[('event', '?'), ('time', '<f8')]
    )

    # Encode features using OneHotEncoder
    # Xt = OneHotEncoder().fit_transform(df[features])
    Xt = CustomOneHotEncoder().fit_transform(df[features])

    return Xt, y, df[var_id].unique()



def surv_curves(
    Xt: np.ndarray,
    y: pd.DataFrame,
    cox_model,
    var_event: str = 'event',
    view_all: bool = True,
    slice_choice: slice = slice(100, 110)
):
    """
    Plot individual survival curves predicted by a fitted Cox model.

    Parameters
    ----------
    Xt : np.ndarray
        Feature matrix used for survival prediction.
    y : pd.DataFrame
        DataFrame containing survival information, including time and event indicators.
    cox_model : object
        Trained Cox model supporting `predict_survival_function`.
    var_event : str, default='event'
        Column name in `y` indicating the event occurrence (1 if event occurred, 0 otherwise).
    view_all : bool, default=True
        If True, plot survival curves for all patients; otherwise, use `slice_choice`.
    slice_choice : slice, default=slice(100, 110)
        Slice object to select a subset of patients for plotting if `view_all` is False.

    Returns
    -------
    list
        A list of survival functions (one per patient) predicted by the Cox model.
    """
    surv_fns = cox_model.predict_survival_function(Xt)

    percent_dead = 100 * y[var_event].sum() / len(y)
    print(f"Percentage of deceased patients: {percent_dead:.3f}%")

    # Define time grid for plotting
    time_points = np.quantile(y["time"], np.linspace(0, 0.3, 120))
    time_points = np.insert(time_points, 0, 0.0)

    legend_handles = []
    legend_labels = []

    patient_slice = slice(0, len(surv_fns)) if view_all else slice_choice

    _, ax = plt.subplots(figsize=(9, 6))
    for fn, label in zip(surv_fns[patient_slice], y["event"][patient_slice].astype(int)):
        (line,) = ax.step(time_points, fn(time_points), where="post", color=f"C{label}", alpha=0.5)
        name = "DEATH" if label == 1 else "SURVIVAL"
        legend_labels.append(name)
        legend_handles.append(line)

    # Remove duplicate labels
    unique_labels, unique_handles = [], []
    for lbl, hdl in zip(legend_labels, legend_handles):
        if lbl not in unique_labels:
            unique_labels.append(lbl)
            unique_handles.append(hdl)

    ax.legend(unique_handles, unique_labels)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(f"Survival probability for patients {patient_slice.start} to {patient_slice.stop - 1}")
    ax.grid(True)
    plt.show()

    return surv_fns


########################################################################################
#                                                                                      #
#                                                                                      #
#                                      Cox - skglm                                     #
#                                                                                      #
#   contrib.scikit-learn.org/skglm/stable/auto_examples/plot_survival_analysis.html    #
#                                                                                      #
#                                                                                      #
########################################################################################



def preprocess_skglm(Xt, y, print_infos=False):
    """
    Prepare design matrix and target array for training a Cox model using skglm.

    Parameters
    ----------
    Xt : pandas.DataFrame or scipy.sparse matrix or np.ndarray
        Feature matrix.
    y : structured np.ndarray
        Array with fields ('event', '?') and ('time', '<f8') representing survival targets.
    print_infos : bool, default=False
        Whether to print shape and dtype information for debugging.

    Returns
    -------
    X : np.ndarray
        Feature matrix in contiguous NumPy format.
    y : np.ndarray
        Target array of shape (n_samples, 2), where columns are (event, time).
    
    Raises
    ------
    TypeError
        If Xt cannot be converted to a NumPy array.
    """
    
    # Convert Xt to NumPy array safely
    if hasattr(Xt, "to_numpy"):
        X = Xt.to_numpy()
    elif hasattr(Xt, "toarray"):  # e.g., scipy sparse matrix
        X = Xt.toarray()
    elif isinstance(Xt, np.ndarray):
        X = Xt
    else:
        raise TypeError("Xt must be a pandas DataFrame, a scipy sparse matrix, or a NumPy array.")

    # Convert y structured array to (event, time) 2D float array
    event = y['event'].astype(float)
    time = y['time']
    y = np.column_stack((event, time))

    # Ensure contiguous memory layout
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)

    if print_infos:
        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        print(f"y shape: {y.shape}, dtype: {y.dtype}")

    return X, y



def skglm_risk(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    print_nonnuls: bool = True
) -> np.ndarray:
    """
    Compute risk coefficients from a Cox model using skglm with L1 regularization.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target array with two columns: 'event' (1 if event occurred, 0 otherwise) and 'time' (duration).
    alpha : float
        Regularization strength for the L1 penalty.
    print_nonnuls : bool, default=True
        If True, print the number of nonzero coefficients in the fitted model.

    Returns
    -------
    np.ndarray
        Risk coefficient vector from the fitted Cox model. Note: returned with a minus sign.
    """
    datafit = compiled_clone(Cox())
    penalty = compiled_clone(L1(alpha))
    datafit.initialize(X, y)
    solver = ProxNewton(fit_intercept=False, max_iter=50)
    w_sk = solver.solve(X, y, datafit, penalty)[0]

    if print_nonnuls:
        print(f"Number of nonzero coefficients in solution: {(w_sk != 0).sum()} out of {len(w_sk)}.")

    return -w_sk  # Note: minus sign is required for correct risk scoring


def sk_cox(
    Xt: pd.DataFrame,
    y: pd.DataFrame,
    infos_preprocess: bool = False,
    var_time: str = 'time',
    var_event: str = 'event',
    alpha: float = 1e-2,
    id_list: Optional[list] = None
):
    """
    Fit a Cox proportional hazards model using risk scores estimated by skglm.

    This function performs preprocessing, computes L1-regularized risk scores with skglm,
    fits a Cox model using lifelines, and optionally attaches patient IDs.

    Parameters
    ----------
    Xt : pd.DataFrame
        Covariate matrix (input features).
    y : pd.DataFrame
        Target DataFrame containing event and time columns.
    infos_preprocess : bool, default=False
        If True, print information about the preprocessing step.
    var_time : str, default='time'
        Column name representing time-to-event.
    var_event : str, default='event'
        Column name representing event indicator (1 if event occurred, 0 otherwise).
    alpha : float, default=1e-2
        Regularization strength for L1 penalty in skglm.
    id_list : list or None, default=None
        Optional list of patient IDs to add to the survival DataFrame.

    Returns
    -------
    tuple
        - Fitted lifelines CoxPHFitter model.
        - DataFrame with columns ['event', 'time', 'risk_score'] (and 'ID' if `id_list` is provided).
        - Estimated coefficient vector from skglm.
        - Processed covariates (X).
        - Processed targets (y).
    """
    # Preprocess data for skglm format
    X, y = preprocess_skglm(Xt, y, print_infos=infos_preprocess)

    print(f" --------> X.shape: {X.shape}\n")
    print(f" --------> y.shape: {y.shape}\n")
    
    # Compute sparse coefficients with skglm
    w_sk = skglm_risk(X, y, alpha)

    print(f"w_sk.shape: {w_sk.shape}")

    # Compute linear risk scores
    risk_scores = np.dot(X, w_sk)

    # Build survival DataFrame
    df_survival = pd.DataFrame({
        'ID':id_list,
        'event': y[:, 0],
        'time': y[:, 1],
        'risk_score': risk_scores
    })

    # Optionally attach patient IDs
    if id_list is not None and len(id_list) > 0:
        if len(df_survival) == len(id_list):
            df_survival['ID'] = id_list
        else:
            raise ValueError("Size mismatch: 'df_survival' and 'id_list' must have the same length.")
    
    # print(f"""
    # df_survival.shape: {df_survival.shape},
    # event.shape: {df_survival['event'].shape}, 
    # time.shape: {df_survival['time'].shape},
    # risk_score.shape: {df_survival['risk_score'].shape}\n
    # """)

    print("---")
    print("NaNs in risk_score:", df_survival['risk_score'].isna().sum())
    print("Infs in risk_score:", np.isinf(df_survival['risk_score']).sum())

    # Remove rows with NaN or inf in risk_score
    df_survival = df_survival[np.isfinite(df_survival['risk_score'])].copy()

    # Fit Cox model on risk scores
    cph = CoxPHFitter()
    cph_fitted = cph.fit(df_survival, duration_col=var_time, event_col=var_event, formula="risk_score")

    print(f"C-index on training data: {c_index_skglm(df_survival):.3f}")

    return cph_fitted, df_survival, w_sk, X, y



def sk_cox_CV(
    Xt: pd.DataFrame,
    y: pd.DataFrame,
    k_fold: int = 5,
    infos_preprocess: bool = False,
    var_time: str = 'time',
    var_event: str = 'event',
    alpha: float = 1e-2,
    id_list: Optional[list] = None
):
    """
    Fit a Cox proportional hazards model with skglm-estimated risk scores and perform cross-validation.

    This function performs preprocessing, computes L1-regularized coefficients,
    generates risk scores, fits a CoxPH model using lifelines, and evaluates the model
    via k-fold cross-validation.

    Parameters
    ----------
    Xt : pd.DataFrame
        Covariate matrix (input features).
    y : pd.DataFrame
        Target DataFrame containing time and event columns.
    k_fold : int, default=5
        Number of folds to use in cross-validation.
    infos_preprocess : bool, default=False
        If True, print preprocessing details.
    var_time : str, default='time'
        Column name representing time-to-event.
    var_event : str, default='event'
        Column name representing event indicator (1 if event occurred, 0 otherwise).
    alpha : float, default=1e-2
        Regularization strength for L1 penalty in skglm.
    id_list : list or None, default=None
        Optional list of patient IDs to attach to the risk DataFrame.

    Returns
    -------
    tuple
        - Fitted lifelines CoxPHFitter model.
        - Risk DataFrame with columns ['event', 'time', 'risk_score'] (and 'ID' if provided).
        - Estimated coefficient vector from skglm.
        - Cross-validation scores (list of C-index values).
        - Processed covariates (X).
        - Processed targets (y).
    """
    # Preprocess input for skglm
    X, y = preprocess_skglm(Xt, y, print_infos=infos_preprocess)

    # Fit sparse model and compute risk scores
    w_sk = skglm_risk(X, y, alpha)
    risk_scores = np.dot(X, w_sk)

    # Build risk dataframe
    df_survival = pd.DataFrame({
        'event': y[:, 0],
        'time': y[:, 1],
        'risk_score': risk_scores
    })

    # Optionally attach patient IDs
    if id_list is not None and len(id_list) > 0:
        if len(df_survival) == len(id_list):
            df_survival['ID'] = id_list
        else:
            raise ValueError("Size mismatch: 'df_survival' and 'id_list' must have the same length.")

    print("---")
    print("NaNs in risk_score:", df_survival['risk_score'].isna().sum())
    print("Infs in risk_score:", np.isinf(df_survival['risk_score']).sum())

    # Remove rows with NaN or inf in risk_score
    df_survival = df_survival[np.isfinite(df_survival['risk_score'])].copy()


    # Initialize and cross-validate CoxPH model
    cph = CoxPHFitter()
    scores = k_fold_cross_validation(
        cph,
        df_survival,
        duration_col=var_time,
        event_col=var_event,
        k=k_fold,
        scoring_method="concordance_index"
    )

    # Fit final model on full data
    cph_fitted = cph.fit(df_survival, duration_col=var_time, event_col=var_event, formula="risk_score")

    print(f"C-index on training data: {c_index_skglm(df_survival):.3f}")

    # Sort by survival time
    df_survival = df_survival.sort_values(by='time', ascending=True).reset_index(drop=True)

    return cph_fitted, df_survival, w_sk, scores, X, y



def sk_cox_cvlambda(
    Xt: pd.DataFrame,
    y: pd.DataFrame,
    infos_preprocess: bool = False,
    var_time: str = 'time',
    var_event: str = 'event',
    lambda_l1_range: list = [1e-3, 1e-2, 1e-1, 1],
    n_folds: int = 5,
    id_list: Optional[list] = None
):
    """
    Fit a CoxPH model with skglm-based risk scores and perform cross-validation over L1 penalties.

    This function performs cross-validation to select the best L1 regularization strength
    (lambda) based on C-index, then fits the final model using the best lambda.

    Parameters
    ----------
    Xt : pd.DataFrame
        Covariate matrix (input features).
    y : pd.DataFrame
        Target DataFrame containing time and event columns.
    infos_preprocess : bool, default=False
        If True, print preprocessing details.
    var_time : str, default='time'
        Column name representing time-to-event.
    var_event : str, default='event'
        Column name representing event indicator (1 if event occurred, 0 otherwise).
    lambda_l1_range : list, default=[1e-3, 1e-2, 1e-1, 1]
        List of L1 regularization values to test.
    n_folds : int, default=5
        Number of folds for cross-validation.
    id_list : list or None, default=None
        Optional list of patient IDs to attach to the risk DataFrame.

    Returns
    -------
    tuple
        - cph_fitted: Fitted CoxPHFitter model.
        - df_survival: DataFrame with event, time, risk_score (and ID if provided).
        - best_w_sk: Estimated coefficients using the best lambda.
        - cindex_scores_output: List of C-index scores for the best lambda across folds (plays the 'score' role).
        - X: Covariate matrix used for final fit.
        - y: Target array used for final fit.
        - best_lambda: Best lambda value selected.
    """
    X, y = preprocess_skglm(Xt, y, print_infos=infos_preprocess)

    best_lambda = None
    best_cindex = 0.0
    best_w_sk = None
    cindex_scores_output = []

    kf = KFold(n_splits=n_folds)

    for lambda_l1 in tqdm(lambda_l1_range, desc="Cross-validating L1 penalties..."):
        cindex_scores = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Fit model on training split
            w_sk = skglm_risk(X_train, y_train, lambda_l1)

            # Compute risk scores on validation split
            risk_scores_val = np.dot(X_val, w_sk)
            df_val = pd.DataFrame({
                'event': y_val[:, 0],
                'time': y_val[:, 1],
                'risk_score': risk_scores_val
            })

            # Evaluate with C-index
            cindex_val = c_index_skglm(df_val)
            cindex_scores.append(cindex_val)

        mean_cindex = np.mean(cindex_scores)

        if mean_cindex > best_cindex:
            best_cindex = mean_cindex
            best_lambda = lambda_l1
            best_w_sk = w_sk
            cindex_scores_output = cindex_scores

    # Fit final model on full data with best lambda
    risk_scores = np.dot(X, best_w_sk)
    df_survival = pd.DataFrame({
        'event': y[:, 0],
        'time': y[:, 1],
        'risk_score': risk_scores
    })

    # Optionally attach patient IDs
    if id_list is not None and len(id_list) > 0:
        if len(df_survival) == len(id_list):
            df_survival['ID'] = id_list
        else:
            raise ValueError("Size mismatch: 'df_survival' and 'id_list' must have the same length.")


    print("---")
    print("NaNs in risk_score:", df_survival['risk_score'].isna().sum())
    print("Infs in risk_score:", np.isinf(df_survival['risk_score']).sum())

    # Remove rows with NaN or inf in risk_score
    df_survival = df_survival[np.isfinite(df_survival['risk_score'])].copy()
    

    cph = CoxPHFitter()
    cph_fitted = cph.fit(df_survival, duration_col=var_time, event_col=var_event, formula="risk_score")

    print(f"Best lambda_l1: {best_lambda}")
    print(f"C-index on training data: {c_index_skglm(df_survival):.3f}")

    return cph_fitted, df_survival, best_w_sk, cindex_scores_output, X, y, best_lambda


def c_index_skglm(
    df_survival: pd.DataFrame,
    var_event: str = 'event',
    var_time: str = 'time',
    var_risk: str = 'risk_score'
) -> float:
    """
    Compute the concordance index (C-index) for a survival dataset.

    This metric evaluates the concordance between predicted risk scores
    and observed survival outcomes.

    Parameters
    ----------
    df_survival : pd.DataFrame
        DataFrame containing time, event, and risk score columns.
    var_event : str, default='event'
        Name of the column indicating event occurrence (1 if event, 0 if censored).
    var_time : str, default='time'
        Name of the column indicating time-to-event or censoring.
    var_risk : str, default='risk_score'
        Name of the column containing predicted risk scores.

    Returns
    -------
    float
        Concordance index between predicted risk and observed time-to-event outcomes.
    """
    return concordance_index(df_survival[var_time], -df_survival[var_risk], df_survival[var_event])



def global_cox_train(
    Xt: np.ndarray,
    y: np.ndarray,
    id_list_train: list,
    learning_cox_map: str,
    lambda_l1_CV: float,
    use_sig_risk_score: bool = False,
    verbose_time: bool = True,
    verbose_scores: bool = True,
    lambda_l1_range: list = [1e-3, 1e-2, 1e-1, 1]
):
    """
    Train a Cox model using a specified method and compute evaluation metrics.

    Supports standard training, k-fold cross-validation, and cross-validation for lambda selection.
    Outputs include the fitted model, survival predictions, coefficients, and performance metrics.

    Parameters
    ----------
    Xt : np.ndarray
        Covariate matrix (n_samples x n_features).
    y : np.ndarray
        Structured array with event and time columns.
    id_list_train : list
        List of unique patient IDs matching the rows in Xt.
    learning_cox_map : str
        Training method to use. One of:
        - 'sk_cox'
        - 'sk_cox_CV'
        - 'sk_cox_cvlambda'
    lambda_l1_CV : float
        L1 regularization strength (used for 'sk_cox' and 'sk_cox_CV').
    use_sig_risk_score : bool, default=False
        If True, compute survival and death probabilities using a sigmoid on the risk score.
    verbose_time : bool, default=True
        If True, print execution time.
    verbose_scores : bool, default=True
        If True, print performance scores (C-index and log-likelihood).
    lambda_l1_range : list, default=[1e-3, 1e-2, 1e-1, 1]
        List of L1 regularization values to test if using 'sk_cox_cvlambda'.

    Returns
    -------
    tuple
        - CoxPHFitter model
        - DataFrame of survival predictions
        - Estimated coefficients
        - Cross-validation scores (or None)
        - Processed feature matrix
        - Processed survival targets
        - C-index score
        - Log-likelihood score
        - Best lambda (or None)
    """
    start_sk = time.time()

    # Train using the specified method
    if learning_cox_map == 'sk_cox':
        cph, df_survival, w_sk, X, y_cox = sk_cox(
            Xt, y, infos_preprocess=False, alpha=lambda_l1_CV, id_list=id_list_train
        )
        scores, best_lambda = None, None

    elif learning_cox_map == 'sk_cox_CV':
        cph, df_survival, w_sk, scores, X, y_cox = sk_cox_CV(
            Xt, y, infos_preprocess=False, alpha=lambda_l1_CV, id_list=id_list_train
        )
        best_lambda = None

    elif learning_cox_map == 'sk_cox_cvlambda':
        cph, df_survival, best_w_sk, cindex_scores_output, X, y_cox, best_lambda = sk_cox_cvlambda(
            Xt, y, infos_preprocess=False, lambda_l1_range=lambda_l1_range, id_list=id_list_train
        )
        w_sk, scores = best_w_sk, cindex_scores_output

    else:
        raise ValueError(
            "Invalid value for 'learning_cox_map'. Must be one of: "
            "'sk_cox', 'sk_cox_CV', or 'sk_cox_cvlambda'."
        )

    end_sk = time.time()
    duration_sk = end_sk - start_sk
    if verbose_time:
        print(f"Execution time: {duration_sk:.2f} seconds ({duration_sk / 60:.2f} minutes)")

    # Evaluation metrics
    cindex_train = cph.score(df_survival, scoring_method='concordance_index')
    log_likelihood = cph.score(df_survival, scoring_method='log_likelihood')

    if verbose_scores:
        print(f"""
        Scores:
        C-index        = {cindex_train:.3f}
        Log-likelihood = {log_likelihood:.2f}
        """)

    # Optional sigmoid transformation
    if use_sig_risk_score:
        df_survival['probability_death'] = 1 / (1 + np.exp(-df_survival['risk_score']))
        df_survival['probability_survival'] = 1 - df_survival['probability_death']

    return cph, df_survival, w_sk, scores, X, y_cox, cindex_train, log_likelihood, best_lambda



def skglm_datatest(
    Xt_test: pd.DataFrame,
    y_test: pd.DataFrame,
    w_sk: np.ndarray,
    cph,
    id_list_test: list,
    plot_curves: bool = True,
    print_all: bool = False,
    indices_selected: list = list(range(20)),
    use_sig_risk_score: bool = False,
    id_list: Optional[list] = None
) -> Tuple[pd.DataFrame, float, np.ndarray, np.ndarray]:
    """
    Apply a trained Cox model to a test dataset and compute the C-index.

    Parameters
    ----------
    Xt_test : pd.DataFrame
        Covariate matrix for the test set.
    y_test : pd.DataFrame
        DataFrame containing 'event' and 'time' columns for the test set.
    w_sk : np.ndarray
        Coefficients from the trained skglm model.
    cph : lifelines.CoxPHFitter
        Trained Cox model used to generate survival curves.
    id_list_test : list
        List of patient IDs for the test dataset.
    plot_curves : bool, default=True
        If True, plot individual survival curves using the fitted model.
    print_all : bool, default=False
        If True, plot survival curves for all patients.
    indices_selected : list, default=range(20)
        Indices of patients to plot if `print_all` is False.
    use_sig_risk_score : bool, default=False
        If True, apply a sigmoid transformation to risk scores for probability estimation.
    id_list : list or None
        Deprecated. Use `id_list_test` instead.

    Returns
    -------
    tuple
        - DataFrame with survival predictions and risk scores.
        - Concordance index on the test set.
        - Processed feature matrix for test set.
        - Processed target array for test set.
    """
    # Preprocess test data
    Xtest, ytest = preprocess_skglm(Xt_test, y_test)

    event_test = y_test['event'].astype(float)
    time_test = y_test['time']
    risk_scores_test = np.dot(Xtest, w_sk)

    df_survival_test = pd.DataFrame({
        'event': event_test,
        'time': time_test,
        'risk_score': risk_scores_test
    })

    # Optionally attach patient IDs
    if id_list_test is not None and len(id_list_test) == len(df_survival_test):
        df_survival_test['ID'] = id_list_test
    elif id_list_test is not None and len(id_list_test) != len(df_survival_test):
        raise ValueError("Length mismatch: 'df_survival_test' and 'id_list_test' must match.")

    # Plot survival curves if requested
    if plot_curves:
        skglm_plt_surv_curves(cph, df_survival_test, print_all=print_all, indices_selected=indices_selected)

    print("---")
    print("NaNs in risk_score:", df_survival_test['risk_score'].isna().sum())
    print("Infs in risk_score:", np.isinf(df_survival_test['risk_score']).sum())

    # Remove rows with NaN or inf in risk_score
    df_survival_test = df_survival_test[np.isfinite(df_survival_test['risk_score'])].copy()
    
    # Compute C-index on the test set
    cindex_test = c_index_skglm(df_survival_test)
    print(f"Concordance index on test set: {cindex_test:.3f}")

    # Optional sigmoid transformation of risk score
    if use_sig_risk_score:
        df_survival_test['probability_death'] = 1 / (1 + np.exp(-df_survival_test['risk_score']))
        df_survival_test['probability_survival'] = 1 - df_survival_test['probability_death']

    return df_survival_test, cindex_test, Xtest, ytest
