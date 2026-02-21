########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                  SIGNATURES MODULES                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################

from typing import Union, Tuple, List

import numpy as np
import pandas as pd

# Signature Package 
import iisignature

from sklearn.preprocessing import StandardScaler


def preprocess_time(
    df: pd.DataFrame,
    time_var: str = 'date_creation',  # Temporal variable (e.g., report date)
    patient_id_col: str = 'ID'        # Patient identifier column
) -> pd.DataFrame:
    """
    Normalize the temporal variable to the [0, 1] interval for each patient.

    Each patient may have multiple reports, each associated with a timestamp.
    This function applies a per-patient min-max normalization to the specified
    temporal variable, mapping it to the [0, 1] range to represent relative time.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing patient report data.
    time_var : str, default='date_creation'
        Column name corresponding to the temporal variable (must be datetime).
    patient_id_col : str, default='ID'
        Column name identifying each patient.

    Returns
    -------
    pd.DataFrame
        DataFrame with two new columns:
        - 'timestamp_OG': original timestamp in Unix format.
        - 'timestamp': normalized timestamp in [0, 1] for each patient.
    """
    # Convert datetime column to Unix timestamp
    df.loc[:, 'timestamp_OG'] = df.loc[:, time_var].apply(lambda x: x.timestamp())

    # Container for transformed patient groups
    groups_transformed = []

    # Process each patient's group
    for patient_id, group in df.groupby(patient_id_col):
        group = group.sort_values(by='timestamp_OG')
        min_time = group['timestamp_OG'].min()
        max_time = group['timestamp_OG'].max()

        if min_time != max_time:
            group['timestamp'] = (group['timestamp_OG'] - min_time) / (max_time - min_time)
        else:
            group['timestamp'] = 0.0  # Assign zero if all timestamps are identical

        groups_transformed.append(group)

    # Concatenate all processed groups
    df_time_transfo = pd.concat(groups_transformed, ignore_index=True)

    print("Timestamps normalized for each patient to the [0, 1] range.")
    return df_time_transfo


def levy_matrix(
    path: np.ndarray,
    order: int = 2,
    use_order1: bool = False
) -> np.ndarray:
    """
    Compute the Lévy area matrix from a path using signature transform.

    If S2 denotes the level-2 elements of the signature (reshaped into a square matrix of dimension d),
    the Lévy area matrix L is defined by: L_ij = 0.5 * (S2_{j,i} - S2_{i,j}).

    Note
    ----
    The constant 1 at level 0 of a signature is deliberately omitted in the output of iisignature.sig().

    Parameters
    ----------
    path : np.ndarray
        A 2D array representing the input path of shape (T, d), where T is the number of time steps,
        and d is the dimensionality of the input features.
    order : int, default=2
        Order of the signature transform. Must be at least 2 to extract Lévy areas.
    use_order1 : bool, default=False
        If True, include level-1 signature terms (raw integrals of the path) in the final output vector.

    Returns
    -------
    np.ndarray
        A 1D array containing the flattened antisymmetric part of the level-2 signature matrix (Lévy areas).
        If `use_order1` is True, the level-1 signature terms are prepended to the output.
    """
    # Compute the signature of the path using iisignature
    signature = iisignature.sig(path, order)

    d = path.shape[1]  # Dimensionality of the path
    order1_signature = signature[:(d + 1)]  # First-order terms (including constant 1, which is excluded in iisignature)

    # Extract and reshape the level-2 terms into a d x d matrix
    matrix_S = signature[(d + 1):].reshape((d, d))

    # Compute the antisymmetric Lévy matrix
    L = 0.5 * (matrix_S - matrix_S.T)

    # Flatten and remove zero entries
    L_flat = L.flatten()
    L_flat = L_flat[L_flat != 0.0]

    # Optionally include level-1 terms
    final_vector = np.concatenate((order1_signature, L_flat)) if use_order1 else L_flat

    print("Lévy matrix computed from signature.")
    return final_vector



def lead_lag_transformation(path: np.ndarray) -> np.ndarray:
    """
    Apply the lead-lag transformation to a 1D path.

    This transformation doubles the temporal resolution by pairing each point
    with its previous value, capturing local dynamics.

    Parameters
    ----------
    path : np.ndarray
        A 1D array representing the input time series.

    Returns
    -------
    np.ndarray
        A 2D array of shape (2, 2 * len(path) - 1) representing the lead-lag path.
    """
    path = np.repeat(path, 2)
    return np.vstack((path[1:], path[:-1]))


def calculate_signature(
    path: np.ndarray,
    order: int = 2,
    use_Levy: bool = False,
    use_log: bool = False,
    apply_lead_lag: bool = False
) -> np.ndarray:
    """
    Compute the signature of a path with optional variants.

    Supports raw, log-signature, Lévy area, and lead-lag transformations.

    Parameters
    ----------
    path : np.ndarray
        Input path as a 2D array of shape (T, d).
    order : int, default=2
        Signature order to compute.
    use_Levy : bool, default=False
        If True, compute Lévy area features instead of full signature.
    use_log : bool, default=False
        If True, compute log-signature instead of raw signature.
    apply_lead_lag : bool, default=False
        If True, apply lead-lag transformation before computing signature.

    Returns
    -------
    np.ndarray
        Signature features extracted from the path.
    """
    if apply_lead_lag:
        path = lead_lag_transformation(path)

    if use_Levy:
        return levy_matrix(path)
    else:
        prep_log = iisignature.prepare(path.shape[1], order)
        return iisignature.logsig(path, prep_log) if use_log else iisignature.sig(path, order)





def encode_missing_paths(df: pd.DataFrame, structured_var_list: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Encode missing values in structured variables as separate indicator columns.

    For each variable in `structured_var_list`, this function creates a new binary
    column indicating the presence of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sequential data.
    structured_var_list : list
        List of structured variable names that may contain missing values.

    Returns
    -------
    tuple
        A tuple containing:
        - The updated DataFrame with additional binary columns for missing value indicators.
        - A list of names of the new missing indicator columns.
    """
    df_encoded = df.copy()

    for var in structured_var_list:
        missing_indicator = f"{var}_missing"
        # 1 if value is missing, 0 otherwise
        df_encoded[missing_indicator] = df_encoded[var].isna().astype(int)

    print("Missing value indicators added for structured variables.")
    return df_encoded, [f"{var}_missing" for var in structured_var_list]



def preprocess_sign(
    df: pd.DataFrame,
    retire_small: bool = False,
    patient_id_col: str = 'ID',
    return_id: bool = False,
    verbose: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Preprocess a DataFrame containing signature features.

    This function removes patients with missing signature values and optionally
    sets near-zero values to zero to improve numerical stability.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with signature features and patient identifiers.
    retire_small : bool, default=False
        If True, values with absolute magnitude less than 1e-15 are replaced with 0.
    patient_id_col : str, default='ID'
        Name of the column containing patient identifiers.
    return_id : bool, default=False
        If True, return the list of patient IDs after filtering.
    verbose : bool, default=True
        If True, print the percentage of rows containing missing signature values.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, np.ndarray)
        Cleaned DataFrame. If `return_id` is True, also returns the array of retained patient IDs.
    """
    # Identify patients with NaNs in any signature column
    ids_with_nan = df[df.filter(like='sig_').isna().any(axis=1)][patient_id_col].unique()

    if verbose:
        percent_with_nan = 100 * len(ids_with_nan) / len(df)
        print(f"Percentage of rows with NaNs in signature columns: {percent_with_nan:.2f}%")

    # Remove patients with NaNs in their signature features
    df = df[~df[patient_id_col].isin(ids_with_nan)]

    # Set very small values to zero if requested
    if retire_small:
        df.update(df.filter(like='sig_').applymap(lambda x: 0 if np.abs(x) < 1e-15 else x))

    id_list = df[patient_id_col].unique()

    return (df, id_list) if return_id else df



def signature_extract0(
    df_OG,
    order=3,
    use_log=False,
    var_temp='timestamp',
    var_date_creation="date_creation",
    var_embd='embeddings_reduced',
    var_patient='ID',
    var_DEATH='DEATH',
    var_date_death="date_death",
    debut_etude="date_start",
    fin_etude="date_end",
    var_known=None,
    use_mat_Levy=False,
    apply_lead_lag=False,
    var_structurees_list_OG=None,
    use_missing_encoding=False,
    verbose=True
):
    """
    Extract signature features from time series data, based on either embedding vectors,
    structured variables, or both.

    Parameters
    ----------
    df_OG : pd.DataFrame
        Input DataFrame with sequential patient data.
    order : int
        Order of the (log-)signature to compute.
    use_log : bool
        If True, compute log-signatures instead of regular signatures.
    var_temp : str
        Name of the temporal variable (normalized between 0 and 1).
    var_embd : str or None
        Column name containing the input embeddings (if any).
    var_patient : str
        Column identifying the patient.
    var_DEATH : str
        Column indicating survival status.
    var_date_death : str
        Column indicating date of death.
    debut_etude : str
        Column name for start of follow-up.
    fin_etude : str
        Column name for end of follow-up.
    var_known : str or None
        Column name for known duration (optional).
    use_mat_Levy : bool
        Whether to compute Levy areas (second-order signature terms).
    apply_lead_lag : bool
        Whether to apply the lead-lag transformation to the path.
    var_structurees_list_OG : list of str or None
        Structured variables to include in the signature path.
    use_missing_encoding : bool
        If True, binary indicators for missing structured variables are appended.
    verbose : bool
        If True, print summary information.

    Returns
    -------
    signature_df : pd.DataFrame
        DataFrame containing extracted signature features.
    nbr_sig : int
        Number of signature components.
    nbr_levy : int
        Number of Levy area components (if applicable).
    """

    df = df_OG.copy()

    var_embd_used = (var_embd is not None)
    embedding_dim = 0

    # If embeddings are used, split them into columns
    if var_embd_used:
        embedding_dim = len(df[var_embd].iloc[0])
        for i in range(embedding_dim):
            df[f'embedding_{i}'] = df[var_embd].apply(lambda x: x[i])

    n_components = 1  # time component always included

    # Structured variables
    if var_structurees_list_OG:
        var_structurees_list = var_structurees_list_OG.copy()

        scaler = StandardScaler()
        df[var_structurees_list] = scaler.fit_transform(df[var_structurees_list])
        df[var_structurees_list] = df[var_structurees_list].fillna(0)

        if use_missing_encoding:
            df, missing_indicators = encode_missing_paths(df, var_structurees_list)
            var_structurees_list += missing_indicators

        n_components += len(var_structurees_list)

    if var_embd_used:
        n_components += embedding_dim

    nbr_sig = iisignature.siglength(n_components, order)
    nbr_logsig = iisignature.logsiglength(n_components, order)
    nbr_sig_order2 = iisignature.siglength(n_components, 2)
    nbr_levy = nbr_sig_order2 - (2 * n_components)

    if verbose:
        if use_log:
            print(f"Number of log-signature components (order {order}): {nbr_logsig}")
        else:
            print(f"Number of signature components (order {order}): {nbr_sig}")

    signature_results = []

    for id, group in df.groupby(var_patient):
        path_cols = [var_temp]

        if var_embd_used:
            path_cols += [f'embedding_{i}' for i in range(embedding_dim)]

        if var_structurees_list_OG:
            path_cols += var_structurees_list

        path = group[path_cols].values.astype(np.float64)

        if path.shape[1] > 256:
            raise ValueError(f"Path dimensionality exceeds allowed limit: {path.shape[1]} > 256")

        start_point = np.zeros((1, path.shape[1]))
        path = np.vstack([start_point, path])

        signature = calculate_signature(
            path,
            order=order,
            use_Levy=use_mat_Levy,
            use_log=use_log,
            apply_lead_lag=apply_lead_lag
        )
        signature_dict = {f'sig_{i+1}': sig for i, sig in enumerate(signature)}

        signature_dict.update({
            var_patient: id,
            var_DEATH: group[var_DEATH].iloc[-1],
            debut_etude: group[debut_etude].iloc[-1],
            fin_etude: group[fin_etude].iloc[-1],
            var_date_death: group[var_date_death].iloc[-1]
        })

        if var_known:
            signature_dict['duration_known'] = group[var_known].iloc[-1]

        signature_results.append(signature_dict)

    signature_df = pd.DataFrame(signature_results)
    cols = [var_patient] + [col for col in signature_df.columns if col != var_patient]
    signature_df = signature_df[cols]

    return signature_df, nbr_sig, nbr_levy


def signature_extract(
    df_OG,
    order=3,
    use_log=False,
    var_temp='timestamp',
    var_date_creation="date_creation",
    var_embd='embeddings_reduced',
    var_patient='ID',
    var_DEATH='DEATH',
    var_date_death="date_death",
    debut_etude="date_start",
    fin_etude="date_end",
    interpolation_type = "linear",
    var_known=None,
    use_mat_Levy=False,
    apply_lead_lag=False,
    var_structurees_list_OG=None,
    use_missing_encoding=False,
    verbose=True
):
    """
    Extract signature features from time series data, based on either embedding vectors,
    structured variables, or both.

    Parameters
    ----------
    df_OG : pd.DataFrame
        Input DataFrame with sequential patient data.
    order : int
        Order of the (log-)signature to compute.
    use_log : bool
        If True, compute log-signatures instead of regular signatures.
    var_temp : str
        Name of the temporal variable (normalized between 0 and 1).
    var_embd : str or None
        Column name containing the input embeddings (if any).
    var_patient : str
        Column identifying the patient.
    var_DEATH : str
        Column indicating survival status.
    var_date_death : str
        Column indicating date of death.
    debut_etude : str
        Column name for start of follow-up.
    fin_etude : str
        Column name for end of follow-up.
    interpolation : {"linear", "zeros"}, default="linear"
        Strategy to handle missing values in structured variables:
        - "linear": perform patient-wise linear interpolation, with backward/forward
          filling as fallback.
        - "zeros": replace all missing values by 0.
    var_known : str or None
        Column name for known duration (optional).
    use_mat_Levy : bool
        Whether to compute Levy areas (second-order signature terms).
    apply_lead_lag : bool
        Whether to apply the lead-lag transformation to the path.
    var_structurees_list_OG : list of str or None
        Structured variables to include in the signature path.
    use_missing_encoding : bool
        If True, binary indicators for missing structured variables are appended.
    verbose : bool
        If True, print summary information.

    Returns
    -------
    signature_df : pd.DataFrame
        DataFrame containing extracted signature features.
    nbr_sig : int
        Number of signature components (regular or log).
    nbr_levy : int
        Number of Levy area components (if applicable).
    """

    df = df_OG.copy()

    var_embd_used = (var_embd is not None)
    embedding_dim = 0

    # If embeddings are used, split them into columns
    if var_embd_used:
        embedding_dim = len(df[var_embd].iloc[0])
        for i in range(embedding_dim):
            df[f'embedding_{i}'] = df[var_embd].apply(lambda x: x[i])

    n_components = 1  # time component always included

    # Structured variables
    if var_structurees_list_OG:
        var_structurees_list = var_structurees_list_OG.copy()

        # Check existence of all structured variables
        missing_cols = [c for c in var_structurees_list if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing structured variables in df: {missing_cols}")

        # (1) Ensure numeric (non-numeric -> NaN)
        df[var_structurees_list] = df[var_structurees_list].apply(pd.to_numeric, errors="coerce")

        # Handle missing values according to interpolation mode
        if interpolation_type == "linear":
            # (2) Sort by time *within patient* before interpolation
            sorted_idx = df.sort_values([var_patient, var_temp]).index
            block = df.loc[sorted_idx, var_structurees_list]

            # Linear interpolate internal gaps, then extend to edges (bfill/ffill)
            block = (
                block.groupby(df.loc[sorted_idx, var_patient], group_keys=False)
                     .apply(lambda g: g.interpolate(method="linear").bfill().ffill())
            )
            # Assign back and restore original row order
            df.loc[sorted_idx, var_structurees_list] = block.values

            # (3) Fallback: if a variable is entirely NaN for a patient, fill with 0
            df[var_structurees_list] = (
                df.groupby(var_patient, group_keys=False)[var_structurees_list]
                  .apply(lambda g: g.fillna(0.0))
            )

        elif interpolation_type == "zeros":
            # Replace missing values by zeros
            df[var_structurees_list] = df[var_structurees_list].fillna(0.0)
        else:
            raise ValueError(f"Unknown interpolation mode: {interpolation_type}")

        # Standardize after imputation
        scaler = StandardScaler()
        df[var_structurees_list] = scaler.fit_transform(df[var_structurees_list])

        # Optional: encode missing indicators if requested
        if use_missing_encoding:
            df, missing_indicators = encode_missing_paths(df, var_structurees_list)
            var_structurees_list += missing_indicators

        n_components += len(var_structurees_list)

    print(f"Number of patients after interpolation:{df[var_patient].nunique()}")
    
    if var_embd_used:
        n_components += embedding_dim


    if use_log:
        nbr_sig = iisignature.logsiglength(n_components, order)
    else:
        nbr_sig = iisignature.siglength(n_components, order)

    nbr_sig_order2 = iisignature.siglength(n_components, 2)
    nbr_levy = nbr_sig_order2 - (2 * n_components)

    if verbose:
        if use_log:
            print(f"Number of log-signature components (order {order}): {nbr_sig}")
        else:
            print(f"Number of signature components (order {order}): {nbr_sig}")
        if use_mat_Levy:
            print(f"Number of Levy area components: {nbr_levy}")

    signature_results = []

    for id, group in df.groupby(var_patient):
        # Ensure chronological sorting
        group = group.sort_values(var_temp)

        path_cols = [var_temp]

        if var_embd_used:
            path_cols += [f'embedding_{i}' for i in range(embedding_dim)]

        if var_structurees_list_OG:
            path_cols += var_structurees_list

        path = group[path_cols].values.astype(np.float64)

        if path.shape[1] > 256:
            raise ValueError(f"Path dimensionality exceeds allowed limit: {path.shape[1]} > 256")

        start_point = np.zeros((1, path.shape[1]))
        path = np.vstack([start_point, path])

        signature = calculate_signature(
            path,
            order=order,
            use_Levy=use_mat_Levy,
            use_log=use_log,
            apply_lead_lag=apply_lead_lag
        )
        signature_dict = {f'sig_{i+1}': sig for i, sig in enumerate(signature)}

        signature_dict.update({
            var_patient: id,
            var_DEATH: group[var_DEATH].iloc[-1],
            debut_etude: group[debut_etude].iloc[-1],
            fin_etude: group[fin_etude].iloc[-1],
            var_date_death: group[var_date_death].iloc[-1]
        })

        if var_known:
            signature_dict['duration_known'] = group[var_known].iloc[-1]

        signature_results.append(signature_dict)

    signature_df = pd.DataFrame(signature_results)
    cols = [var_patient] + [col for col in signature_df.columns if col != var_patient]
    signature_df = signature_df[cols]

    return signature_df, nbr_sig, nbr_levy
