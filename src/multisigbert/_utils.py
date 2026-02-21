########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                     PACKAGES IMPORT                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


import pandas as pd
from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from datetime import timedelta

import torch


# Import for transformers (e.g., OncoBERT)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Package SIF EMBEDDING ###################
from sif_embedding_pkg import arora_methods

# Package COMPRESSION ###################
from compression_pkg import apply_linear_projection

# Package SIGNATURES ###################
from signature_pkg import preprocess_time, signature_extract, preprocess_sign

# Package SURVIVAL ANALYSIS ###################
from survival_analysis_pkg import preprocess_cox, feat_event_extract, global_cox_train, skglm_datatest


# Package SKLEARN
import warnings
# set_config(display="text")  # displays text representation of estimators
warnings.filterwarnings("ignore", 
                        message="invalid value encountered in divide", 
                        category=RuntimeWarning)


from sklearn.model_selection import train_test_split








########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                     DATA PREPROCESS                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


def convert_date_columns(data, verbose=True, date_format="%Y-%m-%d"):
    """
    Automatically convert columns that likely contain dates into pandas datetime format.

    This function scans the column names of the DataFrame and identifies any columns
    starting with 'DATE', 'Date_', or 'date_' as potential date fields. It then attempts
    to parse their values using `pd.to_datetime` with the specified date format, coercing
    invalid entries to NaT.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame potentially containing date-like columns.
    verbose : bool, default=True
        Whether to print messages when missing values (NaT) are detected after conversion.
    date_format : str, default="%Y-%m-%d"
        Expected date format for parsing (e.g. ISO format: year-month-day).

    Returns
    -------
    pd.DataFrame
        The original DataFrame with specified columns converted to datetime format.
    """
    date_columns = [col for col in data.columns if col.startswith('DATE')
                    or col.startswith('Date_') or col.startswith('date_')]

    for col in date_columns:
        data[col] = pd.to_datetime(data[col], format=date_format, errors='coerce')
        if data[col].isna().any() and verbose:
            print(f"Warning: Column '{col}' contains NaT values after conversion with format '{date_format}'.")

    return data




def update_death_status(
    df, 
    var_id='ID', 
    var_death='DEATH', 
    var_time_death='date_death', 
    var_end_date='date_end', 
    verbose=True
):
    """
    Update the death status and end-of-study date in a patient-level DataFrame.

    For rows where the death date is not missing (NaT), this function sets the death indicator
    to 1 and updates the study end date to match the death date. This is useful to ensure 
    consistency between survival labels and timelines.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing survival status and date columns.
    var_id : str, default='ID'
        Column name for unique patient identifiers.
    var_death : str, default='DEATH'
        Binary indicator column (1 = deceased, 0 = censored).
    var_time_death : str, default='date_death'
        Column name containing death dates.
    var_end_date : str, default='date_end'
        Column name representing the end of follow-up (either censoring or death).
    verbose : bool, default=True
        If True, prints the number of newly updated death statuses.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with synchronized death status and end-of-study dates.
    """
    # Count initial number of unique deceased patients
    initial_deceased_count = df[df[var_death] == 1][var_id].nunique()
    
    # Set DEATH = 1 where a death date is known (not NaT)
    df.loc[df[var_time_death].notna(), var_death] = 1

    # Update date_end to match date_death for deceased patients
    mask = (df[var_death] == 1) & (df[var_time_death].notna())
    df.loc[mask, var_end_date] = df.loc[mask, var_time_death]

    # Count updated number of deceased patients
    final_deceased_count = df[df[var_death] == 1][var_id].nunique()

    if verbose:
        print(f"Number of new {var_id} with {var_death}=1: {final_deceased_count - initial_deceased_count}")

    return df



########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                   EMBEDDING EXTRACTION                               #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################



def compute_sentence_embeddings(df, tokenizer, model, device, method='Arora'):
    """
    Compute sentence embeddings for a DataFrame using either the Arora method or CLS token extraction.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing text data.
    - tokenizer: The tokenizer compatible with the language model.
    - model: The pre-trained language model (e.g., BERT-based).
    - device (str or torch.device): The device on which to run the model ('cpu' or 'cuda').
    - method (str): The embedding strategy, either 'Arora' or 'CLS_token'. Default is 'Arora'.

    Returns:
    - pd.DataFrame: The input DataFrame with an added column containing the sentence embeddings.
    """
    allowed_methods = ['CLS_token', 'Arora']
    if method not in allowed_methods:
        warnings.warn(f"Warning: '{method}' is not a supported embedding method. Defaulting to 'CLS_token'.")
        method = 'CLS_token'

    if method == 'Arora':
        print("\n-- Using Arora's method for sentence embedding computation.")
        df = arora_methods(df, tokenizer, model, device)
    elif method == 'CLS_token':
        print("\n-- Using CLS token as sentence embedding.")
        df = compute_embd(df, tokenizer, model, device)

    return df



def load_nlp_model(path_model):
    """
    Load a pretrained NLP model and tokenizer for binary classification, and assign it to the appropriate device.

    Parameters:
    - path_model (str): Path or identifier to the pretrained model directory or Hugging Face model name.

    Returns:
    - tokenizer: Hugging Face tokenizer associated with the model.
    - model: Pretrained transformer model loaded for sequence classification (2 labels).
    - device: torch.device object ('cuda' if available, otherwise 'cpu').
    """
    # Load tokenizer and model with binary classification head
    tokenizer = AutoTokenizer.from_pretrained(path_model)
    model = AutoModelForSequenceClassification.from_pretrained(path_model, num_labels=2)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return tokenizer, model, device


def process_and_export_embeddings(
    df,
    tokenizer,
    model,
    device,
    export_path,
    subdivision=2,
    method_embd="Arora",
    var_id="ID",
    var_embd="embeddings",
    cols_to_drop=None
):
    """
    Process a DataFrame by computing sentence embeddings in batches and exporting each batch to a CSV file.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing at least a text column and an identifier column.
    - tokenizer: Tokenizer compatible with the language model.
    - model: Pretrained language model used for embedding extraction.
    - device (str or torch.device): Computation device ('cpu' or 'cuda').
    - export_path (str): Path to the base output CSV file (one file per batch will be created).
    - subdivision (int): Number of batches to split the dataset into. Default is 2.
    - method_embd (str): Sentence embedding method to use ('Arora' or 'CLS_token'). Default is 'Arora'.
    - var_id (str): Name of the ID column used to split the data by patient. Default is 'ID'.
    - var_embd (str): Name of the column where embeddings are stored. Default is 'embeddings'.
    - cols_to_drop (list or None): List of column names to drop before export. Default drops ['text', 'word_embeddings', 'embeddings'].

    Returns:
    - pd.DataFrame: The original input DataFrame (unchanged).
    """
    if cols_to_drop is None:
        cols_to_drop = ['text', 'word_embeddings', var_embd]

    unique_ids = df[var_id].unique()
    split_size = len(unique_ids) // subdivision

    for i in range(subdivision):
        if i == subdivision - 1:
            selected_ids = unique_ids[i * split_size:]
        else:
            selected_ids = unique_ids[i * split_size:(i + 1) * split_size]

        sub_df = df[df[var_id].isin(selected_ids)].copy()

        # Compute sentence embeddings
        sub_df = compute_sentence_embeddings(sub_df, tokenizer, model, device, method=method_embd)
        print("-------- Sentence embeddings computed.")

        # Convert embeddings to string for export
        sub_df['embeddings_str'] = sub_df[var_embd].apply(lambda x: str(x.tolist()))
        print("-------- Embeddings converted to list format.")

        # Drop unused or large columns before export
        sub_df.drop(columns=[col for col in cols_to_drop if col in sub_df.columns], inplace=True)

        # Export path for current batch
        sub_export_path = export_path.replace('.csv', f'_{i + 1}.csv')
        sub_df.to_csv(sub_export_path, index=False)
        print(f'--- Batch {i + 1} exported successfully to {sub_export_path}')

    return df



########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                        DATA IMPORT                                   #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


def import_and_prepare_dataframe(
    path,
    col_to_tokenize="text",
    target_var="DEATH",
    time_var="date_creation",
    patient_id="ID",
    var_start="date_start",
    var_end="date_end",
    verbose=True
):
    """
    Load and preprocess a patient-level dataframe from CSV, including date conversion,
    target encoding, and computation of study start and end dates for each patient.

    Parameters:
    - path (str): Path to the CSV file.
    - col_to_tokenize (str): Name of the column containing the raw text to tokenize.
    - target_var (str): Name of the binary target variable to encode (e.g., 'DEATH').
    - time_var (str): Name of the timestamp column used to define study period bounds.
    - patient_id (str): Name of the patient identifier column.
    - var_start (str): Name of the column to store the start date of follow-up.
    - var_end (str): Name of the column to store the end date of follow-up.
    - verbose (bool): Whether to print the number of patients. Default is True.

    Returns:
    - pd.DataFrame: A chronologically sorted dataframe with encoded target and computed start/end dates.
    """
    # Load data
    df = pd.read_csv(path)

    # Encode binary target variable
    df[target_var] = df[target_var].map({'O': 1, 'N': 0})

    # Convert all relevant date columns to datetime format
    df = convert_date_columns(df)

    # Compute per-patient temporal boundaries
    df[var_end] = df.groupby(patient_id)[time_var].transform('max')
    df[var_start] = df.groupby(patient_id)[time_var].transform('min')

    # Display number of patients
    if verbose:
        num_patients = df[patient_id].nunique()
        print("-" * 70)
        print(f'Number of patients in the study: {num_patients}')

    return df.sort_values(by=[patient_id, time_var])



def global_data_import(
    path_import,                     # str or list of str (CSV file paths)
    var_id='ID',                     # Updated ID column name
    var_embd_str='embeddings_str',
    var_embd_out='embeddings',
    verbose=True,
    nrows=None,                      # Number of rows to read from each CSV (for testing)
    conformal_list=None              # Optional list of IDs to retain (filtering)
):
    """
    Imports, parses, and formats patient-level data from one or several CSV files. 
    Also assigns the result to the global namespace under standard names.

    Parameters
    ----------
    path_import : str or list of str
        Path(s) to CSV file(s) to import.
    var_id : str, default='ID'
        Column name for the patient identifier.
    var_embd_str : str, default='embeddings_str'
        Column name containing stringified embeddings.
    var_embd_out : str, default='embeddings'
        Name of the output column containing parsed embeddings.
    verbose : bool, default=True
        Whether to print progress and summary information.
    nrows : int or None, default=None
        Number of rows to read from each file (useful for testing with partial data).
    conformal_list : list or None, default=None
        Optional list of patient IDs to keep in the final dataset.

    Returns
    -------
    pd.DataFrame
        The combined and processed DataFrame (also assigned globally as `df_data` and `df_data_OG`).
    """
    start = time.time()

    # Normalize path_import to a list
    if isinstance(path_import, str):
        path_import = [path_import]

    # Read and concatenate all CSVs
    df = pd.concat([pd.read_csv(path, nrows=nrows) for path in path_import], ignore_index=True)

    # Optional filtering based on conformal_list
    if conformal_list is not None:
        df = df[df[var_id].isin(conformal_list)].reset_index(drop=True)

    # Parse embeddings from string to numpy array
    df[var_embd_out] = df[var_embd_str].apply(lambda x: np.fromstring(x, sep=' '))

    # Convert date columns if present
    df = convert_date_columns(df, verbose=False)

    # Update death status if necessary
    df = update_death_status(df, verbose=False)

    # Assign to global namespace
    globals()["df_data"] = df
    globals()["df_data_OG"] = df.copy()

    # Logging
    if verbose:
        duration = time.time() - start
        print(f"\n----- Total import time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"Number of unique patients: {df[var_id].nunique()}")

    # Drop the str column for embeddings
    df = df.drop(columns=[var_embd_str])

    return df




def run_global_data_filtering(
    df_OG,
    Ndays,
    retired_days_method='constant',
    percent_chosen=0.05,
    verbose=True
):
    """
    Runs the full data filtering pipeline:
    - Retires the tail of patient timelines based on the selected strategy,
    - Enforces a minimum number of days between the last report and event/censoring (if applicable),
    - Cleans invalid or short records,
    - Reports key statistics.

    Parameters
    ----------
    df_OG : pd.DataFrame
        Input DataFrame containing patient time series.
    Ndays : int
        Minimum number of days required between the last report and the event/censoring (if applicable).
    retired_days_method : str, default='constant'
        Strategy for retiring the tail of each patient's data.
        Options: 'constant', 'random', 'percent_chosen', 'percent_with_ecart'.
    percent_chosen : float, default=0.05
        Percentage of the most recent reports to remove when applicable.
    verbose : bool, default=True
        If True, prints summary information.

    Returns
    -------
    df_all : pd.DataFrame
        Filtered and cleaned DataFrame.
    nbr_id_removed : int
        Number of patients removed during cleaning.
    min_days_difference : int
        Minimal time difference in days between reports after filtering.
    """

    if retired_days_method == 'constant':
        print(f"Method selected: {retired_days_method} with a fixed cutoff of {Ndays} days.")
        df_all = filter_minimum_days_gap(df_OG, min_days_gap=Ndays, verbose=verbose)

    elif retired_days_method == 'random':
        print(f"Method selected: {retired_days_method} — randomly removing days from the end of each patient trajectory.")
        df_all = random_retirement_cutoff(df_OG, verbose=verbose)

    elif retired_days_method == 'percent_chosen':
        print(f"Method selected: {retired_days_method} — removing the last {percent_chosen:.0%} of observations per patient.")
        df_all = percent_retirement(df_OG, percent_chosen=percent_chosen, verbose=verbose)

    elif retired_days_method == 'percent_with_ecart':
        print(f"Method selected: {retired_days_method} — applying percent retirement ({percent_chosen:.0%}) + minimal gap of {Ndays} days.")
        df_all = percent_retirement_with_ecart(df_OG, percent_chosen=percent_chosen, min_days_gap=Ndays, verbose=verbose)

    else:
        raise ValueError(
            f"Unknown retired_days_method: '{retired_days_method}'.\n"
            " --> Expected 'constant', 'random', 'percent_chosen', or 'percent_with_ecart'."
        )

    # Post-filtering steps
    min_days_difference = calculate_min_date_difference(df_all)
    df_all, nbr_id_removed, min_days_difference = cleaning_data(df_all)
    nbr_deces, proportion_deces = count_deceased_patients(df_all)

    mean_reports_per_id = int(df_all.groupby("ID").size().mean())
    if verbose:
        print(f"Average number of reports per patient: {mean_reports_per_id}")
        print(f"Total number of valid patients: {df_all['ID'].nunique()}")
        print(f"Total number of deceased patients: {nbr_deces} ({proportion_deces:.1%})")
        print(f"Minimum time difference between reports after filtering: {min_days_difference} days")
        print(f"Number of patients removed during cleaning: {nbr_id_removed}")

    return df_all, nbr_id_removed, min_days_difference




########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                               NLP MODEL AND EMBEDDINGS                               #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################

def get_embeddings(text, tokenizer, model, device):
    """
    Generate a sentence embedding from a pre-trained transformer model.

    This function tokenizes the input text, runs it through the given model,
    and computes the mean pooling of the last hidden state to obtain a fixed-size embedding.

    Parameters
    ----------
    text : str
        Input text to encode.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associated with the pre-trained model.
    model : transformers.PreTrainedModel
        Pre-trained transformer model (e.g., BERT, CamemBERT).
    device : torch.device
        Device to run the model on (CPU or GPU).

    Returns
    -------
    np.ndarray
        1D NumPy array representing the embedding of the input text.
    """
    # Tokenize and send inputs to the correct device
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in tokens.items()}

    # Inference without gradient tracking
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the last hidden state and average across tokens
    last_hidden_state = outputs.hidden_states[-1]
    embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()

    return embedding


def compute_embd(df, tokenizer, model, device, var_target="text"):
    """
    Compute and store sentence embeddings for all rows in a DataFrame.

    Applies a transformer-based embedding extraction (`get_embeddings`) to each row of the input
    DataFrame and stores the result in a new column called 'embeddings'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a column with textual data.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associated with the pre-trained model.
    model : transformers.PreTrainedModel
        Pre-trained transformer model used to generate embeddings.
    device : torch.device
        Device to run the model on (CPU or GPU).
    var_target : str, default="text"
        Name of the column in `df` containing the raw text.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional 'embeddings' column.
    """
    embeddings = []
    for text_loc in tqdm(df[var_target], desc="Computing embeddings"):
        embedding = get_embeddings(text_loc, tokenizer, model, device)
        embeddings.append(embedding)

    df["embeddings"] = embeddings
    return df



    

########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                 GLOBAL METHOD IMPORT                                 #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


def prep_import(
    df_OG,
    order_sign=3,
    t_pred=None,
    use_log=False,
    use_mat_Levy=False,
    print_progress=False,
    var_id="ID",
    var_embd="embeddings",
    var_start="date_start",
    var_known="duration_known",
    var_death="DEATH",
    var_duration="duree",
    var_time = 'time', 
    var_event = 'event',
    interpolation_type = "linear",
    var_struct_seq_list_OG=None,
    use_missing_encoding=False,
    retire_duration_known=False,
    redefine_start=False,
    verbose=True
):
    """
    Prepares a dataset for Cox survival analysis using path signature features
    extracted from sequential time-series data such as embeddings.

    This function performs:
    - Time normalization
    - Path signature (or log-signature) extraction
    - Optional missing-value encoding for structured variables
    - Study start date adjustment (landmarking or shifting)
    - Cox-compatible dataset generation with survival outcome formatting

    Parameters
    ----------
    df_OG : pd.DataFrame
        The raw dataset with sequential data per patient.
    order_sign : int, default=3
        Order of the path signature.
    t_pred : int or float, optional
        Landmark prediction time (not used directly in this function).
    use_log : bool, default=False
        If True, compute log-signatures instead of classical ones.
    use_mat_Levy : bool, default=False
        If True, include Lévy areas (second-level iterated integrals).
    print_progress : bool, default=False
        If True, print progress messages throughout the pipeline.
    var_id : str, default="ID"
        Column name for patient identifiers.
    var_embd : str, default="embeddings"
        Column name containing the time-varying embeddings.
    var_start : str, default="date_start"
        Column name indicating the start of follow-up.
    var_known : str, optional
        Column for known observation window before prediction starts.
    var_death : str, default="DEATH"
        Column name indicating event occurrence (1 = event, 0 = censored).
    var_duration : str, default="duree"
        Name of the column containing duration values.
    var_struct_seq_list_OG : list of str, optional
        List of structured time-varying variables to include in the path.
    use_missing_encoding : bool, default=False
        If True, add binary paths to indicate missing values.
    retire_duration_known : bool, default=False
        If True, subtract known duration from total survival time.
    redefine_start : bool, default=False
        If True, shift the study start date by `duration_known`.
    verbose : bool, default=True
        If True, enables verbosity during signature extraction.

    Returns
    -------
    Xt : scipy.sparse matrix
        Matrix of signature features ready for Cox model.
    y : np.ndarray
        Structured array with dtype [('event', '?'), ('time', '<f8')].
    features_name : list of str
        Names of covariates used in the Cox model.
    nbr_sig : int
        Number of signature features.
    nbr_levy : int
        Number of Lévy features (0 if `use_mat_Levy` is False).
    id_list : list
        List of patient IDs retained in the final dataset.
    """

    start = time.time()
    df = df_OG.copy()

    # Normalize time between 0 and 1
    df_time = preprocess_time(df)
    if print_progress:
        print("*" * 30)
        print("----- Time normalization completed")

    # Signature extraction
    df_sign, nbr_sig, nbr_levy = signature_extract(
        df_time,
        order=order_sign,
        var_embd=var_embd,
        use_log=use_log,
        use_mat_Levy=use_mat_Levy,
        interpolation_type = interpolation_type,
        var_structurees_list_OG=var_struct_seq_list_OG,
        use_missing_encoding=use_missing_encoding,
        verbose=verbose
    )
    if print_progress:
        print("*" * 30)
        print("----- Signature extraction completed")

    # Optional filtering of extremely small values (currently disabled)
    df_sign = preprocess_sign(df_sign, retire_small=False, return_id=False)
    if print_progress:
        print("*" * 30)
        print("----- Signature preprocessing completed")

    # Prepare Cox-compatible data
    df_sign_filtered, features_name, id_list = preprocess_cox(
        df_sign,
        debut_etude=var_start,
        return_id=True,
        retire_duration_known=retire_duration_known
    )

    # Optionally shift start date forward if prediction begins later
    if redefine_start and var_known in df_sign_filtered.columns:
        df_sign_filtered[var_start] = df_sign_filtered[var_start] + pd.to_timedelta(df_sign_filtered[var_known], unit='D')
        print("----- Study start date shifted by 'duration_known'.")

    # Generate survival outcome and final feature matrix
    Xt, y, id_list = feat_event_extract(
        df_sign_filtered,
        features=features_name,
        var_id=var_id,
        var_DEATH=var_death,
        var_duree=var_duration
    )

    if print_progress:
        print("Survival analysis preprocessing completed.")
        print(f"Total time: {time.time() - start:.2f} seconds")
        print("-" * 70)

    # ------------------------------------------------------------------
    # Create df_study: combine IDs, features, and survival outcomes
    # ------------------------------------------------------------------
    df_study = pd.DataFrame(Xt, columns=features_name)
    df_study.insert(0, var_id, id_list)  # Add ID as first column
    df_study[var_event] = y[var_event].astype(bool)
    df_study[var_time] = y[var_time].astype(float)

    if print_progress:
        print("Survival analysis preprocessing completed.")
        print(f"Total time: {time.time() - start:.2f} seconds")
        print("-" * 70)
        print(f"df_study created with shape: {df_study.shape}")

    return Xt, y, features_name, nbr_sig, nbr_levy, id_list, df_study



def count_deceased_patients(df, var_id='ID', var_death='DEATH', verbose=True):
    """
    Count the number and percentage of deceased patients in a cohort.

    This function computes the number of unique patients with a recorded death status (i.e., DEATH=1),
    and the corresponding percentage relative to the total number of unique patients.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patient-level data, including identifiers and survival status.
    var_id : str, default='ID'
        Column name identifying each patient.
    var_death : str, default='DEATH'
        Binary column indicating whether the patient is deceased (1) or censored (0).
    verbose : bool, default=True
        If True, prints the number and proportion of deceased patients.

    Returns
    -------
    tuple
        - n_deceased : int
            Number of unique patients marked as deceased.
        - death_percentage : float
            Percentage of deceased patients among the full cohort.

    Notes
    -----
    - Death status is based on the value 1 in the `var_death` column.
    - Only unique patient IDs are considered for the count.
    """
    # Total number of unique patients
    total_patients = df[var_id].nunique()

    # Number of unique deceased patients
    n_deceased = df[df[var_death] == 1][var_id].nunique()

    # Compute percentage
    death_percentage = 100 * n_deceased / total_patients

    # Display if requested
    if verbose:
        print(f"Number of deceased patients = {n_deceased} ({death_percentage:.2f}%)")

    return n_deceased, death_percentage




def plot_start_date_distribution_by_event(
    df, 
    var_event='DEATH', 
    var_start_date='date_start', 
    deceased_color='#E5095C', 
    survival_color='#024dda', 
    bins=30
):
    """
    Plot the distribution of study start dates by survival status.

    This function creates side-by-side histograms showing the distribution of 
    study entry dates (`var_start_date`) for deceased and surviving patients.
    The survival status is determined by the binary column `var_event`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patient-level data, including event status and start dates.
    var_event : str, default='DEATH'
        Column name indicating event status (1 = deceased, 0 = censored).
    var_start_date : str, default='date_start'
        Column name representing the study entry date.
    deceased_color : str, default='#E5095C'
        Color for the deceased group histogram.
    survival_color : str, default='#024dda'
        Color for the survivor group histogram.
    bins : int, default=30
        Number of bins for the histograms.

    Returns
    -------
    None
        The function displays the plots but does not return any object.

    Notes
    -----
    - Patients are split into two groups based on `var_event`.
    - Each group is plotted using a separate histogram on a shared y-axis.
    """
    # Create subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Histogram for deceased patients
    sns.histplot(
        data=df[df[var_event] == 1],
        x=var_start_date,
        bins=bins,
        color=deceased_color,
        kde=False,
        ax=axes[0]
    )
    axes[0].set_title('Deceased Patients (event = 1)')
    axes[0].set_xlabel('Study Start Date')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Histogram for surviving patients
    sns.histplot(
        data=df[df[var_event] == 0],
        x=var_start_date,
        bins=bins,
        color=survival_color,
        kde=False,
        ax=axes[1]
    )
    axes[1].set_title('Surviving Patients (event = 0)')
    axes[1].set_xlabel('Study Start Date')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()






def filter_minimum_days_gap(
    df,
    var_id='ID',
    var_crea='date_creation',
    var_date_death='date_death',
    var_date_start='date_start',
    var_date_end='date_end',
    var_event='DEATH',
    min_days_gap=100,
    only_deceased=True,
    verbose=True
):
    """
    Filters the input DataFrame to retain, for each unique patient ID, only those observations where 
    the time gap between report creation and the patient's date of death (or end of study if censored) 
    is at least `min_days_gap`.

    If `only_deceased=True`, the filter is applied only to deceased patients while retaining all censored ones.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing patient reports.
    var_id : str
        Name of the column with unique patient IDs.
    var_crea : str
        Name of the column with report creation dates.
    var_date_death : str
        Name of the column with death dates (NaT if censored).
    var_date_start : str
        Name of the column with study start dates.
    var_date_end : str
        Name of the column with study end dates.
    var_event : str
        Name of the event column (1 if deceased, 0 if censored).
    min_days_gap : int
        Minimum number of days required between `var_crea` and `var_date_death` or `var_date_end`.
    only_deceased : bool
        If True, apply the filter only to deceased patients.
    verbose : bool
        If True, print summary statistics about the filtering.

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame, including censored patients and with an added 'duration_known' column.
    """
    initial_count = len(df)
    unique_ids_initial = df[var_id].nunique()

    # Compute reference date: death date if available, otherwise end of study
    df['date_reference'] = df.apply(
        lambda row: row[var_date_death] if pd.notna(row[var_date_death]) else row[var_date_end],
        axis=1
    )

    # Compute the number of days between creation and reference date
    df['days_difference'] = (df['date_reference'] - df[var_crea]).dt.days

    # Split the DataFrame into deceased and censored groups
    deceased = df[df[var_event] == 1]
    censored = df[df[var_event] == 0]

    # Apply filtering logic
    if only_deceased:
        deceased_filtered = deceased[deceased['days_difference'] >= min_days_gap]
        df_filtered = pd.concat([deceased_filtered, censored])
    else:
        df_filtered = df[df['days_difference'] >= min_days_gap]

    unique_ids_final = df_filtered[var_id].nunique()
    modified_ids_count = unique_ids_initial - unique_ids_final
    percentage_modified = (modified_ids_count / unique_ids_initial) * 100

    if verbose:
        print(f"Initial number of unique IDs: {unique_ids_initial}")
        print(f"Number of unique IDs after filtering: {unique_ids_final}")
        print(f"Number of modified IDs: {modified_ids_count} ({percentage_modified:.2f}%)")

    # Drop temporary columns
    df_filtered = df_filtered.drop(columns=['days_difference', 'date_reference'], errors='ignore')

    # Compute 'duration_known' based on last report creation date
    last_dates = df_filtered.groupby(var_id)[var_crea].max()
    df_filtered = df_filtered.merge(last_dates.rename('last_crea'), on=var_id)
    df_filtered['duration_known'] = (df_filtered['last_crea'] - df_filtered[var_date_start]).dt.days
    df_filtered = df_filtered[df_filtered['duration_known'] > 0]
    df_filtered = df_filtered.drop(columns=['last_crea'], errors='ignore')

    final_count = len(df_filtered)
    observations_removed = initial_count - final_count

    if verbose:
        print(f"Filtering completed with min_days_gap = {min_days_gap} days.")
        print(f"{observations_removed} observations were removed.\n")

    return df_filtered






def random_retirement_cutoff(
    df_og,
    var_id='ID',
    var_crea='date_creation',
    var_date_start='date_start',
    min_days_removed=20,
    max_days_removed=150,
    random_state=42,
    verbose=True
):
    """
    Supprime aléatoirement les dernières observations des patients en retirant
    entre min_days_removed et max_days_removed jours à partir de la dernière observation connue.

    Parameters
    ----------
    df_og : pd.DataFrame
        Le DataFrame contenant les observations séquentielles.
    var_id : str
        Le nom de la colonne identifiant les patients.
    var_crea : str
        Le nom de la colonne des dates de création des rapports.
    var_date_start : str
        Le nom de la colonne de date de début d'étude.
    min_days_removed : int
        Nombre minimal de jours à retrancher à partir de la dernière observation.
    max_days_removed : int
        Nombre maximal de jours à retrancher.
    random_state : int
        Graine pour la reproductibilité.
    verbose : bool
        Si True, affiche des informations sur la réduction.

    Returns
    -------
    df_cut : pd.DataFrame
        DataFrame réduit, avec observations tronquées aléatoirement pour chaque patient.
    """

    df=df_og.copy()

    if verbose:
        print(f"Total patients before cutoff: {df[var_id].nunique()}")
        print(f"Total observations: {len(df)}")
        
    
    np.random.seed(random_state)
    df_cut = []

    # Calculer pour chaque patient la date limite à conserver
    for id, group in df.groupby(var_id):
        group_loc = group.sort_values(by=var_crea)

        # Retirer un nombre de jours aléatoire
        days_to_remove = np.random.randint(min_days_removed, max_days_removed + 1)

        last_obs = group_loc[var_crea].max()
        cutoff_date = last_obs - pd.Timedelta(days=days_to_remove)

        # Conserver uniquement les observations avant la date de cutoff
        group_cut = group_loc[group_loc[var_crea] <= cutoff_date]

        if not group_cut.empty:
            df_cut.append(group_cut)

    df_cut = pd.concat(df_cut, ignore_index=True)

    # Calcul de la durée connue restante
    last_dates = df_cut.groupby(var_id)[var_crea].max()
    df_cut = df_cut.merge(last_dates.rename('last_crea'), on=var_id)
    df_cut['duration_known'] = (df_cut['last_crea'] - df_cut[var_date_start]).dt.days
    df_cut = df_cut[df_cut['duration_known'] > 0]
    df_cut = df_cut.drop(columns=['last_crea'], errors='ignore')

    if verbose:
        print(f"Total patients after cutoff: {df_cut[var_id].nunique()}")
        print(f"Total observations remaining: {len(df_cut)}")

    return df_cut





def percent_retirement(
    df,
    percent_chosen=0.05,
    var_id='ID',
    var_crea='date_creation',
    var_date_start='date_start',
    verbose=True
):
    """
    Removes the last X% of observations for each patient (identified by var_id),
    based on chronological ordering of reports. The percentage to remove is specified
    by `percent_chosen`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing patient time series data.
    percent_chosen : float, default=0.05
        Percentage (between 0 and 1) of the most recent reports to remove per patient.
    var_id : str, default='ID'
        Column identifying individual patients.
    var_crea : str, default='date_creation'
        Column indicating the creation date of each report.
    var_date_start : str, default='date_start'
        Column indicating the start date of the study per patient.
    verbose : bool, default=True
        If True, prints summary information.

    Returns
    -------
    df_cut : pd.DataFrame
        DataFrame with the last `percent_chosen` fraction of reports removed per patient,
        and a 'duration_known' column added.
    """

    df_cut_list = []

    for id, group in df.groupby(var_id):
        group = group.sort_values(by=var_crea)

        n_obs = len(group)
        n_to_keep = int(np.floor(n_obs * (1 - percent_chosen)))

        if n_to_keep > 0:
            df_cut_list.append(group.iloc[:n_to_keep])

    df_cut = pd.concat(df_cut_list, ignore_index=True)

    # Compute known duration after truncation
    last_dates = df_cut.groupby(var_id)[var_crea].max()
    df_cut = df_cut.merge(last_dates.rename('last_crea'), on=var_id)
    df_cut['duration_known'] = (df_cut['last_crea'] - df_cut[var_date_start]).dt.days
    df_cut = df_cut[df_cut['duration_known'] > 0]
    df_cut = df_cut.drop(columns=['last_crea'], errors='ignore')

    if verbose:
        print(f"Total patients after cutoff: {df_cut[var_id].nunique()}")
        print(f"Total observations remaining: {len(df_cut)}")

    return df_cut





def percent_retirement_with_ecart(
    df,
    percent_chosen=0.05,
    min_days_gap=100,
    var_id='ID',
    var_crea='date_creation',
    var_date_start='date_start',
    var_date_death='date_death',
    var_date_end='date_end',
    var_event='DEATH',
    verbose=True
):
    """
    Combines percent_retirement and minimum-days-gap filtering:
    - Removes the last X% of reports per patient.
    - Filters deceased patients whose last remaining report is too close to their death/censoring date.

    Parameters
    ----------
    df : pd.DataFrame
        Patient time-series data.
    percent_chosen : float, default=0.05
        Proportion of most recent observations to remove per patient.
    min_days_gap : int, default=100
        Minimum number of days required between the last observation and death/censoring.
    var_id : str, default='ID'
        Column name for patient identifier.
    var_crea : str, default='date_creation'
        Column name for report creation date.
    var_date_start : str
        Column name for study start date.
    var_date_death : str
        Column name for date of death.
    var_date_end : str
        Column name for censoring date (used when death date is missing).
    var_event : str
        Event indicator (1 = deceased, 0 = censored).
    verbose : bool, default=True
        Whether to print summary information.

    Returns
    -------
    df_filtered : pd.DataFrame
        Filtered DataFrame with reports dropped and patients with short gaps excluded.
    """

    # Step 1: Apply standard percent retirement (drop last X% of reports)
    df_cut = percent_retirement(
        df,
        percent_chosen=percent_chosen,
        var_id=var_id,
        var_crea=var_crea,
        var_date_start=var_date_start,
        verbose=verbose
    )

    # Step 2: Filter out deceased patients with insufficient gap between last report and death
    df_cut = df_cut.copy()

    # Determine the reference date for each row (death if available, otherwise censoring)
    df_cut['date_reference'] = df_cut.apply(
        lambda row: row[var_date_death] if pd.notna(row[var_date_death]) else row[var_date_end],
        axis=1
    )

    # Compute the number of days between the last observation and the reference date
    df_cut['days_difference'] = (df_cut['date_reference'] - df_cut[var_crea]).dt.days

    # Separate deceased and censored patients
    deceased = df_cut[df_cut[var_event] == 1]
    censored = df_cut[df_cut[var_event] == 0]

    # Filter out deceased patients with too short a gap
    deceased_filtered = deceased[deceased['days_difference'] >= min_days_gap]

    # Combine the filtered deceased and all censored patients
    df_filtered = pd.concat([deceased_filtered, censored], ignore_index=True)

    # Cleanup temporary columns
    df_filtered = df_filtered.drop(columns=['days_difference', 'date_reference'], errors='ignore')

    if verbose:
        print(f"Patients after percent + gap filtering: {df_filtered[var_id].nunique()}")
        print(f"Remaining observations: {len(df_filtered)}")

    return df_filtered



def make_train_test(
    df_OG, 
    var_id='ID', 
    var_date='date_creation', 
    min_date='1990-01-01', 
    n_group=10, 
    random_state=177, 
    size_test=0.5,
    verbose=False
):
    """
    Splits a DataFrame into a training set and multiple test sets without balancing for survival status.

    Parameters
    ----------
    df_OG : pd.DataFrame
        The full dataset containing patient-level data, including identifier and date columns.
    var_id : str, default='ID'
        Name of the column identifying each patient.
    var_date : str, default='date_creation'
        Name of the column containing the date of the medical report.
    min_date : str, default='1990-01-01'
        Minimum accepted date for filtering patient records.
    n_group : int, default=10
        Number of groups to split the test set into.
    random_state : int, default=177
        Random seed for reproducibility.
    size_test : float, default=0.5
        Proportion of patients to include in the test set.
    verbose : bool, default=False
        Whether to print detailed information about the split.

    Returns
    -------
    tuple
        - df_train_new : pd.DataFrame
            The unbalanced training set.
        - test_groups : list of pd.DataFrame
            List of test sets split into `n_group` subsets.
    """
    df = df_OG.copy()

    # Shuffle within each patient group
    df = df.groupby(var_id, group_keys=False).apply(
        lambda x: x.sample(frac=1, random_state=random_state)
    ).reset_index(drop=True)

    # Filter patients with all dates before min_date
    min_date = pd.to_datetime(min_date)
    df = df[df.groupby(var_id)[var_date].transform('min') > min_date]

    # Unique patient IDs
    unique_ids = df[var_id].unique()

    # Train/test split
    train_ids, test_ids = train_test_split(
        unique_ids, test_size=size_test, random_state=random_state
    )

    df_train_new = df[df[var_id].isin(train_ids)]
    df_test_combined = df[df[var_id].isin(test_ids)]

    # Split test set into n groups
    test_ids_splits = np.array_split(test_ids, n_group)
    test_groups = [
        df_test_combined[df_test_combined[var_id].isin(split)]
        for split in test_ids_splits
    ]

    if verbose:
        print(f"Number of unique patients in training set: {df_train_new[var_id].nunique()}")
        for i, group in enumerate(test_groups, start=1):
            print(f"Number of unique patients in test group {i}: {group[var_id].nunique()}")

    return df_train_new, test_groups




def make_df_conform(
    df: pd.DataFrame,
    var_id: str = 'ID',
    var_start: str = 'date_start',
    var_end: str = 'date_end',
    var_death: str = 'DEATH',
    var_death_date: str = 'date_death',
    var_T: str = 'T_days',
    var_known: Optional[str] = None,     # e.g. 'duration_known'
    var_gap: str = 'death_know_gap',
    limite_gap: Optional[int] = None,    # ← FIXED here
    verbose: bool = True
):
    """
    Conform dataset for survival analysis using patient-level T_days.
    Returns (df_filtered, df_last_obs_one_row_per_patient, id_list).
    """
    # Ensure datetime
    for col in [var_start, var_end, var_death_date]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # If T_days missing, compute once from last row per patient
    if var_T not in df.columns:
        if verbose:
            print(f"[make_df_conform] '{var_T}' not found — computing from dates.")
        df_last_obs_tmp = (
            df.sort_values(by=var_end)
              .groupby(var_id, as_index=False)
              .last()
              .loc[:, [var_id, var_death, var_start, var_end, var_death_date]]
              .assign(**{
                  var_T: lambda x: np.where(
                      x[var_death] == 1,
                      (x[var_death_date] - x[var_start]).dt.days,
                      (x[var_end] - x[var_start]).dt.days
                  )
              })
        )
        df = df.merge(df_last_obs_tmp[[var_id, var_T]], on=var_id, how='left')

    # Build one-line-per-patient view
    keep_cols = [c for c in [var_id, var_death, var_start, var_end, var_death_date, var_T, var_known] if c]
    df_last_obs = (
        df.sort_values(by=var_end)
          .groupby(var_id, as_index=False)
          .last()[keep_cols]
          .copy()
    )

    # Compute death gap only for deceased
    if var_known:
        if var_known not in df_last_obs.columns:
            raise ValueError(f"'{var_known}' not found in DataFrame.")
        df_last_obs[var_gap] = np.where(
            df_last_obs[var_death] == 1,
            (df_last_obs[var_death_date] - (df_last_obs[var_start] + pd.to_timedelta(df_last_obs[var_known], unit='d'))).dt.days,
            np.nan
        )

    # Filter invalid cases
    df_last_obs = (
        df_last_obs
        .dropna(subset=[var_start, var_end, var_T])
        .loc[df_last_obs[var_T] > 0]
        .reset_index(drop=True)
    )
    id_list = df_last_obs[var_id].values

    # Filter df accordingly
    initial_lines = len(df)
    initial_ids = df[var_id].nunique()
    df = df[df[var_id].isin(id_list)].copy()
    final_lines = len(df)
    final_ids = df[var_id].nunique()

    if verbose:
        print(f"Removed rows: {initial_lines - final_lines}")
        print(f"Removed patient IDs: {initial_ids - final_ids}")
        kept_prop = (final_ids / initial_ids) if initial_ids else 0.0
        print(f"Kept patient IDs: {final_ids}/{initial_ids} ({kept_prop:.1%})")

    # Optional: filter by death gap threshold
    if limite_gap is not None and var_known:
        pre_gap_n = df_last_obs.shape[0]
        df_last_obs = df_last_obs[df_last_obs[var_gap] <= limite_gap].copy()
        removed_gap = pre_gap_n - df_last_obs.shape[0]
        id_list = df_last_obs[var_id].values
        df = df[df[var_id].isin(id_list)].copy()
        if verbose:
            print(f"Removed by {var_gap} ≤ {limite_gap}: {removed_gap} patients")

    return df, df_last_obs, id_list



def make_df_conform0(
    df,
    var_id='ID',
    var_start='date_start',
    var_end='date_end',
    var_death='DEATH',
    var_death_date='date_death',
    var_time='duration',
    var_known=None,#'duration_known',
    var_gap='death_know_gap',
    limite_gap=None,
    verbose=True
):
    """
    Prepares a DataFrame for survival analysis by keeping only the last observation
    of each patient, computing survival durations, and filtering out invalid cases.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing patient-level longitudinal records.
    var_id : str, optional
        Name of the column containing patient identifiers (default: 'ID').
    var_start : str, optional
        Column with start dates of the study (default: 'date_start').
    var_end : str, optional
        Column with end dates of the study (default: 'date_end').
    var_death : str, optional
        Binary column indicating death (1) or censoring (0) (default: 'DEATH').
    var_death_date : str, optional
        Column with death dates (default: 'date_death').
    var_time : str, optional
        Name of the column to store computed durations (default: 'duration').
    var_known : str, optional
        Column with number of observed days before prediction (default: None).
    var_gap : str, optional
        Column to store the gap between last report and death date (only for deceased).
    limite_gap : int or None, optional
        If set, only patients with death_know_gap <= limite_gap are retained.
    verbose : bool, optional
        If True, prints summary of filtering steps (default: True).

    Returns
    -------
    df : pd.DataFrame
        Filtered DataFrame with only valid and conforming patients.
    df_last_obs : pd.DataFrame
        One-line-per-patient DataFrame used for survival analysis.
    id_list : np.ndarray
        List of valid patient IDs included in the final dataset.
    """

    # Ensure date columns are in datetime format
    df[var_start] = pd.to_datetime(df[var_start])
    df[var_end] = pd.to_datetime(df[var_end])
    df[var_death_date] = pd.to_datetime(df[var_death_date])

    if var_known:
        # Prepare one row per patient (last observation)
        df_last_obs = (
            df.sort_values(by=var_end)
              .groupby(df[var_id])
              .last()
              .loc[:, [var_id, var_death, var_start, var_end, var_death_date, var_known]]
              .assign(
                  **{var_time: lambda x: np.where(
                      x[var_death] == 1,
                      (x[var_death_date] - x[var_start]).dt.days,
                      (x[var_end] - x[var_start]).dt.days
                  )}
              )
              .reset_index(drop=True)
        )

        # Compute death gap only for deceased
        df_last_obs[var_gap] = np.where(
            df_last_obs[var_death] == 1,
            (df_last_obs[var_death_date] - (df_last_obs[var_start] + pd.to_timedelta(df_last_obs[var_known], unit='d'))).dt.days,
            np.nan
        )

    else:
        # Same as above without using var_known
        df_last_obs = (
            df.sort_values(by=var_end)
              .groupby(df[var_id])
              .last()
              .loc[:, [var_id, var_death, var_start, var_end, var_death_date]]
              .assign(
                  **{var_time: lambda x: np.where(
                      x[var_death] == 1,
                      (x[var_death_date] - x[var_start]).dt.days,
                      (x[var_end] - x[var_start]).dt.days
                  )}
              )
              .reset_index(drop=True)
        )

    # Filter out invalid cases
    df_last_obs = (
        df_last_obs
        .dropna(subset=[var_start, var_end])
        .loc[df_last_obs[var_time] > 0]
        .reset_index(drop=True)
    )

    # Sort by increasing duration
    df_last_obs = df_last_obs.sort_values(by=var_time, ascending=True).reset_index(drop=True)
    id_list = df_last_obs[var_id].values

    # Filter main DataFrame based on valid patient IDs
    initial_lines = len(df)
    initial_ids = df[var_id].nunique()

    df = df[df[var_id].isin(id_list)]

    final_lines = len(df)
    final_ids = df[var_id].nunique()

    if verbose:
        print(f"Number of removed rows: {initial_lines - final_lines}")
        print(f"Number of removed patient IDs: {initial_ids - final_ids}")

    # Optional: filter by death gap threshold
    if limite_gap:
        initial_count = df_last_obs.shape[0]
        df_last_obs = df_last_obs[df_last_obs[var_gap] <= limite_gap]
        removed_count = initial_count - df_last_obs.shape[0]

        if verbose:
            print(f"Number of patients removed due to limite_gap = {limite_gap}: {removed_count}")

        id_list = df_last_obs[var_id].values
        df = df[df[var_id].isin(id_list)]

    return df, df_last_obs, id_list




def calculate_min_date_difference(
    df,
    var_id='ID',
    var_date_death='date_death',
    var_dcrea='date_creation',
    verbose=True
):
    """
    Compute the minimum time gap (in days) between the last available medical report
    and the recorded date of death for each patient.

    This function filters patients with a known death date, identifies the latest 
    report (`var_dcrea`) for each of them, and computes the time difference between 
    that report and their death. The minimum of these values is returned.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patient-level data, including creation dates and death dates.
    var_id : str, default='ID'
        Column name for patient identifiers.
    var_date_death : str, default='date_death'
        Column containing the date of death.
    var_dcrea : str, default='date_creation'
        Column containing the timestamp of each medical report.
    verbose : bool, default=True
        If True, prints the minimum day difference found.

    Returns
    -------
    int
        Minimum number of days between the last available report and the date of death.

    Notes
    -----
    - Only patients with a non-missing death date are considered.
    - If no valid patients are found, returns NaN.
    """
    # Keep only patients with a known death date
    df_with_death = df[df[var_date_death].notna()]

    # Compute last report date per patient
    last_dcrea_dates = df_with_death.groupby(var_id)[var_dcrea].max()

    # Build merged DataFrame: one row per patient with last report and death date
    df_with_death = df_with_death[[var_id, var_date_death]].drop_duplicates().set_index(var_id)
    df_with_death['last_dcrea'] = last_dcrea_dates

    # Compute time gap (in days) between last report and death
    df_with_death['days_difference'] = (df_with_death[var_date_death] - df_with_death['last_dcrea']).dt.days

    # Get the minimum gap across all patients
    min_days_difference = df_with_death['days_difference'].min()

    if verbose:
        print(f"Minimum time between last report and {var_date_death}: {min_days_difference} days")

    return min_days_difference




def cleaning_data(
    df, 
    var_id='ID', 
    var_date_death='date_death', 
    var_dcrea='date_creation', 
    verbose=True
):
    """
    Clean a DataFrame by removing patients whose last report date occurs after their recorded date of death.

    This function identifies patients with negative time gaps between their last available medical report 
    and their death date. These cases are considered inconsistent and are removed from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patient-level data, including report dates and death dates.
    var_id : str, default='ID'
        Column name for patient identifiers.
    var_date_death : str, default='date_death'
        Column name representing the date of death.
    var_dcrea : str, default='date_creation'
        Column name representing the date of each medical report.
    verbose : bool, default=True
        If True, prints details about the cleaning process.

    Returns
    -------
    df_cleaned : pd.DataFrame
        Filtered DataFrame with inconsistent patients removed.
    nbr_id_removed : int
        Number of patient IDs that were removed due to inconsistencies.
    min_days_difference : int
        Minimum time gap (in days) between last report and death in the original data.

    Notes
    -----
    - Only patients with a non-missing death date are considered.
    - The time gap is computed as (date_death - last_report_date).
    - Patients with negative gaps are removed from the output DataFrame.
    """
    # Filter patients with known date of death
    df_with_death = df[df[var_date_death].notna()]

    # Compute last report date per patient
    last_dcrea_dates = df_with_death.groupby(var_id)[var_dcrea].max()

    # Build a DataFrame with death date and last report date per patient
    df_with_death = df_with_death[[var_id, var_date_death]].drop_duplicates().set_index(var_id)
    df_with_death['last_dcrea'] = last_dcrea_dates

    # Compute time gap between last report and date of death
    df_with_death['days_difference'] = (df_with_death[var_date_death] - df_with_death['last_dcrea']).dt.days

    # Identify patients with invalid (negative) gaps
    id_to_remove = df_with_death[df_with_death['days_difference'] < 0].index
    nbr_id_removed = len(id_to_remove)

    # Remove those patients from the original DataFrame
    df_cleaned = df[~df[var_id].isin(id_to_remove)]

    # Get minimum gap value for reporting
    min_days_difference = df_with_death['days_difference'].min()

    if verbose:
        if nbr_id_removed > 0:
            print(f"Minimum gap between last report and {var_date_death}: {min_days_difference} days")
            print(f"Number of patients removed: {nbr_id_removed}")
        else:
            print("No patients removed. All records are consistent.")
        print("-" * 30)

    return df_cleaned, nbr_id_removed, min_days_difference






########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                      GLOBAL PROCESS                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################



def global_sigbert_process(
    max_reports,
    df_all,
    df_train_new_OG,
    test_groups,
    R_comp,
    lambda_l1_CV=0.7,
    order_sign=2,
    use_mat_Levy=False,
    print_progress=False,
    var_id="ID",
    var_embd="embeddings",
    var_crea="date_creation",
    var_death="DEATH",
    var_duration="duration",
    learning_cox_map="sk_cox",
    interpolation_type = "linear",
    var_struct_seq_list=None,
    df_matrix_covar=None,
    use_other_covar=False,
    use_standard_scale=False,
    id_list_training=None
):
    from sklearn.preprocessing import StandardScaler

    print(f"\n### Processing for max_reports = {max_reports} ###")
    time_start = time.time()

    df_all_quartile = df_all.sort_values(by=[var_id, var_crea]).groupby(var_id).head(max_reports)
    std_reports = df_all_quartile.groupby(var_id)[var_crea].count().std()

    df_train_new, df_last_obs, id_list_train = make_df_conform(df_train_new_OG, var_T=var_duration,verbose=False)
    total_train = df_last_obs[var_id].nunique()
    print(f"\nTotal number of individuals in the train set: {total_train}")

    for i, df_test in enumerate(test_groups, start=1):
        nam_test_new = f"df_test{i}_new"
        nam_test_last_obs = f"df_test{i}_last_obs"
        nam_id_list = f"id_list_test{i}"

        for var_name in [nam_test_new, nam_test_last_obs, nam_id_list]:
            if var_name in globals():
                del globals()[var_name]

        globals()[nam_test_new], globals()[nam_test_last_obs], globals()[nam_id_list] = make_df_conform(df_test, verbose=False)

    df_test_new = pd.concat([globals()[f'df_test{i}_new'] for i in range(1, len(test_groups) + 1)], axis=0)
    df_last_obs_test_all = pd.concat([globals()[f'df_test{i}_last_obs'] for i in range(1, len(test_groups) + 1)], axis=0)
    total_test = df_last_obs_test_all[var_id].nunique()
    print(f"Total number of individuals in the validation set: {total_test}\n")

    df_everyone = pd.concat([df_train_new, df_test_new], axis=0)
    total_unique_patients = df_everyone[var_id].nunique()
    total_number_of_reports = len(df_everyone)
    mean_reports_per_patient = df_everyone.groupby(var_id).size().mean()
    total_deceased_patients = df_everyone[df_everyone[var_death] == 1][var_id].nunique()
    total_censored_patients = df_everyone[df_everyone[var_death] == 0][var_id].nunique()

    df_all_last = pd.concat([df_last_obs, df_last_obs_test_all])
    mean_study_time = np.mean(df_all_last[var_duration])
    std_study_time = np.std(df_all_last[var_duration])

    df_train = apply_linear_projection(df_train_new, R_comp, var_embd=var_embd)

    start_training = time.time()

    # Initialisation of df_study_all
    df_study_all = pd.DataFrame()
    
    Xt, y, features_name, nbr_sig, nbr_levy, id_list_train_V2, df_study_train = prep_import(
        df_train,
        t_pred=None,
        order_sign=order_sign,
        interpolation_type = interpolation_type,
        var_struct_seq_list_OG=var_struct_seq_list,
        use_mat_Levy=use_mat_Levy,
        print_progress=print_progress
    )

    # Add `df_study_train` in `df_study_all`
    df_study_all = pd.concat([df_study_all, df_study_train], ignore_index=True)
    
    if use_other_covar:
        if df_matrix_covar is not None:
            df_sig = pd.DataFrame({var_id: id_list_train_V2})
            df_Xt = pd.DataFrame(Xt)
            df_temp = pd.concat([df_sig, df_Xt], axis=1)
            df_merge = df_temp.merge(df_matrix_covar, on=var_id, how='left')
            
            id_list_training = list(df_merge[var_id])
            print(f"Number of individuals in training: {len(id_list_training)}")
            
            df_merge = df_merge.drop(columns=[var_id])
            Xt = df_merge.to_numpy()
        else:
            raise ValueError("`use_other_covar=True` but `df_matrix_covar` is None.")

    if use_standard_scale:
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)

    print(f"Signature feature computation took {time.time() - start_training:.2f}s\n")

    print(" --------------- Linear LASSO training --------------- ")
    cph, df_survival, w_sk, scores, X, y_cox, c_index_train, log_likelihood, _ = global_cox_train(
        Xt, y,
        id_list_train=id_list_training,
        learning_cox_map=learning_cox_map,
        lambda_l1_CV=lambda_l1_CV
    )
    print(" ---------------  --------------- ")

    c_index_test_list = []
    df_survival_test_list = []

    for i, df_test_new in enumerate(test_groups, start=1):
        df_test = apply_linear_projection(df_test_new, R_comp, var_embd=var_embd)
        Xt_test, y_test, _, _, _, id_list_test_i, df_study_test_i = prep_import(
            df_test,
            t_pred=None,
            order_sign=order_sign,
            interpolation_type = interpolation_type,
            var_struct_seq_list_OG=var_struct_seq_list,
            use_mat_Levy=use_mat_Levy,
            print_progress=print_progress
        )

        # Add `df_study_test_i` in `df_study_all`
        df_study_all = pd.concat([df_study_all, df_study_test_i], ignore_index=True)


        if use_other_covar:
            if df_matrix_covar is not None:
                df_sig_test = pd.DataFrame({var_id: id_list_test_i})
                df_Xt_test = pd.DataFrame(Xt_test)
                df_temp_test = pd.concat([df_sig_test, df_Xt_test], axis=1)
                df_merge_test = df_temp_test.merge(df_matrix_covar, on=var_id, how='left')
                df_merge_test = df_merge_test.drop(columns=[var_id])
                Xt_test = df_merge_test.to_numpy()
            else:
                raise ValueError("`use_other_covar=True` but `df_matrix_covar` is None.")

        if use_standard_scale:
            Xt_test = scaler.transform(Xt_test)

        print(f"\n--- Test Case {i} for max_reports = {max_reports} ---")
        df_survival_test, c_index_test, Xtest, ytest = skglm_datatest(
            Xt_test,
            y_test,
            w_sk,
            cph,
            id_list_test_i,
            plot_curves=False
        )

        c_index_test_list.append(c_index_test)
        df_survival_test_list.append(df_survival_test)
        print("--- ---")

    c_index_test_mean = np.mean(c_index_test_list)
    c_index_test_std = np.std(c_index_test_list, ddof=1)
    time_end = time.time() - time_start

    df_results = pd.DataFrame([{
        "Max Reports": max_reports,
        "Mean C-index": c_index_test_mean,
        "Std C-index": c_index_test_std,
        "Total Deceased Patients": total_deceased_patients,
        "Total Censored Patients": total_censored_patients,
        "Total Unique Patients": total_unique_patients,
        "Total Number of Reports": total_number_of_reports,
        "Mean Study Time (days)": np.round(mean_study_time, 3),
        "Std Study Time (days)": np.round(std_study_time, 3),
        "Execution Time (s)": np.round(time_end, 2),
        "Mean Reports per Patient": np.round(mean_reports_per_patient, 3),
        "Std Reports per Patient": np.round(std_reports, 3),
    }])

    print(f"Mean c-index (test): {c_index_test_mean:.4f}")
    print(f"Standard deviation of c-index (test): {c_index_test_std:.4f}")

    # Add column "ID" to df_survival using id_list_train_V2
    # if len(id_list_train_V2) == len(df_survival):
        # df_survival["ID"] = id_list_train_V2
    # else:
        # df_survival["ID"] = id_list_training

    return (
        df_results, cph, df_survival, w_sk, scores, X, y, y_cox,
        c_index_train, c_index_test_list, c_index_test_mean,
        c_index_test_std, df_survival_test_list, df_study_all
    )





def global_sigbert_structured_process(
    df_train_new_OG,
    test_groups,
    var_struct_seq_list,
    lambda_l1_CV=0.7,
    order_sign=2,
    use_mat_Levy=False,
    print_progress=False,
    var_id="ID",
    var_crea="date_creation",
    var_death="DEATH",
    var_duration="duration",
    interpolation_type = "linear",
    learning_cox_map="sk_cox"
):
    """
    Pipeline for survival prediction on structured covariates using signature features
    extracted from structured sequences, without embedding compression.

    Parameters
    ----------
    df_train_new_OG : pd.DataFrame
        Raw training data.
    test_groups : list of pd.DataFrame
        List of test datasets.
    var_struct_seq_list_OG : list of str
        List of column names to be used as sequential structured variables for signature extraction.
    lambda_l1_CV : float, optional
        Regularization strength for LASSO (default: 0.7).
    order_sign : int, optional
        Order of the path signature used to extract features (default: 2).
    use_mat_Levy : bool, optional
        Whether to use the Lévy area matrix (default: False).
    print_progress : bool, optional
        Whether to display progress information (default: False).
    var_id : str, optional
        Name of the column corresponding to patient ID (default: "ID").
    var_crea : str, optional
        Name of the column corresponding to the report creation date (default: "date_creation").
    var_death : str, optional
        Name of the column corresponding to the death event indicator (default: "DEATH").
    var_duration : str, optional
        Name of the column for survival duration (default: "duration").
    learning_cox_map : str, optional
        String key for the Cox model training function (default: "sk_cox").
    Returns
    -------
    Same outputs as global_sigbert_process, adapted to structured inputs.
    """

    time_start = time.time()

    # TRAIN set processing
    df_train_new, df_last_obs, id_list_train = make_df_conform(df_train_new_OG, verbose=False, var_id=var_id)
    total_train = df_last_obs[var_id].nunique()
    print(f"\nTotal number of individuals in the train set: {total_train}")

    # TEST set processing
    for i, df_test in enumerate(test_groups, start=1):
        nam_test_new = f"df_test{i}_new"
        nam_test_last_obs = f"df_test{i}_last_obs"
        nam_id_list = f"id_list_test{i}"

        for var_name in [nam_test_new, nam_test_last_obs, nam_id_list]:
            if var_name in globals():
                del globals()[var_name]

        globals()[nam_test_new], globals()[nam_test_last_obs], globals()[nam_id_list] = make_df_conform(df_test, 
                                                                                                        verbose=False,
                                                                                                        var_id=var_id)

    df_test_new = pd.concat([globals()[f'df_test{i}_new'] for i in range(1, len(test_groups) + 1)], axis=0)
    df_last_obs_test_all = pd.concat([globals()[f'df_test{i}_last_obs'] for i in range(1, len(test_groups) + 1)], axis=0)
    total_test = df_last_obs_test_all[var_id].nunique()
    print(f"Total number of individuals in the validation set: {total_test}\n")

    # Combine all
    df_everyone = pd.concat([df_train_new, df_test_new], axis=0)

    # Global stats
    total_unique_patients = df_everyone[var_id].nunique()
    total_number_of_reports = len(df_everyone)
    mean_reports_per_patient = df_everyone.groupby(var_id).size().mean()
    total_deceased_patients = df_everyone[df_everyone[var_death] == 1][var_id].nunique()
    total_censored_patients = df_everyone[df_everyone[var_death] == 0][var_id].nunique()

    df_all_last = pd.concat([df_last_obs, df_last_obs_test_all])
    mean_study_time = np.mean(df_all_last[var_duration])
    std_study_time = np.std(df_all_last[var_duration])

    # Signature feature extraction (TRAIN)
    start_training = time.time()
    Xt, y, features_name, nbr_sig, nbr_levy, id_list_train_V2 = prep_import(
        df_train_new,
        var_struct_seq_list_OG=var_struct_seq_list,
        order_sign=order_sign,
        use_mat_Levy=use_mat_Levy,
        print_progress=print_progress,
        var_id=var_id,
        var_embd=None,
        interpolation_type=interpolation_type
    )
    duration_training = time.time() - start_training
    print(f"Signature feature computation took {duration_training:.2f}s ({duration_training / 60:.2f}min).")

    # Cox model training
    print(" --------------- Linear LASSO training --------------- ")
    cph, df_survival, w_sk, scores, X, y_cox, c_index_train, log_likelihood, _ = global_cox_train(
        Xt, y, id_list_train_V2,
        learning_cox_map=learning_cox_map,
        lambda_l1_CV=lambda_l1_CV
    )
    print(" ---------------  --------------- ")

    # TEST evaluation
    c_index_test_list = []
    df_survival_test_list = []

    for i, df_test_new in enumerate(test_groups, start=1):
        df_test = globals()[f'df_test{i}_new']
        Xt_test, y_test, _, _, _, id_list_test_i = prep_import(
            df_test,
            var_struct_seq_list_OG=var_struct_seq_list,
            order_sign=order_sign,
            use_mat_Levy=use_mat_Levy,
            print_progress=print_progress,
            var_id=var_id,
            var_embd=None,
            interpolation_type = interpolation_type
        )

        print(f"\n--- Test Case {i} ---")
        df_survival_test, c_index_test, Xtest, ytest = skglm_datatest(
            Xt_test,
            y_test,
            w_sk,
            cph,
            id_list_test_i,
            plot_curves=False
        )

        c_index_test_list.append(c_index_test)
        df_survival_test_list.append(df_survival_test)
        print("--- ---")

    # Summary
    c_index_test_mean = np.mean(c_index_test_list)
    c_index_test_std = np.std(c_index_test_list, ddof=1)
    time_end = time.time() - time_start

    df_results = pd.DataFrame([{
        "Mean C-index": c_index_test_mean,
        "Std C-index": c_index_test_std,
        "Total Deceased Patients": total_deceased_patients,
        "Total Censored Patients": total_censored_patients,
        "Total Unique Patients": total_unique_patients,
        "Total Number of Reports": total_number_of_reports,
        "Mean Study Time (days)": np.round(mean_study_time, 3),
        "Std Study Time (days)": np.round(std_study_time, 3),
        "Execution Time (s)": np.round(time_end, 2),
        "Mean Reports per Patient": np.round(mean_reports_per_patient, 3)
    }])

    print(f"Mean c-index (test): {c_index_test_mean:.4f}")
    print(f"Standard deviation of c-index (test): {c_index_test_std:.4f}")

    return (
        df_results, cph, df_survival, w_sk, scores, X, y, y_cox,
        c_index_train, c_index_test_list, c_index_test_mean,
        c_index_test_std, df_survival_test_list
    )



def global_sigbert_structured_process_V2(
    df_train_new_OG,
    test_groups,
    var_struct_seq_list,
    R_comp=None,
    lambda_l1_CV=0.7,
    order_sign=2,
    use_mat_Levy=False,
    print_progress=False,
    var_id="ID",
    var_embd=None,
    var_crea="date_creation",
    var_death="DEATH",
    var_event='event',
    var_time='time',
    var_duration="duration",
    learning_cox_map="sk_cox",
    df_matrix_covar=None,
    use_other_covar=False,
    use_standard_scale=False,
    id_list_training=None
):
    """
    Pipeline for survival prediction on structured covariates using signature features
    extracted from sequential structured variables, with optional inclusion of additional
    non-sequential covariates.

    Parameters
    ----------
    df_train_new_OG : pd.DataFrame
        Raw training data.
    test_groups : list of pd.DataFrame
        List of test datasets.
    var_struct_seq_list : list of str
        List of column names to be used as sequential structured variables for signature extraction.
    lambda_l1_CV : float, optional
        Regularization strength for LASSO (default: 0.7).
    order_sign : int, optional
        Order of the path signature used to extract features (default: 2).
    use_mat_Levy : bool, optional
        Whether to use the Lévy area matrix in signature computation (default: False).
    print_progress : bool, optional
        Whether to print progress updates during feature extraction (default: False).
    var_id : str, optional
        Column name identifying patients (default: "ID").
    var_crea : str, optional
        Column name for report creation date (default: "date_creation").
    var_death : str, optional
        Column name for death event indicator (default: "DEATH").
    var_duration : str, optional
        Column name for survival duration (default: "duration").
    learning_cox_map : str, optional
        Backend used for Cox model training (default: "sk_cox").
    df_matrix_covar : pd.DataFrame, optional
        DataFrame of additional covariates to be concatenated to the design matrix.
        Must include `var_id` column and be aligned by patient.
    use_other_covar : bool, optional
        Whether to include `df_matrix_covar` in the final design matrix (default: False).
    use_standard_scale : bool, optional  
        Whether to standardize the final design matrix using sklearn's StandardScaler  
        (default: False).  

    Returns
    -------
    df_results : pd.DataFrame
        Summary table with C-index scores, cohort characteristics, and execution time.
    cph : CoxPHFitter
        Trained Cox proportional hazards model.
    df_survival : pd.DataFrame
        Survival prediction dataframe for the training set.
    w_sk : np.ndarray
        Model weights (risk scores).
    scores : np.ndarray
        Linear predictors from the Cox model.
    X : np.ndarray
        Design matrix used for training.
    y : np.ndarray
        Array with event indicators and durations.
    y_cox : np.ndarray
        Structured survival array (event, duration).
    c_index_train : float
        C-index on the training set.
    c_index_test_list : list of float
        List of C-index values per test group.
    c_index_test_mean : float
        Mean C-index across test groups.
    c_index_test_std : float
        Standard deviation of the test C-index.
    df_survival_test_list : list of pd.DataFrame
        Survival prediction outputs for each test group.
    """

    time_start = time.time()

    # TRAIN
    df_train_new, df_last_obs, id_list_train = make_df_conform(df_train_new_OG, verbose=False)
    total_train = df_last_obs[var_id].nunique()
    print(f"\nTotal number of individuals in the train set: {total_train}")

    # TESTS
    for i, df_test in enumerate(test_groups, start=1):
        nam_test_new = f"df_test{i}_new"
        nam_test_last_obs = f"df_test{i}_last_obs"
        nam_id_list = f"id_list_test{i}"

        for var_name in [nam_test_new, nam_test_last_obs, nam_id_list]:
            if var_name in globals():
                del globals()[var_name]

        globals()[nam_test_new], globals()[nam_test_last_obs], globals()[nam_id_list] = make_df_conform(df_test, verbose=False)

    df_test_new = pd.concat([globals()[f'df_test{i}_new'] for i in range(1, len(test_groups) + 1)], axis=0)
    df_last_obs_test_all = pd.concat([globals()[f'df_test{i}_last_obs'] for i in range(1, len(test_groups) + 1)], axis=0)
    total_test = df_last_obs_test_all[var_id].nunique()
    print(f"Total number of individuals in the validation set: {total_test}\n")

    # Global stats
    df_everyone = pd.concat([df_train_new, df_test_new], axis=0)
    total_unique_patients = df_everyone[var_id].nunique()
    total_number_of_reports = len(df_everyone)
    mean_reports_per_patient = df_everyone.groupby(var_id).size().mean()
    total_deceased_patients = df_everyone[df_everyone[var_death] == 1][var_id].nunique()
    total_censored_patients = df_everyone[df_everyone[var_death] == 0][var_id].nunique()

    df_all_last = pd.concat([df_last_obs, df_last_obs_test_all])
    mean_study_time = np.mean(df_all_last[var_duration])
    std_study_time = np.std(df_all_last[var_duration])

    if var_embd:
        df_train = apply_linear_projection(df_train_new, R_comp, var_embd=var_embd)
    else:
        df_train = df_train_new

    # Signature features
    start_training = time.time()
    Xt, y, features_name, nbr_sig, nbr_levy, id_list_train_V2 = prep_import(
        df_train,
        var_embd=var_embd,
        var_struct_seq_list_OG=var_struct_seq_list,
        order_sign=order_sign,
        use_mat_Levy=use_mat_Levy,
        print_progress=print_progress,
        interpolation_type = interpolation_type
    )
    duration_training = time.time() - start_training
    print(f"Signature feature computation took {duration_training:.2f}s ({duration_training / 60:.2f}min).")

    if use_other_covar:
        print("---- Use of other covariates ----\n")
        if df_matrix_covar is not None:
            # Merge covariate matrix using var_id
            df_sig = pd.DataFrame({var_id: id_list_train_V2})
            df_Xt = pd.DataFrame(Xt)
            df_temp = pd.concat([df_sig, df_Xt], axis=1)

            df_merge = df_temp.merge(df_matrix_covar, on=var_id, how='left')
            df_merge = df_merge.drop(columns=[var_id])
            Xt = df_merge.to_numpy()
            # print(f"Number of covariates: {Xt.shape[1]}")
        else:
            raise ValueError("`use_other_covar=True` but `df_matrix_covar` is None.")

    # Standardize the features in Xt using StandardScaler if requested
    if use_standard_scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xt = scaler.fit_transform(Xt)

    # Restrict id_list_train_V2 to only those IDs present in df_train_new_OG
    id_list_train_V2 = list(set(id_list_train_V2).intersection(df_train_new_OG[var_id].unique()))

    # Ensure Xt is a DataFrame with patient IDs as index
    if isinstance(Xt, np.ndarray):
        Xt = pd.DataFrame(Xt, index=id_list_train_V2)
    
    # Ensure y is a DataFrame, and set patient IDs as index if needed
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y, columns=[var_event, var_time])  # or adapt columns accordingly
        y.index = id_list_train_V2
    elif var_id in y.columns:
        y = y.set_index(var_id)
    
    # Intersect the indices to keep only patients present in both Xt and y
    common_ids = Xt.index.intersection(y.index)
    
    # Align Xt and y to the same patient set
    Xt = Xt.loc[common_ids]
    y = y.loc[common_ids].reset_index()  # Reset index for y if required by later code
    
    # Final check for alignment
    assert Xt.shape[0] == y.shape[0], "Mismatch between Xt and y lengths"



    # print(f"Xt shape: {Xt.shape}")
    # print(f"y shape: {y.shape}")
    # print(f"Index match: {Xt.index.equals(y.index)}")

    # Cox model
    print(" --------------- Linear LASSO training --------------- ")
    cph, df_survival, w_sk, scores, X, y_cox, c_index_train, log_likelihood, _ = global_cox_train(
        Xt, y, 
        id_list_train=id_list_training, #id_list_train=id_list_train_V2,
        learning_cox_map=learning_cox_map,
        lambda_l1_CV=lambda_l1_CV
    )
    print(" ---------------  --------------- ")

    # TESTS
    c_index_test_list = []
    df_survival_test_list = []

    for i, _ in enumerate(test_groups, start=1):
        df_test = globals()[f'df_test{i}_new']
        if var_embd:
            df_test = apply_linear_projection(df_test, R_comp, var_embd=var_embd)
                
        Xt_test, y_test, _, _, _, id_list_test_i = prep_import(
            df_test,
            var_embd=var_embd,
            var_struct_seq_list_OG=var_struct_seq_list,
            order_sign=order_sign,
            use_mat_Levy=use_mat_Levy,
            print_progress=print_progress,
            interpolation_type = interpolation_type
        )

        # Merge with additional covariates if requested
        if use_other_covar:
            print(f"---- Use of other covariates (Test Case {i}) ----\n")
            if df_matrix_covar is not None:
                # Merge covariate matrix using var_id
                df_sig_test = pd.DataFrame({var_id: id_list_test_i})
                df_Xt_test = pd.DataFrame(Xt_test)
                df_temp_test = pd.concat([df_sig_test, df_Xt_test], axis=1)

                df_merge_test = df_temp_test.merge(df_matrix_covar, on=var_id, how='left')
                df_merge_test = df_merge_test.drop(columns=[var_id])
                Xt_test = df_merge_test.to_numpy()
            else:
                raise ValueError("`use_other_covar=True` but `df_matrix_covar` is None.")

        # Standardize Xt_test if required
        if use_standard_scale:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            Xt_test = scaler.fit_transform(Xt_test)

        print(f"\n--- Test Case {i} ---")
        df_survival_test, c_index_test, Xtest, ytest = skglm_datatest(
            Xt_test,
            y_test,
            w_sk,
            cph,
            id_list_test_i,
            plot_curves=False
        )
        
        c_index_test_list.append(c_index_test)
        df_survival_test_list.append(df_survival_test)
        print("--- ---")

    # Summary
    c_index_test_mean = np.mean(c_index_test_list)
    c_index_test_std = np.std(c_index_test_list, ddof=1)
    time_end = time.time() - time_start

    df_results = pd.DataFrame([{
        "Mean C-index": c_index_test_mean,
        "Std C-index": c_index_test_std,
        "Total Deceased Patients": total_deceased_patients,
        "Total Censored Patients": total_censored_patients,
        "Total Unique Patients": total_unique_patients,
        "Total Number of Reports": total_number_of_reports,
        "Mean Study Time (days)": np.round(mean_study_time, 3),
        "Std Study Time (days)": np.round(std_study_time, 3),
        "Execution Time (s)": np.round(time_end, 2),
        "Mean Reports per Patient": np.round(mean_reports_per_patient, 3)
    }])

    print(f"Mean c-index (test): {c_index_test_mean:.4f}")
    print(f"Standard deviation of c-index (test): {c_index_test_std:.4f}")

    return (
        df_results, cph, df_survival, w_sk, scores, X, y, y_cox,
        c_index_train, c_index_test_list, c_index_test_mean,
        c_index_test_std, df_survival_test_list
    )




########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                  Structured Variables                                #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################



def load_sequence_data(
    path_seq: str,
    ippr_conforme: Iterable,
    sep: str = ";",
    id_col_original: str = "IPPR",
    id_col_final: str = "ID",
    date_col: str = "DATE_DONNEE",
    type_col: str = "TYPE_DONNEE_EVOLUTIVE",
    label_col: str = "LIBELLE",
    unit_col: str = "UNITE",
    limit_ids: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and filter sequential data from a CSV file, normalize identifiers, and build a
    unique mapping between an evolutionary type and its label/unit.

    This function generalizes a common notebook cell:
      1) Read the CSV of evolving data (semicolon-separated by default).
      2) Normalize the identifier column: 'IPPR' -> 'ID' (configurable).
      3) Filter rows to the provided list of conforming IDs.
      4) Parse the date column to datetime.
      5) Build a unique mapping between `type_col` and (label_col, unit_col).

    Parameters
    ----------
    path_seq : str
        Path to the input CSV file (e.g., './donnees-evolutives.csv').
    ippr_conforme : Iterable
        Collection of valid identifiers to keep. Can be a list, set, Series, etc.
        If it's a DataFrame/Series containing the original ID column, pass the
        extracted iterable; otherwise provide the iterable directly.
    sep : str, default=";"
        Field delimiter used in the CSV.
    id_col_original : str, default="IPPR"
        Name of the identifier column in the raw file (will be renamed).
    id_col_final : str, default="ID"
        Final standardized name for the identifier column.
    date_col : str, default="DATE_DONNEE"
        Name of the date column to parse with pandas.to_datetime.
    type_col : str, default="TYPE_DONNEE_EVOLUTIVE"
        Name of the column describing the type of evolving information.
    label_col : str, default="LIBELLE"
        Name of the label column associated with the type.
    unit_col : str, default="UNITE"
        Name of the unit column associated with the type.
    limit_ids : int or None, default=None
        If provided, only the first `limit_ids` identifiers from `ippr_conforme`
        are kept (useful for quick tests).

    Returns
    -------
    df_seq : pd.DataFrame
        Filtered DataFrame with the identifier column renamed to `id_col_final`
        and the date column parsed as datetime.
    mapping_type : pd.DataFrame
        A de-duplicated mapping indexed by `type_col` with columns
        `[label_col, unit_col]`.

    Raises
    ------
    ValueError
        If required columns are missing from the input file.

    Examples
    --------
    >>> df_seq, mapping = load_sequence_data(
    ...     path_seq="./donnees-evolutives.csv",
    ...     ippr_conforme=list_of_ids  # or a pandas Series of IDs
    ... )
    """
    # Read CSV
    df_seq = pd.read_csv(path_seq, sep=sep)

    # Validate required columns
    required_cols = {id_col_original, date_col, type_col, label_col, unit_col}
    missing = required_cols.difference(df_seq.columns)
    if missing:
        raise ValueError(f"Missing required columns in '{path_seq}': {sorted(missing)}")

    ippr_conforme = ippr_conforme[id_col_original].tolist()
    
    # Filter to conforming IDs
    # df_seq = df_seq[df_seq[id_col_original].isin(ids_set)].copy()
    df_seq = df_seq[df_seq[id_col_original].isin(ippr_conforme)].copy()
    # Parse date column
    df_seq[date_col] = pd.to_datetime(df_seq[date_col], errors="coerce")

    # Build unique mapping (TYPE_DONNEE_EVOLUTIVE -> [LIBELLE, UNITE])
    mapping_type = (
        df_seq[[type_col, label_col, unit_col]]
        .drop_duplicates()
        .set_index(type_col)
        .sort_index()
    )

    # Rename identifier column: 'IPPR' -> 'ID' (configurable)
    if id_col_original != id_col_final:
        df_seq = df_seq.rename(columns={id_col_original: id_col_final})

    return df_seq, mapping_type



def build_wide_table_and_metadata(
    df_seq: pd.DataFrame,
    mapping_type_libelle: pd.DataFrame,
    *,
    # Column names (raw)
    var_id_original: str = "ID",
    var_temp: str = "DATE_DONNEE",
    type_col: str = "TYPE_DONNEE_EVOLUTIVE",
    value_col: str = "CODE",
    label_col: str = "LIBELLE",
    unit_col: str = "UNITE",
    # Normalization
    var_id_final: str = "ID",
    # Variable taxonomy
    ordinal_codes: Sequence[str] = ("PS", "KAR"),
    # Keep list (will be constructed inside as [ID, DATE] + covariables_list)
    covariables_list: Sequence[str] = ("PO", "KAR", "PS", "IMC", "TA", "PL", "TAI"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a patient-date wide table from a long sequential dataset and augment metadata.

    This function generalizes a notebook workflow:
      1) Collect the set of unique (patient, date) combinations.
      2) Left-join the long table to ensure a full grid of (patient, date).
      3) Pivot to wide with one column per `type_col`, values from `value_col`.
      4) Convert columns to numeric whenever possible (handle comma decimals).
      5) Compute non-NaN coverage per variable and add it into `mapping_type_libelle`.
      6) Classify variables as NUMERICAL/CATEGORICAL, override given `ordinal_codes` to ORDINAL.
      7) Compute number of modalities for categorical variables (∞ for numerical).
      8) Keep only requested columns: [ID, DATE] + `covariables_list`.
      9) Return the filtered wide table and the aligned/augmented mapping.

    Parameters
    ----------
    df_seq : pd.DataFrame
        Long-form sequential data containing at least:
        [var_id_original, var_temp, type_col, value_col] and ideally [label_col, unit_col].
    mapping_type_libelle : pd.DataFrame
        Metadata table indexed by `type_col` with at least columns [label_col, unit_col].
        It will be augmented with coverage and typing info and filtered to variables
        effectively present in the wide table.
    var_id_original : str, default="IPPR"
        Name of the raw patient identifier column in `df_seq`.
    var_temp : str, default="DATE_DONNEE"
        Name of the time column.
    type_col : str, default="TYPE_DONNEE_EVOLUTIVE"
        Name of the variable/type column to pivot into wide columns.
    value_col : str, default="CODE"
        Name of the value column to populate wide cells.
    label_col : str, default="LIBELLE"
        Column in `mapping_type_libelle` describing the type's label.
    unit_col : str, default="UNITE"
        Column in `mapping_type_libelle` describing the type's unit. Presence is used
        as a proxy to decide NUMERICAL vs. CATEGORICAL (can be overridden downstream).
    var_id_final : str, default="ID"
        Final normalized patient identifier name. The wide table will expose this name.
    ordinal_codes : Sequence[str], default=("PS","KAR")
        Set of variable codes from `type_col` that should be forced to ORDINAL.
    covariables_list : Sequence[str], default=("PO","KAR","PS","IMC","TA","PL","TAI")
        List of target variable columns to retain (alongside ID and DATE).

    Returns
    -------
    df_wide_filtered : pd.DataFrame
        Wide-format table with columns [var_id_final, var_temp] + `covariables_list`
        (kept only if present in the pivoted data).
    mapping_libelle_filtered : pd.DataFrame
        `mapping_type_libelle` augmented with:
            - PERCENT_NON_NAN : float in [0,100]
            - Type_Var        : {"NUMERICAL","CATEGORICAL","ORDINAL"}
            - NBR_MOD         : number of levels for categorical, ∞ for numerical
        and filtered to the variables present in `df_wide_filtered`.

    Notes
    -----
    - Numeric conversion: any non-ID/non-date column is attempted as numeric using:
        str -> replace comma decimal -> to_numeric(errors='coerce').
      Columns that cannot be parsed remain NaN (and will count in coverage).
    - Coverage metric is computed after parsing on the pivoted wide table.
    """
    # --- 1) Unique dates per patient
    df_dates = (
        df_seq.groupby(var_id_original)[var_temp]
        .unique()
        .explode()
        .reset_index()
        .rename(columns={var_id_original: var_id_final})
    )

    # --- 2) Left join to have full (ID, DATE) grid
    df_seq_norm = df_seq.rename(columns={var_id_original: var_id_final})
    df_wide = df_dates.merge(
        df_seq_norm[[var_id_final, var_temp, type_col, value_col]],
        on=[var_id_final, var_temp],
        how="left",
    )

    # --- 3) Pivot to wide
    df_wide = df_wide.pivot_table(
        index=[var_id_final, var_temp],
        columns=type_col,
        values=value_col,
        aggfunc="first",
    ).reset_index()

    # --- 4) Numeric conversion when possible (exclude ID and DATE)
    candidate_cols = [c for c in df_wide.columns if c not in (var_id_final, var_temp)]
    # Replace comma decimals and coerce
    for col in candidate_cols:
        # Only operate on object/string-like columns; float dtypes are left as-is
        if df_wide[col].dtype == object:
            # Safe string ops (NaN preserved), then numeric coercion
            df_wide[col] = pd.to_numeric(
                df_wide[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    # Clean any literal 'nan' strings that might remain (defensive)
    df_wide.replace("nan", np.nan, inplace=True)

    # --- 5) Non-NaN coverage per variable (exclude ID and DATE)
    non_nan_ratio = df_wide.drop(columns=[var_id_final, var_temp]).notna().mean()
    percent_non_nan = (non_nan_ratio * 100.0).round(2)

    # Ensure mapping index is type codes
    if mapping_type_libelle.index.name != type_col:
        # If mapping has a column for type_col, set it as index, else assume current index is correct
        if type_col in mapping_type_libelle.columns:
            mapping_type_libelle = mapping_type_libelle.set_index(type_col)

    # Align coverage onto mapping (missing vars → 0%)
    mapping_type_libelle = mapping_type_libelle.copy()
    mapping_type_libelle["PERCENT_NON_NAN"] = mapping_type_libelle.index.map(percent_non_nan)
    mapping_type_libelle["PERCENT_NON_NAN"] = mapping_type_libelle["PERCENT_NON_NAN"].fillna(0.0)

    # --- 6) Variable typing
    # Default: NUMERICAL if unit present, else CATEGORICAL
    mapping_type_libelle["Type_Var"] = mapping_type_libelle[unit_col].apply(
        lambda u: "NUMERICAL" if pd.notna(u) else "CATEGORICAL"
    )
    # Override: specified codes as ORDINAL
    if ordinal_codes:
        mapping_type_libelle.loc[
            mapping_type_libelle.index.isin(ordinal_codes), "Type_Var"
        ] = "ORDINAL"

    # --- 7) Number of modalities (∞ for numerical)
    def _n_modalities(var_code: str) -> float:
        if var_code not in df_wide.columns:
            return 0.0
        return (
            float("inf")
            if mapping_type_libelle.loc[var_code, "Type_Var"] == "NUMERICAL"
            else float(pd.Series(df_wide[var_code]).nunique(dropna=True))
        )

    mapping_type_libelle["NBR_MOD"] = mapping_type_libelle.index.map(_n_modalities)

    # Sort by coverage (descending)
    mapping_type_libelle = mapping_type_libelle.sort_values(
        by="PERCENT_NON_NAN", ascending=False
    )

    # --- 8) Keep only requested columns: [ID, DATE] + covariables_list
    keep_columns: List[str] = [var_id_final, var_temp, *covariables_list]
    df_wide_filtered = df_wide[[c for c in keep_columns if c in df_wide.columns]].copy()

    # --- 9) Filter mapping to variables actually present in the wide table (excluding ID/DATE)
    present_vars = [c for c in df_wide_filtered.columns if c not in (var_id_final, var_temp)]
    mapping_libelle_filtered = mapping_type_libelle[
        mapping_type_libelle.index.isin(present_vars)
    ].copy()

    return df_wide_filtered, mapping_libelle_filtered


def deduplicate_by_timestamp(
    df_merge,
    covariables_list,
    var_id="ID",
    var_crea="date_creation",
    var_embd="embeddings"
):
    """
    Deduplicate rows sharing the same (ID, date_creation), prioritizing rows with non-null embeddings.
    If multiple rows have null embeddings but non-null structured data, keep the first.

    If no duplicate timestamps are found, returns the original DataFrame and prints confirmation.
    """
    duplicated_mask = df_merge.duplicated(subset=[var_id, var_crea], keep=False)
    nb_duplicates = duplicated_mask.sum()

    if nb_duplicates == 0:
        print("No duplicated timestamps found.")
        return df_merge.copy()

    print(f"Found {nb_duplicates} duplicated rows sharing the same (ID, date_creation). Proceeding to deduplicate...")

    # Helper flags
    df_merge['embeddings_is_notnull'] = df_merge[var_embd].notnull().astype(int)
    df_merge['covariables_notnull'] = df_merge[covariables_list].notnull().any(axis=1).astype(int)

    # Sort by priority
    df_merge = df_merge.sort_values(
        by=[var_id, var_crea, 'embeddings_is_notnull', 'covariables_notnull'],
        ascending=[True, True, False, False]
    )

    # Drop duplicates (keep highest priority per group)
    df_cleaned = df_merge.drop_duplicates(subset=[var_id, var_crea], keep='first')

    # Drop helper columns
    df_cleaned = df_cleaned.drop(columns=['embeddings_is_notnull', 'covariables_notnull'])
    print(f"Deduplication done: {nb_duplicates} rows affected.")
    return df_cleaned




########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                   LANDMARK APPROACH                                  #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################



def compute_survival_time(df: pd.DataFrame,
                          var_id: str = 'ID',
                          var_death: str = 'DEATH',
                          var_start: str = 'date_start',
                          var_end: str = 'date_end',
                          var_death_date: str = 'date_death',
                          new_var_time: str = 'T_days',
                          verbose: bool = True):
    """
    Compute survival time T_i = (date_death - date_start) if death occurred,
    else (date_end - date_start). The result is added to df for each patient's
    observations and also returned as a patient-level summary.
    """
    # Ensure datetime format
    for col in [var_start, var_end, var_death_date]:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute T_i for each patient (one row per ID)
    df_surv = (
        df.sort_values(by=var_end)
          .groupby(var_id, as_index=False)
          .last()
          .loc[:, [var_id, var_death, var_start, var_end, var_death_date]]
          .assign(
              **{
                  new_var_time: lambda x: np.where(
                      x[var_death] == 1,
                      (x[var_death_date] - x[var_start]).dt.days,
                      (x[var_end] - x[var_start]).dt.days
                  )
              }
          )
    )

    # Merge survival times back to full df
    df = df.merge(df_surv[[var_id, new_var_time]], on=var_id, how='left')

    if verbose:
        print(f"[compute_survival_time] {df_surv.shape[0]} patients processed.")
        print(f"  Mean T: {df_surv[new_var_time].mean():.1f} days "
              f"({df_surv[new_var_time].std():.1f} SD)")
        print(f"  Deaths: {df_surv[var_death].sum()} / {df_surv.shape[0]}")

    return df, df_surv



def define_landmark_cohort(df: pd.DataFrame,
                           landmark_months: int,
                           var_time: str = 'date_creation',
                           var_id: str = 'ID',
                           var_start: str = 'date_start',
                           var_end: str = 'date_end',
                           var_T: str = 'T_days',
                           window_months: int = 6,
                           verbose: bool = True):
    """
    Build the landmark cohort for a given relative time (in months) since baseline.
    Patients with T_days >= L_days are kept, reports restricted to [L-w, L].
    Returns:
        - df_L : sequential data restricted to [L-w, L]
        - patients_in_study : list of eligible IDs
        - df_gamma : one row per patient with gamma_i(L)
    """

    # Ensure datetime types
    for col in [var_time, var_start, var_end]:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Convert months to days
    L_days = int(landmark_months * 30)
    w_days = int(window_months * 30)

    # Patients still in study at the landmark: T_days >= L_days
    patients_in_study = df.loc[df[var_T] >= L_days, var_id].unique().tolist()

    # Subset: keep only patients at risk
    df_sub = df[df[var_id].isin(patients_in_study)].copy()

    # Compute days since start
    df_sub['days_since_start'] = (df_sub[var_time] - df_sub[var_start]).dt.days

    # Apply time window [L-w, L]
    total_obs_before = len(df_sub)
    df_L = df_sub[(df_sub['days_since_start'] >= (L_days - w_days)) &
                  (df_sub['days_since_start'] <= L_days)].copy()
    total_obs_after = len(df_L)
    obs_removed = total_obs_before - total_obs_after
    prop_removed = obs_removed / total_obs_before if total_obs_before > 0 else 0

    # ---------------------------------------------------------------
    # Compute gamma_i(L), but DO NOT MERGE into sequential dataframe
    # ---------------------------------------------------------------
    first_obs = (
        df_L.groupby(var_id)['days_since_start']
        .min()
        .reset_index()
        .rename(columns={'days_since_start': 'first_obs_days'})
    )
    first_obs['gamma'] = ((L_days - first_obs['first_obs_days']) < w_days).astype(int)

    # Prepare gamma output: only ID and gamma
    df_gamma = first_obs[[var_id, 'gamma']].copy()
    # ---------------------------------------------------------------

    # ===============================================================
    # === IMPORTANT LANDMARK FIX: UPDATE RESIDUAL SURVIVAL TIME R ===
    # ===============================================================
    df_L["R"] = df_L[var_T] - L_days
    # ===============================================================

    # ====================================================================
    # === SECOND IMPORTANT FIX: UPDATE EVENT INDICATOR FOR LANDMARKING ===
    # === δ_i(L) = 1 if patient dies after L; 0 otherwise                ===
    # ====================================================================
    df_L["DEATH_L"] = (
        (df_L[var_T] > L_days) &
        (df_L["DEATH"] == 1)
    ).astype(int)
    # ====================================================================

    # --- Verbose summary ---
    if verbose:
        n_total = df[var_id].nunique()
        n_kept = len(patients_in_study)
        n_gamma = df_gamma['gamma'].sum()
        prop_gamma = n_gamma / n_kept if n_kept > 0 else 0

        print(f"[Landmark {landmark_months} mo] Patients kept: {n_kept}/{n_total} "
              f"({n_total - n_kept} excluded)")
        print(f"  → Short-history γ=1: {n_gamma}/{n_kept} = {prop_gamma:.1%}")
        print(f"  → Observations kept: {total_obs_after}/{total_obs_before} "
              f"({100 * (1 - prop_removed):.1f}% retained, {100 * prop_removed:.1f}% removed)")
        print(f"  → Residual survival R_i computed (min={df_L['R'].min()} days).")
        print(f"  → Landmark events computed: DEATH_L sum = {df_L['DEATH_L'].sum()}.")

    return df_L, patients_in_study, df_gamma