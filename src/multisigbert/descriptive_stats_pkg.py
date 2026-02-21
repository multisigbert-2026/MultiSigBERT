import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np


def print_dataset_statistics(df):
    """
    Computes and prints basic descriptive statistics about the survival dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the columns:
        - 'ID' : patient identifier
        - 'DEATH' : 1 if the patient is deceased, 0 if censored

    Prints
    ------
    - Total number of patients
    - Total number of medical reports
    - Average number of reports per patient
    - Number of deceased patients
    - Number of censored patients
    """
    # Compute statistics
    num_unique_patients = df['ID'].nunique()
    num_total_reports = len(df)
    mean_reports_per_patient = df.groupby('ID').size().mean()
    num_deceased = df[df['DEATH'] == 1]['ID'].nunique()
    num_censored = df[df['DEATH'] == 0]['ID'].nunique()

    # Print results
    print(f"Total number of patients in the dataset: {num_unique_patients}")
    print(f"Total number of medical reports: {num_total_reports}")
    print(f"Average number of reports per patient: {mean_reports_per_patient:.2f}")
    print(f"Number of deceased patients: {num_deceased}")
    print(f"Number of censored patients: {num_censored}")


def plot_report_distribution_per_patient(df, export_path=None):
    """
    Plots the distribution of the number of reports per patient.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column 'ID' indicating patient identifiers.
    export_path : str or None, default=None
        If provided, saves the plot to this path as a PNG file.
    """
    report_counts = df['ID'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.histplot(report_counts, bins=20, kde=True, color='green')

    plt.xlabel("Number of reports per patient", fontsize=13)
    plt.ylabel("Frequency", fontsize=13)
    plt.title("Distribution of number of reports per patient", fontsize=15)
    plt.grid(True)

    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')

    plt.show()


def load_dataframe_and_extract_unique_ids(
    file_path,
    id_column,
    verbose=True
):
    """
    Load a CSV file and extract unique non-null identifiers from a given column.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    id_column : str
        Name of the identifier column.
    verbose : bool, optional
        If True, prints basic information.

    Returns
    -------
    df : pandas.DataFrame
        Loaded DataFrame.
    unique_ids : list
        List of unique identifiers.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the identifier column is missing.
    """

    # Check file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    # Load DataFrame
    df = pd.read_csv(file_path)

    if verbose:
        print(f"DataFrame loaded successfully. Shape: {df.shape}")

    # Check identifier column
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame.")

    # Extract unique non-null identifiers
    unique_ids = df[id_column].dropna().unique().tolist()

    if verbose:
        print(f"Number of unique IDs in column '{id_column}': {len(unique_ids)}")
        print("First 10 IDs:", unique_ids[:10])

    return df, unique_ids




def build_cancer_category_dataframe(
    df_id_source,
    id_column_source,
    path_fiche_tumeur_csv,
    fiche_id_column,
    topo_column,
    path_mapping_excel,
    mapping_topo_column,
    mapping_category_column,
    df_reference,
    reference_id_column,
    fiche_sep=";",
    rename_to_id="ID",
    category_replacements=None,
    verbose=True
):
    """
    Build a patient-level cancer category DataFrame and compute ID overlap
    with a reference dataset.

    Parameters
    ----------
    df_id_source : pandas.DataFrame
        DataFrame containing the patient IDs of interest.
    id_column_source : str
        Column name in `df_id_source` containing patient IDs.
    path_fiche_tumeur_csv : str
        Path to the tumor registry CSV file.
    fiche_id_column : str
        Name of the ID column in the tumor registry file.
    topo_column : str
        Column in tumor registry containing topography codes.
    path_mapping_excel : str
        Path to the cancer label mapping Excel file.
    mapping_topo_column : str
        Column in mapping file used as topography key.
    mapping_category_column : str
        Column in mapping file containing cancer categories.
    df_reference : pandas.DataFrame
        Reference DataFrame used to compute ID overlap.
    reference_id_column : str
        ID column name in `df_reference`.
    fiche_sep : str, optional
        Separator used in tumor CSV file (default=";").
    rename_to_id : str, optional
        Unified ID column name (default="ID").
    category_replacements : dict, optional
        Dictionary for category renaming.
    verbose : bool, optional
        If True, prints overlap statistics.

    Returns
    -------
    df_cancer_id : pandas.DataFrame
        DataFrame with columns [ID, Categorie_generale].
    nb_common_ids : int
        Number of common IDs with reference dataset.
    percentage_common : float
        Percentage of overlap relative to reference unique IDs.
    """

    # ------------------------------------------------------------------
    # Extract conform IDs
    # ------------------------------------------------------------------
    df_id_source = df_id_source.rename(columns={id_column_source: rename_to_id})
    id_list = df_id_source[rename_to_id].dropna().tolist()

    # ------------------------------------------------------------------
    # Load tumor registry file
    # ------------------------------------------------------------------
    df_ft = pd.read_csv(path_fiche_tumeur_csv, sep=fiche_sep)
    df_ft = df_ft.rename(columns={fiche_id_column: rename_to_id})

    # Filter on conform IDs
    df_ft = df_ft[df_ft[rename_to_id].isin(id_list)]

    # ------------------------------------------------------------------
    # Load mapping file
    # ------------------------------------------------------------------
    df_mapping = pd.read_excel(path_mapping_excel)

    if category_replacements is not None:
        df_mapping[mapping_category_column] = df_mapping[
            mapping_category_column
        ].replace(category_replacements)

    # ------------------------------------------------------------------
    # Keep last tumor record per patient
    # ------------------------------------------------------------------
    df_diagno = df_ft.groupby(rename_to_id).last().reset_index()

    # Create short topography code (first 3 characters)
    df_diagno["TOPO_SHORT"] = df_diagno[topo_column].astype(str).str[:3]

    # Merge mapping
    df_diagno = df_diagno.merge(
        df_mapping[[mapping_topo_column, mapping_category_column]],
        left_on="TOPO_SHORT",
        right_on=mapping_topo_column,
        how="left"
    )

    # ------------------------------------------------------------------
    # Build cancer ID DataFrame
    # ------------------------------------------------------------------
    df_cancer_id = df_diagno[[rename_to_id, mapping_category_column]].copy()

    # ------------------------------------------------------------------
    # Compute ID overlap
    # ------------------------------------------------------------------
    common_ids = set(df_reference[reference_id_column]).intersection(
        set(df_cancer_id[rename_to_id])
    )

    nb_common_ids = len(common_ids)
    total_reference_ids = df_reference[reference_id_column].nunique()
    percentage_common = (
        (nb_common_ids / total_reference_ids) * 100
        if total_reference_ids > 0 else 0.0
    )

    if verbose:
        print(f"Number of common IDs: {nb_common_ids}")
        print(f"Percentage of common IDs: {percentage_common:.2f}%")

    return df_cancer_id, nb_common_ids, percentage_common



def encode_and_complete_cancer_categories(
    df_cancer_id,
    id_column,
    category_column,
    df_reference,
    reference_id_column,
    translation_dict,
    default_category="Other or unknown",
    good_id_list=None,
    verbose=True
):
    """
    Translate cancer categories to English, fill missing categories,
    complete missing IDs from a reference dataset, and optionally filter
    on a predefined list of valid IDs.

    Parameters
    ----------
    df_cancer_id : pandas.DataFrame
        DataFrame containing patient IDs and cancer categories.
    id_column : str
        Column name containing patient IDs in `df_cancer_id`.
    category_column : str
        Column name containing cancer categories.
    df_reference : pandas.DataFrame
        Reference DataFrame used to identify missing IDs.
    reference_id_column : str
        Column name of patient IDs in `df_reference`.
    translation_dict : dict
        Dictionary mapping original category labels to translated labels.
    default_category : str, optional
        Default category assigned to missing or unmapped values.
    good_id_list : list or set, optional
        If provided, restricts the final DataFrame to these IDs.
    verbose : bool, optional
        If True, prints summary statistics.

    Returns
    -------
    df_cancer_id_encoded : pandas.DataFrame
        Processed DataFrame with completed and translated categories.
    """

    df = df_cancer_id.copy()

    # ------------------------------------------------------------------
    # Translate categories
    # ------------------------------------------------------------------
    df[category_column] = df[category_column].map(translation_dict)

    # Fill missing translations
    df[category_column] = df[category_column].fillna(default_category)

    # ------------------------------------------------------------------
    # Identify missing IDs relative to reference
    # ------------------------------------------------------------------
    reference_ids = set(df_reference[reference_id_column])
    current_ids = set(df[id_column])

    common_ids = reference_ids.intersection(current_ids)
    missing_ids = reference_ids - common_ids

    # Create missing ID rows with default category
    if len(missing_ids) > 0:
        df_missing = pd.DataFrame({
            id_column: list(missing_ids),
            category_column: default_category
        })

        df = pd.concat([df, df_missing], ignore_index=True)

    # ------------------------------------------------------------------
    # Optional filtering on good ID list
    # ------------------------------------------------------------------
    if good_id_list is not None:
        df = df[df[id_column].isin(good_id_list)]

    if verbose:
        print(f"Number of reference IDs: {len(reference_ids)}")
        print(f"Number of common IDs: {len(common_ids)}")
        print(f"Number of missing IDs added: {len(missing_ids)}")
        print(f"Final dataset size: {df.shape}")

    return df



def plot_cancer_distribution(
    df,
    category_column,
    figsize=(10, 6),
    palette="viridis",
    title="Distribution of Cancer Types",
    xlabel="Number of Patients",
    ylabel="Cancer Type",
    xlim=None,
    grid=True,
    save_path=None,
    dpi=300,
    bbox_inches="tight",
    verbose=True
):
    """
    Plot and optionally save a horizontal bar chart of cancer category distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cancer categories.
    category_column : str
        Column name containing categorical cancer labels.
    figsize : tuple, optional
        Figure size (width, height).
    palette : str, optional
        Seaborn color palette.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    xlim : tuple or None, optional
        Limits for the X-axis (e.g., (0, 820)).
    grid : bool, optional
        Whether to display gridlines on the X-axis.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    dpi : int, optional
        Resolution for saved figure.
    bbox_inches : str, optional
        Bounding box option for saving.
    verbose : bool, optional
        If True, prints summary information.

    Returns
    -------
    cancer_counts : pandas.Series
        Series containing category counts.
    """

    # ------------------------------------------------------------------
    # Compute category counts
    # ------------------------------------------------------------------
    cancer_counts = df[category_column].value_counts()

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    plt.figure(figsize=figsize)
    sns.barplot(
        y=cancer_counts.index,
        x=cancer_counts.values,
        palette=palette
    )

    # Add value labels at the end of bars
    for index, value in enumerate(cancer_counts.values):
        plt.text(value + max(cancer_counts.values) * 0.01,
                 index,
                 str(value),
                 va='center')

    # Titles and labels
    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.yticks(fontsize=15)

    if xlim is not None:
        plt.xlim(xlim)

    if grid:
        plt.grid(axis='x', linestyle='--', alpha=0.4)

    # ------------------------------------------------------------------
    # Save figure if requested
    # ------------------------------------------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)

        if verbose:
            print(f"Figure saved to: {save_path}")

    plt.show()

    if verbose:
        print("Cancer distribution plot generated successfully.")

    return cancer_counts




def plot_translated_source_distribution(
    df,
    source_column,
    regroup_dict,
    translation_dict,
    figsize=(10, 6),
    title="Distribution of Report Types",
    xlabel="Number of Occurrences (Frequency)",
    ylabel="Report Type",
    xlim=None,
    save_path=None,
    dpi=300,
    bbox_inches="tight",
    verbose=True
):
    """
    Standardize, translate, and plot the distribution of report sources.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing report sources.
    source_column : str
        Column name containing source labels.
    regroup_dict : dict
        Dictionary used to merge similar source labels.
    translation_dict : dict
        Dictionary mapping French labels to English labels.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    xlim : tuple or None, optional
        X-axis limits.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    dpi : int, optional
        Resolution for saved figure.
    bbox_inches : str, optional
        Bounding box option when saving.
    verbose : bool, optional
        If True, prints summary information.

    Returns
    -------
    source_counts : pandas.Series
        Frequency of each translated source.
    """

    df_plot = df.copy()

    # ------------------------------------------------------------------
    # Regroup similar labels
    # ------------------------------------------------------------------
    df_plot[source_column] = df_plot[source_column].replace(regroup_dict)

    # ------------------------------------------------------------------
    # Translate to English
    # ------------------------------------------------------------------
    df_plot[source_column] = (
        df_plot[source_column]
        .map(translation_dict)
        .fillna(df_plot[source_column])
    )

    # ------------------------------------------------------------------
    # Compute frequencies
    # ------------------------------------------------------------------
    source_counts = df_plot[source_column].value_counts()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    colors = plt.cm.viridis(
        np.linspace(0.7, 0.2, len(source_counts))
    )

    plt.figure(figsize=figsize)
    bars = plt.barh(
        source_counts.index,
        source_counts.values,
        color=colors,
        edgecolor="black"
    )

    x_max = source_counts.max()

    if xlim is None:
        plt.xlim([0, x_max * 1.1])
    else:
        plt.xlim(xlim)

    # Add counts at bar ends
    for bar, count in zip(bars, source_counts.values):
        plt.text(
            bar.get_width() + x_max * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center"
        )

    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.yticks(fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save if requested
    # ------------------------------------------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)

        if verbose:
            print(f"Figure saved to: {save_path}")

    plt.show()

    if verbose:
        print("Source distribution plotted successfully.")

    return source_counts



def plot_reports_per_patient_distribution(
    df,
    id_column,
    bins=100,
    kde=True,
    color="green",
    figsize=(10, 6),
    title="Distribution of number of reports per patient",
    xlabel="Number of reports per patient",
    ylabel="Frequency",
    save_path=None,
    dpi=300,
    bbox_inches="tight",
    verbose=True
):
    """
    Plot the distribution of the number of reports per patient.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing patient IDs.
    id_column : str
        Column name corresponding to patient identifiers.
    bins : int, optional
        Number of histogram bins.
    kde : bool, optional
        Whether to overlay a kernel density estimate.
    color : str, optional
        Histogram color.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    dpi : int, optional
        Resolution for saved figure.
    bbox_inches : str, optional
        Bounding box option when saving.
    verbose : bool, optional
        If True, prints summary statistics.

    Returns
    -------
    report_counts : pandas.Series
        Number of reports per patient.
    """

    # ------------------------------------------------------------------
    # Compute number of reports per patient
    # ------------------------------------------------------------------
    report_counts = df[id_column].value_counts()

    # ------------------------------------------------------------------
    # Plot histogram
    # ------------------------------------------------------------------
    plt.figure(figsize=figsize)
    sns.histplot(report_counts, bins=bins, kde=kde, color=color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    # ------------------------------------------------------------------
    # Save figure if requested
    # ------------------------------------------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)

        if verbose:
            print(f"Figure saved to: {save_path}")

    plt.show()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    if verbose:
        print(f"Total number of patients: {report_counts.shape[0]}")
        print(
            f"Mean number of reports per patient: "
            f"{report_counts.mean():.2f} (sd {report_counts.std():.2f})."
        )
        print(f"Median number of reports per patient: {report_counts.median():.2f}")
        print(f"Maximum number of reports for a single patient: {report_counts.max()}")

    return report_counts


def make_df_info(df, columns_to_keep, var_id='ID', var_start='date_start', var_end='date_end', verbose=True):
    """
    Process a DataFrame to keep only the last observation per patient and selected columns,
    and compute the duration in days between start and end dates.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing longitudinal patient data.
    - columns_to_keep (list): List of columns to retain in the output.
    - var_id (str): Name of the patient ID column.
    - var_start (str): Name of the start date column.
    - var_end (str): Name of the end date column.
    - verbose (bool): Whether to print the number of unique patients.

    Returns:
    - pd.DataFrame: Processed DataFrame with one row per patient and a 'duration' column.
    """
    # Keep only the last observation per patient (assuming df is sorted chronologically)
    df_last = df.groupby(var_id).tail(1).copy()

    # Keep only the requested columns
    df_last = df_last[columns_to_keep].copy()

    # Compute duration in days
    df_last['duration'] = (df_last[var_end] - df_last[var_start]).dt.total_seconds() / (3600 * 24)

    # Print number of unique patients
    if verbose:
        print(f"Number of patients: {df[var_id].nunique()}")

    return df_last


def describe_and_plot_earliest_diagnosis_dates(
    df_ft,
    id_column,
    diag_date_column,
    compliant_ids=None,
    date_strip=True,
    errors="coerce",
    bins=50,
    color="#2C43E9",
    alpha=0.7,
    figsize=(10, 6),
    title="Distribution of Diagnosis Dates",
    xlabel="Year of Diagnosis",
    ylabel="Number of Patients",
    rotation=45,
    save_path=None,
    dpi=300,
    bbox_inches="tight",
    verbose=True
):
    """
    Convert diagnosis dates to datetime, keep the earliest diagnosis date per patient,
    compute descriptive date statistics, and plot the date distribution.

    Parameters
    ----------
    df_ft : pandas.DataFrame
        Tumor registry DataFrame containing diagnosis dates.
    id_column : str
        Patient ID column name.
    diag_date_column : str
        Diagnosis date column name (string or datetime-like).
    compliant_ids : list or set or None, optional
        If provided, restricts the analysis to these IDs.
    date_strip : bool, optional
        If True, strips whitespace before datetime conversion (for string columns).
    errors : str, optional
        Error handling passed to pandas.to_datetime (default="coerce").
    bins : int, optional
        Number of histogram bins.
    color : str, optional
        Histogram color.
    alpha : float, optional
        Histogram transparency.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    rotation : int, optional
        Rotation of x-tick labels.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    dpi : int, optional
        Resolution of saved figure.
    bbox_inches : str, optional
        Bounding box option for saving.
    verbose : bool, optional
        If True, prints descriptive statistics.

    Returns
    -------
    df_earliest : pandas.DataFrame
        DataFrame reduced to one row per patient (earliest diagnosis date).
    stats : dict
        Dictionary with descriptive statistics (min, max, mode, percentiles).
    """

    df = df_ft.copy()

    # ------------------------------------------------------------------
    # Convert to datetime
    # ------------------------------------------------------------------
    series = df[diag_date_column]

    if date_strip and series.dtype == "object":
        series = series.astype(str).str.strip()

    df[diag_date_column] = pd.to_datetime(series, errors=errors)

    # ------------------------------------------------------------------
    # Keep earliest diagnosis per patient
    # ------------------------------------------------------------------
    df_earliest = (
        df.dropna(subset=[diag_date_column])
          .sort_values(diag_date_column)
          .drop_duplicates(subset=[id_column], keep="first")
    )

    # ------------------------------------------------------------------
    # Apply compliant ID filtering if requested
    # ------------------------------------------------------------------
    if compliant_ids is not None:
        df_earliest = df_earliest[df_earliest[id_column].isin(compliant_ids)]

    # ------------------------------------------------------------------
    # Descriptive statistics (computed on earliest dates)
    # ------------------------------------------------------------------
    stats = {
        "earliest_date": df_earliest[diag_date_column].min(),
        "latest_date": df_earliest[diag_date_column].max(),
        "mode_date": None,
        "p5_date": df_earliest[diag_date_column].quantile(0.05),
        "p50_date": df_earliest[diag_date_column].quantile(0.50),
        "p95_date": df_earliest[diag_date_column].quantile(0.95),
        "n_patients": df_earliest[id_column].nunique()
    }

    mode_series = df_earliest[diag_date_column].mode()
    if mode_series.shape[0] > 0:
        stats["mode_date"] = mode_series.iloc[0]

    if verbose:
        print(f"Earliest date: {stats['earliest_date']}")
        print(f"Latest date: {stats['latest_date']}")
        print(f"Mode (most frequent date): {stats['mode_date']}")
        print(f"5th percentile (P5): {stats['p5_date']}")
        print(f"Median date (P50): {stats['p50_date']}")
        print(f"95th percentile (P95): {stats['p95_date']}")
        print(f"Number of patients: {stats['n_patients']}")

    # ------------------------------------------------------------------
    # Plot histogram (on earliest dates)
    # ------------------------------------------------------------------
    plt.figure(figsize=figsize)
    sns.histplot(df_earliest[diag_date_column], bins=bins, color=color, alpha=alpha)

    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(rotation=rotation)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        if verbose:
            print(f"Figure saved to: {save_path}")

    plt.show()

    return df_earliest, stats


def plot_event_duration_distribution(
    df_info,
    event_column,
    duration_column,
    event_value=1,
    bins=50,
    color="#B21D61",
    alpha=0.7,
    figsize=(10, 6),
    title="Distribution of study duration for patients for whom the event occurred",
    xlabel="Study duration (days)",
    ylabel="Number of patients",
    save_path=None,
    dpi=300,
    bbox_inches="tight",
    verbose=True
):
    """
    Plot the duration distribution for patients experiencing the event
    and compute summary statistics.

    Parameters
    ----------
    df_info : pandas.DataFrame
        DataFrame containing survival information.
    event_column : str
        Column indicating event occurrence (e.g., 1 = event).
    duration_column : str
        Column containing survival duration.
    event_value : int or float, optional
        Value indicating that the event occurred.
    bins : int, optional
        Number of histogram bins.
    color : str, optional
        Histogram color.
    alpha : float, optional
        Transparency level.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    save_path : str or None, optional
        If provided, saves the figure.
    dpi : int, optional
        Resolution of saved figure.
    bbox_inches : str, optional
        Bounding box option for saving.
    verbose : bool, optional
        If True, prints summary statistics.

    Returns
    -------
    data_event : pandas.DataFrame
        Filtered DataFrame containing only event cases.
    summary_stats : pandas.Series
        Descriptive statistics of the duration variable.
    """

    # ------------------------------------------------------------------
    # Filter event cases
    # ------------------------------------------------------------------
    data_event = df_info[df_info[event_column] == event_value]

    # ------------------------------------------------------------------
    # Plot histogram
    # ------------------------------------------------------------------
    plt.figure(figsize=figsize)
    sns.histplot(
        data_event[duration_column],
        bins=bins,
        color=color,
        alpha=alpha
    )

    plt.title(title, fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # ------------------------------------------------------------------
    # Save figure if requested
    # ------------------------------------------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)

        if verbose:
            print(f"Figure saved to: {save_path}")

    plt.show()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    summary_stats = data_event[duration_column].describe()

    if verbose:
        print("Summary statistics for event cases:")
        print(summary_stats)

    return data_event, summary_stats



def plot_km_train_vs_validation_with_logrank(
    df_info,
    id_column,
    duration_column,
    event_column,
    df_train,
    test_groups,
    conform_ids,
    train_id_column=None,
    test_id_column=None,
    train_label="Train",
    validation_label="Validation",
    figsize=(8, 6),
    ci_show=True,
    ci_alpha=0.3,
    xlim=None,
    title="Kaplan-Meier Curves: Train vs Validation",
    xlabel="Time (days)",
    ylabel="Survival Probability",
    grid_alpha=0.4,
    save_path=None,
    dpi=300,
    bbox_inches="tight",
    verbose=True
):
    """
    Plot Kaplan-Meier curves for train vs validation cohorts (filtered by conforming IDs),
    compute a log-rank test, and report median event time among event cases.

    Parameters
    ----------
    df_info : pandas.DataFrame
        DataFrame containing survival information (one row per patient recommended).
    id_column : str
        Patient ID column in `df_info`.
    duration_column : str
        Duration column in `df_info`.
    event_column : str
        Event indicator column in `df_info` (1 = event occurred).
    df_train : pandas.DataFrame
        Training DataFrame containing patient IDs.
    test_groups : list
        List of validation/test DataFrames containing patient IDs.
    conform_ids : list or set
        List/set of IDs to keep (e.g., conforming patients).
    train_id_column : str or None, optional
        ID column name in `df_train`. If None, defaults to `id_column`.
    test_id_column : str or None, optional
        ID column name in each df in `test_groups`. If None, defaults to `id_column`.
    train_label : str, optional
        Label used in the Kaplan-Meier plot for the training cohort.
    validation_label : str, optional
        Label used in the Kaplan-Meier plot for the validation cohort.
    figsize : tuple, optional
        Figure size.
    ci_show : bool, optional
        Whether to display confidence intervals.
    ci_alpha : float, optional
        Transparency for confidence intervals.
    xlim : tuple or None, optional
        X-axis limits, e.g. (110, 10000).
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    grid_alpha : float, optional
        Grid transparency.
    save_path : str or None, optional
        If provided, saves the figure to this path.
    dpi : int, optional
        Resolution for saved figure.
    bbox_inches : str, optional
        Bounding box option for saving.
    verbose : bool, optional
        If True, prints log-rank results and median event times.

    Returns
    -------
    df_info_train : pandas.DataFrame
        Filtered `df_info` for training cohort.
    df_info_val : pandas.DataFrame
        Filtered `df_info` for validation cohort.
    logrank_results : lifelines.statistics.StatisticalResult
        Log-rank test results object.
    median_event_time_train : float
        Median duration among event cases in train cohort.
    median_event_time_val : float
        Median duration among event cases in validation cohort.
    """

    train_id_column = id_column if train_id_column is None else train_id_column
    test_id_column = id_column if test_id_column is None else test_id_column

    # ------------------------------------------------------------------
    # Filter IDs for train and validation based on conforming IDs
    # ------------------------------------------------------------------
    conform_ids_set = set(conform_ids)

    train_ids = set(df_train[train_id_column].dropna().unique())
    train_ids_conform = list(train_ids.intersection(conform_ids_set))

    val_ids = set()
    for df_test in test_groups:
        val_ids.update(df_test[test_id_column].dropna().unique())
    val_ids_conform = list(val_ids.intersection(conform_ids_set))

    # ------------------------------------------------------------------
    # Subset df_info for KM fitting
    # ------------------------------------------------------------------
    df_info_train = df_info[df_info[id_column].isin(train_ids_conform)].copy()
    df_info_val = df_info[df_info[id_column].isin(val_ids_conform)].copy()

    # ------------------------------------------------------------------
    # Fit Kaplan-Meier curves
    # ------------------------------------------------------------------
    kmf_train = KaplanMeierFitter()
    kmf_val = KaplanMeierFitter()

    kmf_train.fit(
        df_info_train[duration_column],
        event_observed=df_info_train[event_column],
        label=train_label
    )
    kmf_val.fit(
        df_info_val[duration_column],
        event_observed=df_info_val[event_column],
        label=validation_label
    )

    # ------------------------------------------------------------------
    # Log-rank test
    # ------------------------------------------------------------------
    logrank_results = logrank_test(
        df_info_train[duration_column],
        df_info_val[duration_column],
        event_observed_A=df_info_train[event_column],
        event_observed_B=df_info_val[event_column]
    )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    colors = plt.cm.viridis(np.linspace(0.7, 0.2, 2))

    plt.figure(figsize=figsize)
    kmf_val.plot(ci_show=ci_show, ci_alpha=ci_alpha, color=colors[0])
    kmf_train.plot(ci_show=ci_show, ci_alpha=ci_alpha, color=colors[1])

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if xlim is not None:
        plt.xlim(xlim)

    plt.legend(fontsize=12)
    plt.grid(alpha=grid_alpha)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        if verbose:
            print(f"Figure saved to: {save_path}")

    plt.show()

    # ------------------------------------------------------------------
    # Median event time among event cases
    # ------------------------------------------------------------------
    median_event_time_train = (
        df_info_train.loc[df_info_train[event_column] == 1, duration_column].median()
    )
    median_event_time_val = (
        df_info_val.loc[df_info_val[event_column] == 1, duration_column].median()
    )

    if verbose:
        print(
            "Log-rank summary:\n"
            f"{logrank_results.summary}\n"
            f"Log-rank p-value = {logrank_results.p_value:.4f}\n"
        )
        print(
            f"Median event time ({train_label}) for {event_column} == 1: "
            f"{median_event_time_train:.2f} days"
        )
        print(
            f"Median event time ({validation_label}) for {event_column} == 1: "
            f"{median_event_time_val:.2f} days"
        )
        print(f"Train cohort size: {df_info_train.shape[0]} patients")
        print(f"Validation cohort size: {df_info_val.shape[0]} patients")

    return (
        df_info_train,
        df_info_val,
        logrank_results,
        median_event_time_train,
        median_event_time_val
    )