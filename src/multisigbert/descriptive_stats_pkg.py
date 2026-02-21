import matplotlib.pyplot as plt
import seaborn as sns

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