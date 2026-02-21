########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                       METRICS                                        #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################

import pandas as pd
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway, kruskal, pearsonr, spearmanr, ttest_ind, skew, normaltest, kstest, gamma, weibull_min, lognorm

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from sklearn.linear_model import HuberRegressor
from sksurv.metrics import cumulative_dynamic_auc


def jackknife_confidence_interval(scores, alpha=0.05):
    """
    Compute a confidence interval using the Jackknife resampling method.

    The Jackknife method estimates the variance of a statistic (e.g., C-index)
    by systematically leaving out one observation at a time. This function
    provides a bias-corrected point estimate and computes the corresponding
    confidence interval under normal approximation.

    Parameters
    ----------
    scores : list or np.ndarray
        List of evaluation scores (e.g., C-index values from multiple test folds).
    alpha : float, default=0.05
        Significance level for the confidence interval (e.g., 0.05 for 95% CI).

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the estimated confidence interval.

    Notes
    -----
    - The normal quantile is fixed to z = 1.96 for 95% CI (useful for small sample size).
    - The result includes a bias correction using the Jackknife estimate.
    - Assumes the score distribution is approximately symmetric.
    """
    n = len(scores)
    scores = np.array(scores)

    # Compute the full-sample mean
    mean_score = np.mean(scores)

    # Compute leave-one-out means
    jackknife_means = np.array([
        (np.sum(scores) - scores[i]) / (n - 1)
        for i in range(n)
    ])

    # Bias correction
    jackknife_mean = np.mean(jackknife_means)
    bias_corrected_mean = n * mean_score - (n - 1) * jackknife_mean

    # Jackknife variance and standard deviation
    jackknife_var = (n - 1) / n * np.sum((jackknife_means - jackknife_mean) ** 2)
    jackknife_std = np.sqrt(jackknife_var)

    # Confidence interval using normal approximation
    z = 1.96  # For 95% CI
    lower_bound = bias_corrected_mean - z * jackknife_std
    upper_bound = bias_corrected_mean + z * jackknife_std

    return lower_bound, upper_bound



def evaluate_correlation(
    df_survival_test_list,
    verbose=True
):
    """
    Evaluates the correlation between predicted risk scores and survival times
    (log-transformed), restricted to deceased patients, using both Pearson and Spearman metrics.

    This function processes a list of test DataFrames, each containing survival information
    (columns 'event', 'time', 'risk_score'), computes correlation statistics for each,
    and summarizes the results across all splits.

    Parameters
    ----------
    df_survival_test_list : list of pd.DataFrame
        A list of DataFrames, one per test fold or group, each containing:
        - 'event': binary indicator (1 = death, 0 = censored)
        - 'time': survival duration
        - 'risk_score': predicted risk score from a model
    verbose : bool, default=True
        If True, prints summary statistics (mean and standard deviation of correlations and p-values).

    Returns
    -------
    results : dict
        Dictionary containing lists of correlation coefficients and p-values across folds:
            - 'pearson_corr'
            - 'pearson_pval'
            - 'spearman_corr'
            - 'spearman_pval'
    summary_stats : dict
        Dictionary containing mean and standard deviation for each correlation metric:
            {metric_name: (mean, std)}
    """

    results = {
        "pearson_corr": [],
        "pearson_pval": [],
        "spearman_corr": [],
        "spearman_pval": []
    }

    # Loop through each test group
    for df_test in df_survival_test_list:
        # Filter to deceased patients only
        df_deceased = df_test[df_test['event'] == 1]

        # Extract risk scores and log-transformed survival times
        risk_scores = df_deceased['risk_score']
        log_survival_times = np.log(df_deceased['time'])

        # Compute Pearson and Spearman correlations
        pearson_corr, pearson_pval = pearsonr(risk_scores, log_survival_times)
        spearman_corr, spearman_pval = spearmanr(risk_scores, log_survival_times)

        # Store results
        results["pearson_corr"].append(pearson_corr)
        results["pearson_pval"].append(pearson_pval)
        results["spearman_corr"].append(spearman_corr)
        results["spearman_pval"].append(spearman_pval)

    # Compute mean and standard deviation for each metric
    summary_stats = {
        key: (np.mean(values), np.std(values)) for key, values in results.items()
    }

    # Display results if verbose
    if verbose:
        print(f"Pearson correlation: {summary_stats['pearson_corr'][0]:.3f} (sd {summary_stats['pearson_corr'][1]:.4f})")
        print(f"Pearson p-value: {summary_stats['pearson_pval'][0]:.3e} (sd {summary_stats['pearson_pval'][1]:.3e})")
        print("")
        print(f"Spearman correlation: {summary_stats['spearman_corr'][0]:.3f} (sd {summary_stats['spearman_corr'][1]:.4f})")
        print(f"Spearman p-value: {summary_stats['spearman_pval'][0]:.3e} (sd {summary_stats['spearman_pval'][1]:.3e})")

    return results, summary_stats




def plot_risk_scatter(
    X: np.ndarray,
    y: np.ndarray,
    w_sk: np.ndarray,
    deceased_color: str = '#E5095C',
    survival_color: str = '#024dda',
    log_scale: bool = False,
    only_deceased: bool = False,
    make_reg_exp: bool = False,
    eps_Huber: float = 1.0,
    epsilon_log: float = 1e-10,
    fig_name: str = './results/risk_scatter.png'
):
    """
    Plot a scatter graph of risk scores vs. study durations, with optional log scaling and robust regression.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix for patients.
    y : np.ndarray
        Array containing events (1 for death, 0 for censoring) and survival durations.
    w_sk : np.ndarray
        Risk coefficients estimated from skglm with LASSO.
    deceased_color : str, default='#E5095C'
        Color for deceased patients.
    survival_color : str, default='#024dda'
        Color for censored patients.
    log_scale : bool, default=False
        If True, use logarithmic scale for the y-axis.
    only_deceased : bool, default=False
        If True, plot only deceased patients.
    make_reg_exp : bool, default=False
        If True, perform Huber regression on log(times) vs. risk scores.
    eps_Huber : float, default=1.0
        Epsilon parameter for HuberRegressor.
    epsilon_log : float, default=1e-10
        Small epsilon added to avoid log(0).
    fig_name : str, default='./results/risk_scatter.png'
        Path to save the resulting figure.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    events = y[:, 0]
    times = y[:, 1]
    risk_scores = np.dot(X, w_sk)

    plt.figure(figsize=(10, 6))

    # Deceased patients
    deceased = (events == 1)
    plt.scatter(risk_scores[deceased], times[deceased], color=deceased_color, label='Deceased')

    if not only_deceased:
        # Censored patients
        surviving = (events == 0)
        plt.scatter(risk_scores[surviving], times[surviving], color=survival_color, label='Survived')

    # Optional regression with Huber
    if make_reg_exp:
        log_times_deceased = np.log(times[deceased] + epsilon_log).reshape(-1, 1)
        model = HuberRegressor(epsilon=eps_Huber).fit(log_times_deceased, risk_scores[deceased])
        predicted_risks = model.predict(log_times_deceased)
        plt.plot(predicted_risks, times[deceased], color='purple', label='Huber Reg (deceased)')

        if not only_deceased:
            valid_times_surviving = times[surviving] > 0
            log_times_surviving = np.log(times[surviving][valid_times_surviving] + epsilon_log).reshape(-1, 1)
            model = HuberRegressor(epsilon=eps_Huber).fit(log_times_surviving, risk_scores[surviving][valid_times_surviving])
            predicted_risks = model.predict(log_times_surviving)
            plt.plot(predicted_risks, times[surviving][valid_times_surviving], color='orange', label='Huber Reg (survived)')

    plt.xlabel('Risk Score')
    plt.ylabel('Study Duration (time)')
    plt.title('Scatter Plot: Risk Score vs. Study Duration')

    if log_scale:
        plt.yscale('log')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

    if fig_name:
        plt.savefig(fig_name)

    plt.show()


def plot_risk_distribution(
    X: np.ndarray,
    y: np.ndarray,
    w_sk: np.ndarray,
    deceased_color: str = '#E5095C',
    survival_color: str = '#024dda'
) -> pd.DataFrame:
    """
    Plot the distribution of risk scores for deceased vs. surviving patients.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix used to compute risk scores.
    y : np.ndarray
        Array containing event indicators (1 if deceased, 0 if censored) and time.
        The first column represents the event status.
    w_sk : np.ndarray
        Coefficient vector used to compute linear risk scores.
    deceased_color : str, default='#E5095C'
        Color used for plotting deceased patients.
    survival_color : str, default='#024dda'
        Color used for plotting censored patients.

    Returns
    -------
    pd.DataFrame
        DataFrame containing risk scores and event status for each patient.
    """
    events = y[:, 0]
    risk_scores = np.dot(X, w_sk)

    df = pd.DataFrame({
        'Risk Score': risk_scores,
        'Status': np.where(events == 1, 'Deceased', 'Survived')
    })

    plt.figure(figsize=(10, 6))

    # Histogram and KDE overlay
    palette = {'Deceased': deceased_color, 'Survived': survival_color}
    sns.histplot(
        data=df,
        x='Risk Score',
        hue='Status',
        element='step',
        stat='density',
        common_norm=False,
        palette=palette
    )
    sns.kdeplot(data=df[df['Status'] == 'Deceased'], x='Risk Score', color=deceased_color, label='Deceased KDE')
    sns.kdeplot(data=df[df['Status'] == 'Survived'], x='Risk Score', color=survival_color, label='Survived KDE')

    plt.xlabel('Risk Score')
    plt.ylabel('Density')
    plt.title('Distribution of Risk Scores by Status')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    return df



def plot_event_time_distribution(
    y: np.ndarray,
    deceased_color: str = '#E5095C',
    survival_color: str = '#024dda'
) -> pd.DataFrame:
    """
    Plot the distribution of event or censoring times by survival status.

    Parameters
    ----------
    y : np.ndarray
        Array containing event indicators and times.
        The first column indicates event status (1 for death, 0 for censoring).
        The second column contains time-to-event or censoring.
    deceased_color : str, default='#E5095C'
        Color for deceased patients.
    survival_color : str, default='#024dda'
        Color for censored (surviving) patients.

    Returns
    -------
    pd.DataFrame
        DataFrame containing event times and status labels.
    """
    events = y[:, 0]
    times = y[:, 1]

    df = pd.DataFrame({
        'Time': times,
        'Status': np.where(events == 1, 'Deceased', 'Survived')
    })

    plt.figure(figsize=(10, 6))

    palette = {'Deceased': deceased_color, 'Survived': survival_color}
    sns.histplot(data=df, x='Time', hue='Status', element='step', stat='density', common_norm=False, palette=palette)

    sns.kdeplot(data=df[df['Status'] == 'Deceased'], x='Time', color=deceased_color, label='Deceased KDE')
    sns.kdeplot(data=df[df['Status'] == 'Survived'], x='Time', color=survival_color, label='Survived KDE')

    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.title('Distribution of Event Times by Status')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

    return df


def ttest_risk_scores(df: pd.DataFrame, alpha_ttest: float = 0.05) -> dict:
    """
    Perform a Student's t-test to compare mean risk scores between deceased and surviving groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing risk scores and patient status.
        Expected columns: 'Risk Score' and 'Status' ('Deceased' or 'Survived').
    alpha_ttest : float, default=0.05
        Significance level for the t-test.

    Returns
    -------
    dict
        Dictionary with test statistics and interpretation.
    """
    # Extract risk scores by group
    risks_deceased = df[df['Status'] == 'Deceased']['Risk Score']
    risks_surviving = df[df['Status'] == 'Survived']['Risk Score']

    # Perform independent two-sample t-test
    t_stat, p_value = ttest_ind(risks_deceased, risks_surviving)

    # Interpretation based on significance level
    if p_value < alpha_ttest:
        conclusion = "Reject the null hypothesis: mean risk scores are statistically different between groups."
    else:
        conclusion = "Fail to reject the null hypothesis: mean risk scores are not statistically different between groups."

    # Display results
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(conclusion)

    return {
        't_statistic': round(t_stat, 3),
        'p_value': round(p_value, 3),
        'conclusion': conclusion
    }



def plot_risk_score_distribution_by_event(
    df_survival,
    deceased_color='#E5095C',
    survival_color='#024dda',
    var_id = 'ID',
    risk_column='risk_score',
    use_ttest=True,
    export_plot=None
):
    """
    Plot the distribution of risk scores stratified by event status (deceased or censored).
    Optionally performs a t-test to compare risk score means between groups.

    Parameters
    ----------
    df_survival : pd.DataFrame
        DataFrame containing at least the columns 'event' and the risk score column.
    deceased_color : str, default='#E5095C'
        Color for patients who experienced the event.
    survival_color : str, default='#024dda'
        Color for censored patients.
    var_id : str, default='ID'
        Column name for patient IDs.
    risk_column : str, default='risk_score'
        Column name containing predicted risk scores.
    use_ttest : bool, default=True
        If True, performs a t-test on the risk scores between groups.
    export_plot : str or None, default=None
        If specified, path to save the figure.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[dict]]
        - The DataFrame used for plotting, with risk score and status.
        - The t-test results dictionary if `use_ttest` is True, else None.
    """
    # Build DataFrame for plotting
    df_plot = pd.DataFrame({
        'ID': df_survival[var_id],
        'Risk Score': df_survival[risk_column],
        'Status': np.where(df_survival['event'] == 1, 'Deceased', 'Censored')
    })

    # Initialize plot
    plt.figure(figsize=(10, 6))
    palette = {'Deceased': deceased_color, 'Censored': survival_color}

    # Plot histogram and KDE
    sns.histplot(
        data=df_plot,
        x='Risk Score',
        hue='Status',
        element='step',
        stat='density',
        common_norm=False,
        palette=palette
    )
    sns.kdeplot(
        data=df_plot[df_plot['Status'] == 'Deceased'],
        x='Risk Score',
        color=deceased_color,
        label='KDE Deceased'
    )
    sns.kdeplot(
        data=df_plot[df_plot['Status'] == 'Censored'],
        x='Risk Score',
        color=survival_color,
        label='KDE Censored'
    )

    # Format plot
    plt.xlabel('Predicted Risk Score')
    plt.ylabel('Density')
    plt.title('Distribution of Risk Scores by Event Status')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

    if export_plot:
        plt.savefig(export_plot, dpi=300, bbox_inches='tight')
    plt.show()

    # Perform t-test if requested
    ttest_results = None
    if use_ttest:
        ttest_results = ttest_risk_scores(df_plot)
    
    return df_plot, ttest_results




def plot_risk_score_histogram(
    df_survival,
    risk_column='risk_score',
    color='green',
    export_plot=None,
    signif=6,
    do_ttest=True,
    alpha_ttest=0.05
):
    values = df_survival[risk_column].dropna().values

    # Descriptive statistics
    desc_stats = {
        'n_obs': int(len(values)),
        'mean': np.round(np.mean(values), signif),
        'std': np.round(np.std(values, ddof=1), signif),
        'skewness': np.round(skew(values), signif),
        'min': np.round(np.min(values), signif),
        'Q1': np.round(np.percentile(values, 25), signif),
        'median': np.round(np.median(values), signif),
        'Q3': np.round(np.percentile(values, 75), signif),
        'max': np.round(np.max(values), signif)
    }

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(values, kde=True, color=color, ax=ax)
    ax.set_xlabel('Predicted Risk Score')
    ax.set_ylabel('Count')
    ax.set_title('Risk Score Distribution')
    plt.show()

    # Normality test
    if do_ttest:
        stat, pval_norm = normaltest(values)
        if pval_norm < alpha_ttest:
            print(f"Normality test p-value = {pval_norm:.3g} < {alpha_ttest}. Reject normality assumption.")
        else:
            print(f"Normality test p-value = {pval_norm:.3g} >= {alpha_ttest}. Fail to reject normality assumption.")

        # Fit and KS-test for alternative distributions
        for dist_name, dist in [('Gamma', gamma), ('Weibull', weibull_min), ('LogNormal', lognorm)]:
            params = dist.fit(values)
            # Build cdf function with fitted params
            cdf_func = lambda x: dist.cdf(x, *params)
            stat, pval = kstest(values, cdf_func)
            print(f"{dist_name} KS test p-value = {pval:.3g} (params={params})")

    return desc_stats







def plot_km_by_risk_quartiles(
    df_survival,
    export_fig=False,
    path_export_fig="./KM_risk_quartiles.png",
    time_max_days=3650
):
    """
    Plots Kaplan-Meier survival curves stratified by risk score quartiles
    and performs log-rank tests to assess survival differences between groups.

    Parameters
    ----------
    df_survival : pd.DataFrame
        DataFrame containing the survival data. Must include:
            - 'risk_score': predicted risk score from a survival model
            - 'time': survival duration
            - 'event': event indicator (1 = death, 0 = censored)
    export_fig : bool, default=False
        If True, saves the figure to disk.
    path_export_fig : str, default="./KM_risk_quartiles.png"
        Path to save the figure if `export_fig` is True.
    time_max_days : int, default=3650
        Maximum time in days to show on the x-axis.

    Returns
    -------
    results_pairwise : lifelines.statistics.StatisticalResult
        Log-rank test results comparing Q3 and Q4.
    results_global : lifelines.statistics.StatisticalResult
        Log-rank test results across all quartiles.
    """

    df = df_survival.copy()

    # 1. Divide risk scores into quartiles
    df["Quartile"] = pd.qcut(df["risk_score"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4"])

    # 2. Initialize Kaplan-Meier fitter
    kmf = KaplanMeierFitter()

    # 3. Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    # Store quartile-specific DataFrames for log-rank testing
    quartile_groups = {}

    # 4. Loop through quartiles and plot curves
    for i, quartile in enumerate(df["Quartile"].unique()):
        mask = df["Quartile"] == quartile
        group = df.loc[mask]

        kmf.fit(group["time"], event_observed=group["event"], label=quartile)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i], alpha=0.8)

        quartile_groups[quartile] = group

    # 5. Log-rank test between Q3 and Q4
    results_pairwise = logrank_test(
        quartile_groups["Q3"]["time"], quartile_groups["Q4"]["time"],
        event_observed_A=quartile_groups["Q3"]["event"],
        event_observed_B=quartile_groups["Q4"]["event"]
    )

    # 6. Formatting the plot
    ax.set_title("Kaplan-Meier Survival Curves by Risk Score Quartile", fontsize=17)
    ax.set_xlabel("Time (days)", fontsize=15)
    ax.set_ylabel("Survival Probability", fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(title="Risk Score Quartile", fontsize=12, loc="lower left", frameon=True)

    # 7. Global log-rank test across all quartiles
    results_global = multivariate_logrank_test(
        df["time"],
        df["Quartile"],
        df["event"]
    )

    # Add global p-value text to the plot
    ax.text(
        x=time_max_days * 0.6, y=1,
        s=f"Global Log-rank test p-value: {results_global.p_value:.2e}",
        fontsize=10, bbox=dict(facecolor='white', edgecolor='black')
    )

    # 8. Final adjustments and export
    plt.xlim([0, time_max_days])
    if export_fig:
        plt.savefig(path_export_fig, dpi=300, bbox_inches='tight')

    plt.show()

    return results_pairwise, results_global, quartile_groups




def plot_boxplot_log_time_by_quartile(
    df_survival,
    quartile_groups=None,
    export_fig=False,
    path_export_fig="./results/boxplot_log_time_quartiles.png",
    print_median_time=False
):
    """
    Plots a boxplot of log survival times (for uncensored patients) grouped by risk score quartiles.
    Performs ANOVA and Kruskal-Wallis tests to assess statistical differences across groups.

    Parameters
    ----------
    df_survival : pd.DataFrame
        DataFrame containing the columns:
            - 'event': 1 if the event occurred (death), 0 if censored
            - 'time': survival time
            - 'Quartile': categorical quartile of the risk score (e.g., Q1, Q2, Q3, Q4)
    quartile_groups : dict, optional
        Dictionary of quartile-specific DataFrames. Required if `print_median_time` is True.
    export_fig : bool, default=False
        If True, saves the figure to disk at `path_export_fig`.
    path_export_fig : str, default="./boxplot_log_time_quartiles.png"
        Path where the plot is saved if `export_fig` is True.
    print_median_time : bool, default=False
        If True, prints the median and mean event times per quartile.

    Returns
    -------
    tuple
        (anova_pval, kruskal_pval): the p-values from ANOVA and Kruskal-Wallis tests.
    """

    # 1. Check for 'Quartile' column
    if 'Quartile' not in df_survival.columns:
        raise ValueError("The input DataFrame must contain a 'Quartile' column.")

    # 2. Filter for uncensored patients
    df_deceased = df_survival[df_survival['event'] == 1].copy()
    df_deceased['log_time'] = np.log(df_deceased['time'])

    # 3. Define colors for each quartile (reverse order to match visual preference)
    colors = ["#e7298a", "#7570b3", "#d95f02", "#1b9e77"]  # Q4 to Q1

    # 4. Ensure Quartile is categorical with expected order
    quartile_order = ["Q1 (Low)", "Q2", "Q3", "Q4"]
    df_deceased['Quartile'] = pd.Categorical(df_deceased['Quartile'], categories=quartile_order, ordered=True)

    # 5. Plot boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="Quartile", y="log_time", hue="Quartile",
        data=df_deceased, palette=colors, dodge=False, legend=False
    )

    # 6. Customize plot
    plt.title("Log (Study) Time Distribution by Risk Quartile for Uncensored Individuals", fontsize=15)
    plt.xlabel("Risk Score Quartile", fontsize=15)
    plt.ylabel("log(Study Time)", fontsize=15)
    plt.grid(axis='y', linestyle="--", alpha=0.7)

    # 7. Compute p-values (ANOVA and Kruskal-Wallis)
    quartile_values = [df_deceased[df_deceased["Quartile"] == q]["log_time"]
                       for q in quartile_order if q in df_deceased["Quartile"].unique()]
    anova_pval = f_oneway(*quartile_values).pvalue
    kruskal_pval = kruskal(*quartile_values).pvalue

    # 8. Display p-values on the plot
    plt.text(
        x=2.8,
        y=df_deceased["log_time"].max() * 0.9,
        s=f"ANOVA p-value: {anova_pval:.2e}\nKruskal-Wallis p-value: {kruskal_pval:.2e}",
        fontsize=10,
        bbox=dict(facecolor='white', edgecolor='black')
    )

    # 9. Add legend manually
    handles = [plt.Line2D([0], [0], marker='s', color='w',
                          markerfacecolor=colors[i], markersize=10)
               for i in range(4)]
    plt.legend(handles, quartile_order, title="Quartiles",
               bbox_to_anchor=(1.1, 0.2), loc="center", frameon=True)

    # 10. Save or show
    if export_fig:
        plt.savefig(path_export_fig, dpi=300, bbox_inches='tight')
    plt.show()

    # 11. Optionally print median/mean event time per quartile
    if print_median_time:
        if quartile_groups is None:
            raise ValueError("quartile_groups must be provided if print_median_time is True.")
        for quartile in quartile_order:
            if quartile not in quartile_groups:
                continue
            group = quartile_groups[quartile]
            total = group.shape[0]
            n_events = group["event"].sum()
            print(f"Number of uncensored individuals in {quartile}: {n_events} out of {total}")
            median_time = group.loc[group["event"] == 1, "time"].median()
            mean_time = group.loc[group["event"] == 1, "time"].mean()
            print(f"{quartile} - Median event time: {median_time:.2f} days")
            print(f"{quartile} - Mean event time: {mean_time:.2f} days\n")

    return anova_pval, kruskal_pval



########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                   Time-dependant AUC                                 #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################

def time_dep_AUC(
    y,
    y_test,
    df_survival_test,
    Ndays = 0,
    plot_curve = True
):
    """
    Compute time-dependent AUCs for survival data using cumulative dynamic AUC.

    Parameters
    ----------
    y : pd.DataFrame
        Training dataset containing 'time' and 'event' columns.
    y_test : pd.DataFrame
        Test dataset containing 'time' and 'event' columns.
    df_survival_test : pd.DataFrame
        Test dataset containing survival predictions including 'risk_score'.
    Ndays : int, default=0
        Minimum time from which to start computing AUC.
    plot_curve : bool, default=True
        If True, plot the time-dependent AUC curve.

    Returns
    -------
    mean_auc : float
        Mean cumulative dynamic AUC.
    Nstart : int
        Start time used after internal adjustments.
    times : np.ndarray
        Array of time points where AUC was evaluated.
    time_auc : np.ndarray
        Array of time-dependent AUC values.
    """
    # Filter for uncensored events only
    y_censored = y[y['event'] == 1]
    y_test_censored = y_test[y_test['event'] == 1]

    # Define time range based on uncensored cases
    T_max = min(y_censored['time'].max(), y_test_censored['time'].max())
    T_min = max(y_censored['time'].min(), y_test_censored['time'].min())

    # Filter datasets within this time window
    y_filtered = y[(y['time'] >= T_min) & (y['time'] <= T_max)]
    mask_test = (y_test['time'] >= T_min) & (y_test['time'] <= T_max)
    y_test_filtered = y_test[mask_test]
    risk_scores_test_filtered = df_survival_test['risk_score'][mask_test]

    # Initialize time grid for AUC evaluation
    Nstart = Ndays
    times = np.arange(Nstart, 8000, 1)

    # Try computing cumulative dynamic AUC; adjust Nstart if necessary
    while True:
        try:
            time_auc, mean_auc = cumulative_dynamic_auc(
                y_filtered,
                y_test_filtered,
                risk_scores_test_filtered,
                times
            )
            break
        except ValueError as e:
            if "all times must be within follow-up time of test data" in str(e):
                Nstart += 1
                times = np.arange(Nstart, 8000, 1)
            else:
                raise e

    # Plot AUC curve
    if plot_curve:
        plt.plot(times, time_auc, label="Time-dependent AUC", color='green')
        plt.axhline(mean_auc, linestyle='-.', color='red', label=f"Mean AUC: {mean_auc:.3f}")
        plt.xlabel("Time")
        plt.ylabel("AUC")
        plt.title("Time-dependent Concordance (AUC)")
        plt.legend()
        plt.grid()
        plt.show()

    return mean_auc, Nstart, times, time_auc


def plot_dynamic_auc(
    y,
    df_survival_test_list,
    test_groups,
    Nstart=0,
    xlim=None,
    ylim=None,
    y_time_max = 3800,
    export_fig=None,
    verbose=True
):
    """
    Computes and plots time-dependent AUC curves across multiple test sets using the cumulative dynamic AUC method.

    Parameters
    ----------
    y : np.ndarray
        Structured array with fields 'event' and 'time' for the training set.
    df_survival_test_list : list of pd.DataFrame
        List of test DataFrames, each containing 'event', 'time', and 'risk_score'.
    test_groups : list
        Identifiers for each test set.
    Nstart : int, default=0
        Minimum time from which to start AUC evaluation.
    y_time_max : int, default=3800
        Maximum survival time (e.g., administrative censoring at 10 years).
    export_fig : bool, default=None
        Path to export the figure as a PNG file.
    verbose : bool, default=True
        Whether to print intermediate results.

    Returns
    -------
    mean_auc_list : list of float
        Mean AUC for each test set.
    mean_auc_per_time : np.ndarray
        Mean AUC across folds at each time point.
    std_auc_per_time : np.ndarray
        Standard deviation of AUC across folds at each time point.
    times : np.ndarray
        Evaluation time points used for the AUC computation.
    """
    mean_auc_list = []
    dynamic_times = []
    dynamic_auc_list = []

    for i in range(len(test_groups)):
        y_filtered = y[y["time"] <= y_time_max]

        df_survival_test = df_survival_test_list[i]
        df_survival_test = df_survival_test[df_survival_test["time"] <= y_time_max]

        y_test = np.array(list(zip(
            df_survival_test["event"].astype(bool),
            df_survival_test["time"].astype(float)
        )), dtype=[("event", "bool"), ("time", "float")])

        risk_scores_test = df_survival_test["risk_score"].values

        if np.std(risk_scores_test) == 0:
            if verbose:
                print(f"Warning: Constant risk scores in Test {i+1}")
            continue

        min_time = max(y_filtered["time"].min(), y_test["time"].min())
        max_time = min(y_filtered["time"].max(), y_test["time"].max()) - 1e-5
        times = np.arange(max(Nstart, min_time), max_time, 1)
        times = times[times < y_filtered["time"].max()]

        try:
            time_auc, mean_auc = cumulative_dynamic_auc(
                y_filtered,
                y_test,
                risk_scores_test,
                times
            )

            valid_time_auc = time_auc[~np.isnan(time_auc)]
            mean_auc = valid_time_auc.mean() if len(valid_time_auc) > 0 else np.nan

            mean_auc_list.append(mean_auc)
            dynamic_times.append(times)
            dynamic_auc_list.append(time_auc)

            if verbose:
                print(f"Mean AUC for Test {i+1}: {mean_auc:.3f}")

        except Exception as e:
            if verbose:
                print(f"Error computing dynamic AUC for Test {i+1}: {e}")
            continue

    mean_dynamic_auc = np.nanmean(mean_auc_list)
    std_dynamic_auc = np.nanstd(mean_auc_list, ddof=1)

    if verbose:
        print(f"\nMean Dynamic AUC across all tests: {mean_dynamic_auc:.3f}")
        print(f"Standard Deviation of Dynamic AUC: {std_dynamic_auc:.3f}")

    max_length = max(len(auc) for auc in dynamic_auc_list)
    dynamic_auc_list_uniform = np.array([
        np.pad(auc, (0, max_length - len(auc)), constant_values=np.nan)
        for auc in dynamic_auc_list
    ])

    times = dynamic_times[np.argmax([len(t) for t in dynamic_times])]

    mean_auc_per_time = np.nanmean(dynamic_auc_list_uniform, axis=0)
    std_auc_per_time = np.nanstd(dynamic_auc_list_uniform, axis=0)

    ci_factor = 1.96 / np.sqrt(len(mean_auc_list))
    upper_bound = mean_auc_per_time + ci_factor * std_auc_per_time
    lower_bound = mean_auc_per_time - ci_factor * std_auc_per_time

    plt.figure(figsize=(10, 6))
    plt.plot(times, mean_auc_per_time, label="Mean AUC per Time", color="#071AFF")
    plt.fill_between(times, lower_bound, upper_bound, color="#07AAE4", alpha=0.2, label="95% CI")
    plt.axhline(mean_dynamic_auc, linestyle="-.", color="#B21D61",
                label=f"Overall AUC: {mean_dynamic_auc:.3f} (sd {std_dynamic_auc:.3f})")

    plt.xlabel("Time", fontsize=15)
    plt.ylabel("AUC", fontsize=15)
    plt.title("Time-dependent AUC with 95% Confidence Interval", fontsize=17)
    plt.legend(fontsize=12)
    plt.grid()
    # Optional axis limits
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


    if export_fig:
        plt.savefig(export_fig, dpi=300, bbox_inches='tight')

    plt.show()

    return mean_auc_list, mean_auc_per_time, std_auc_per_time, times


def brier_score_ipcw_with_cph(
    df_survival, 
    cph,
    evaluation_times,
    verbose=False
):
    """
    Compute the IPCW-weighted Brier score for a Cox proportional hazards model.

    This function evaluates the time-dependent Brier score at specified time points,
    using inverse probability of censoring weighting (IPCW) to handle right-censored data.

    Parameters
    ----------
    df_survival : pd.DataFrame
        DataFrame containing survival predictions and outcomes. Must include the following columns:
        - 'time': observed follow-up time
        - 'event': event indicator (1 = event occurred, 0 = censored)
        - 'ID': patient identifier
        - 'risk_score': risk score output from the Cox model (linear predictor)
    cph : lifelines.CoxPHFitter
        A trained Cox proportional hazards model from the `lifelines` package.
    evaluation_times : array-like
        Array of time points at which to evaluate the Brier score.
    verbose : bool, default=False
        If True, print progress updates for each computation step.

    Returns
    -------
    brier_scores : np.ndarray
        Array of Brier scores evaluated at each time point in `evaluation_times`.

    Notes
    -----
    - The predicted survival probability at time `t` is computed as:
        Ŝ(t | x) = S₀(t)^exp(risk_score)
    - IPCW weights are derived using the Kaplan-Meier estimator of the censoring distribution.
    - If censoring probability at time `t` is zero, the Brier score is set to NaN to avoid division by zero.
    """
    times = df_survival['time'].values
    events = df_survival['event'].values
    risk_scores = df_survival['risk_score'].values

    # Extract baseline survival function from the trained Cox model
    baseline_survival_function = cph.baseline_survival_.squeeze()

    if verbose:
        print("Step 1: Interpolating baseline survival function...")

    def baseline_survival(t):
        if t <= baseline_survival_function.index.min():
            return baseline_survival_function.iloc[0]
        elif t >= baseline_survival_function.index.max():
            return baseline_survival_function.iloc[-1]
        else:
            return np.interp(t, baseline_survival_function.index, baseline_survival_function.values)

    # Step 2: Predict survival probabilities at each evaluation time
    if verbose:
        print("Step 2: Computing survival predictions...")
    survival_predictions = np.array([
        [baseline_survival(t) ** np.exp(risk_score) for t in evaluation_times]
        for risk_score in risk_scores
    ])

    # Step 3: Fit Kaplan-Meier estimator for censoring distribution
    if verbose:
        print("Step 3: Fitting Kaplan-Meier for censoring...")
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=(1 - events))  # 1 - event to model censoring

    if verbose:
        print("Step 4: Predicting censoring probabilities...")
    censoring_probabilities = kmf.predict(evaluation_times)

    # Step 5: Compute IPCW-weighted Brier score at each time point
    if verbose:
        print("Step 5: Looping over evaluation times...")
    brier_scores = []

    for idx, t in enumerate(evaluation_times):
        censoring_probability_t = censoring_probabilities.loc[t]

        if censoring_probability_t > 0:
            weights = (times >= t) / censoring_probability_t
        else:
            weights = np.zeros_like(times)

        observed = (times > t).astype(float)
        errors = (observed - survival_predictions[:, idx]) ** 2

        if np.sum(weights) > 0:
            brier_score_t = np.sum(weights * errors) / np.sum(weights)
        else:
            brier_score_t = np.nan

        brier_scores.append(brier_score_t)

    return np.array(brier_scores)



def evaluate_brier_score_over_time(
    df_survival,
    cph,
    times,
    brier_score_function,
    export_fig=False,
    path_export_fig="./results/brier_score_plot.png",
    verbose=True
):
    """
    Evaluates and plots the Brier Score over time for a Cox model using IPCW correction.
    Also computes mean Brier Scores over predefined time intervals.

    Parameters
    ----------
    df_survival : pd.DataFrame
        DataFrame with survival data including 'event', 'time', and 'risk_score' columns.
    cph : lifelines CoxPHFitter
        Trained Cox proportional hazards model.
    times : np.ndarray
        Time points (e.g., from the test set) used to determine evaluation range.
    brier_score_function : callable
        Function that computes Brier scores given (df_survival, cph, evaluation_times).
        Should return a 1D np.ndarray of Brier scores.
    export_fig : bool, default=False
        If True, saves the plot as an image.
    path_export_fig : str, default="./brier_score_plot.png"
        Path to save the figure if export_fig is True.
    verbose : bool, default=True
        If True, displays interval-based summaries of Brier scores.

    Returns
    -------
    evaluation_times : np.ndarray
        Sorted array of evaluation times used to compute Brier scores.
    brier_scores : np.ndarray
        Brier scores corresponding to each evaluation time.
    interval_results : list of tuples
        List of (start, end, mean_brier, std_brier) for each interval.
    """

    # 1. Define evaluation times (100 points between min and max, plus clinical landmarks)
    evaluation_times = np.linspace(times.min(), times.max(), 100)
    clinical_landmarks = np.array([365, 730, 1095, 1825, 3650])
    evaluation_times = np.sort(np.unique(np.concatenate((evaluation_times, clinical_landmarks))))

    # 2. Compute Brier scores using the provided function
    brier_scores = brier_score_function(df_survival, cph, evaluation_times)

    # 3. Define clinical time intervals (in days)
    intervals = [
        (0, 365), (365, 730), (730, 1095), (1095, 1461),     # Years 1–4
        (0, 730), (0, 1095), (0, 1825), (0, 3650),           # 0 to 2, 3, 5, 10 years
        (365, 1095)                                          # Year 2–3
    ]

    # 4. Compute mean and std of Brier scores per interval
    interval_results = []
    for start, end in intervals:
        indices = np.where((evaluation_times >= start) & (evaluation_times < end))[0]
        if len(indices) > 0:
            mean_brier = np.mean(brier_scores[indices])
            std_brier = np.std(brier_scores[indices])
        else:
            mean_brier, std_brier = np.nan, np.nan
        interval_results.append((start, end, mean_brier, std_brier))

    # 5. Verbose output of results
    if verbose:
        print("\n=== Brier Score Summary by Time Interval ===")
        for start, end, mean, std in interval_results:
            print(f"Interval {start}–{end} days: Mean = {mean:.4f} ± {std:.4f}")
        print("============================================\n")

    # 6. Plot Brier score over time
    plt.figure(figsize=(10, 6))
    plt.plot(evaluation_times, brier_scores, label='Model Brier Score', color='#50C878')
    plt.axhline(0.5, color='red', linestyle='--', label='Naïve Baseline (BS=0.5)')
    plt.title('Brier Score over Time', fontsize=16)
    plt.xlabel('Time (days)', fontsize=14)
    plt.ylabel('Brier Score', fontsize=14)
    plt.legend()
    plt.grid(True)

    if export_fig:
        plt.savefig(path_export_fig, dpi=300, bbox_inches='tight')

    plt.show()

    return evaluation_times, brier_scores, interval_results



def evaluate_brier_score_multiple_tests(
    df_survival_test_list,
    cph,
    evaluation_times,
    brier_score_function,
    export_fig=False,
    path_export_fig="./results/brier_score_multiple_tests.png",
    verbose=True
):
    """
    Evaluates and plots the Brier Score across multiple test sets with 95% confidence intervals.

    Parameters
    ----------
    df_survival_test_list : list of pd.DataFrame
        List of test datasets. Each DataFrame must contain the columns:
        ['time', 'event', 'ID', 'risk_score'].
    cph : lifelines.CoxPHFitter
        Trained Cox proportional hazards model used for risk estimation.
    evaluation_times : np.ndarray
        Array of time points at which to evaluate the Brier Score.
    brier_score_function : callable
        Function with signature (df_survival, cph, evaluation_times) → np.ndarray of Brier scores.
    export_fig : bool, default=False
        If True, saves the resulting plot to `path_export_fig`.
    path_export_fig : str, default="./brier_score_multiple_tests.png"
        File path to save the plot, if `export_fig=True`.
    verbose : bool, default=True
        If True, prints intermediate statistics.

    Returns
    -------
    brier_scores_array : np.ndarray
        Array of shape (n_tests, len(evaluation_times)) containing Brier Scores.
    bs_mean : np.ndarray
        Mean Brier Score across test sets at each evaluation time.
    bs_std : np.ndarray
        Standard deviation of Brier Scores across test sets.
    """

    # 1. Compute Brier Scores for each test dataset
    brier_scores_list = []
    for df_test in tqdm(df_survival_test_list, desc="Computing Brier Scores"):
        brier_scores = brier_score_function(df_test, cph, evaluation_times)
        brier_scores_list.append(brier_scores)

    # 2. Convert to NumPy array
    brier_scores_array = np.array(brier_scores_list)  # Shape: (n_tests, len(evaluation_times))

    # 3. Compute mean and 95% CI bounds
    bs_mean = np.nanmean(brier_scores_array, axis=0)
    bs_std = np.nanstd(brier_scores_array, axis=0)
    bs_upper = bs_mean + 1.96 * bs_std
    bs_lower = bs_mean - 1.96 * bs_std

    # 4. Verbose output
    if verbose:
        print("\n=== Brier Score Summary ===")
        print(f"Mean BS across tests: {np.nanmean(bs_mean):.4f}")
        print(f"BS range: [{np.nanmin(bs_mean):.4f} ;  {np.nanmax(bs_mean):.4f}]")
        print("===========================\n")

    # 5. Plot Brier Score curve with 95% CI
    plt.figure(figsize=(8, 5))
    plt.plot(evaluation_times, bs_mean, label='Mean Brier Score', color='#50C878')
    plt.fill_between(evaluation_times, bs_lower, bs_upper, color='#50C878', alpha=0.2, label='95% CI')
    plt.axhline(0.25, color='red', linestyle='--', label='Reference Threshold (BS = 0.25)')

    plt.title('Brier Score (BS) with 95% Confidence Interval', fontsize=15)
    plt.xlabel('Time (days)', fontsize=13)
    plt.ylabel('Brier Score', fontsize=13)
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.xlim(evaluation_times[0], 3650)  # Up to 10 years
    plt.ylim([0, 0.63])

    if export_fig:
        plt.savefig(path_export_fig, dpi=300, bbox_inches='tight')

    plt.show()

    return brier_scores_array, bs_mean, bs_std, bs_upper, bs_lower


def summarize_brier_scores_at_times(
    evaluation_times,
    bs_mean,
    bs_lower,
    bs_upper,
    times_of_interest=[365, 701, 1095, 1825, 3650],
    verbose=True
):
    """
    Summarizes the Brier Score at specific evaluation times.

    Parameters
    ----------
    evaluation_times : np.ndarray
        Array of time points where Brier Scores were computed.
    bs_mean : np.ndarray
        Mean Brier Score at each evaluation time.
    bs_lower : np.ndarray
        Lower bound of the 95% confidence interval for Brier Score.
    bs_upper : np.ndarray
        Upper bound of the 95% confidence interval for Brier Score.
    times_of_interest : list of int, default=[365, 701, 1095, 1825, 3650]
        Specific time points (in days) at which to summarize the Brier Score.
    verbose : bool, default=True
        If True, prints the results.

    Returns
    -------
    bs_results : dict
        Dictionary where keys are time points and values are dicts with
        'Mean BS' and 'Std BS' (standard deviation estimated from CI width).
    """

    bs_results = {}

    for t in times_of_interest:
        closest_idx = np.abs(evaluation_times - t).argmin()
        mean_bs = bs_mean[closest_idx]
        std_bs = (bs_upper[closest_idx] - bs_lower[closest_idx]) / (2 * 1.96)  # Approximate std from CI
        bs_results[t] = {
            "Mean BS": mean_bs,
            "Std BS": std_bs
        }

    if verbose:
        for t, values in bs_results.items():
            if t == 365:
                label = "1 year"
            elif t == 1095:
                label = "3 years"
            elif t == 1825:
                label = "5 years"
            elif t == 3650:
                label = "10 years"
            else:
                label = f"{t} days"
            print(f"BS at {label}: Mean = {values['Mean BS']:.4f}, Std = {values['Std BS']:.4f}")

    return bs_results


def compute_and_plot_ibs_with_ci(
    evaluation_times,
    bs_mean,
    bs_lower,
    bs_upper,
    times_of_interest=[365, 701, 1095, 1825, 3650],
    plot_baseline=0.25,
    verbose=True
):
    """
    Computes and plots the Integrated Brier Score (IBS) with 95% confidence intervals 
    over time, and summarizes IBS at specific time points.

    Parameters
    ----------
    evaluation_times : np.ndarray
        Array of time points where the Brier Score was evaluated.
    bs_mean : np.ndarray
        Mean Brier Score values over time.
    bs_lower : np.ndarray
        Lower bound of the 95% confidence interval for the Brier Score.
    bs_upper : np.ndarray
        Upper bound of the 95% confidence interval for the Brier Score.
    times_of_interest : list of int, default=[365, 701, 1095, 1825, 3650]
        Time points (in days) at which to summarize IBS values.
    plot_baseline : float, default=0.25
        Reference threshold line to be plotted on the IBS graph.
    verbose : bool, default=True
        If True, prints the IBS values at the selected time points.

    Returns
    -------
    ibs_results : dict
        Dictionary mapping time points to dictionaries containing the mean and
        approximate standard deviation of the IBS at that time.
    """

    # Compute cumulative IBS (integral approximation via trapezoidal rule)
    delta_t = np.gradient(evaluation_times)
    ibs_mean = np.cumsum(bs_mean * delta_t) / (evaluation_times - evaluation_times[0])
    ibs_lower = np.cumsum(bs_lower * delta_t) / (evaluation_times - evaluation_times[0])
    ibs_upper = np.cumsum(bs_upper * delta_t) / (evaluation_times - evaluation_times[0])

    # Plot IBS with 95% confidence interval
    plt.figure(figsize=(8, 5))
    plt.plot(evaluation_times, ibs_mean, label='Mean IBS', color='#50C878')
    plt.fill_between(evaluation_times, ibs_lower, ibs_upper, alpha=0.2, color='#50C878', label='95% CI')
    plt.axhline(plot_baseline, color='red', linestyle='--', label=f'Reference Threshold (IBS = {plot_baseline:.2f})')

    plt.title('Integrated Brier Score (IBS) with 95% Confidence Interval')
    plt.xlabel('Time (days)')
    plt.ylabel('IBS')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlim(evaluation_times[0], 7307 / 2)  # up to 10 years
    plt.ylim([0, 0.45])
    plt.show()

    # Extract IBS at specific time points
    ibs_results = {}
    for t in times_of_interest:
        closest_idx = np.abs(evaluation_times - t).argmin()
        std_ibs = (ibs_upper[closest_idx] - ibs_lower[closest_idx]) / (2 * 1.96)
        ibs_results[t] = {
            "Mean IBS": ibs_mean[closest_idx],
            "Std IBS": std_ibs
        }

    if verbose:
        for t, values in ibs_results.items():
            if t == 365:
                label = "1 year"
            elif t == 1095:
                label = "3 years"
            elif t == 1825:
                label = "5 years"
            elif t == 3650:
                label = "10 years"
            else:
                label = f"{t} days"
            print(f"IBS at {label}: Mean = {values['Mean IBS']:.4f}, Std = {values['Std IBS']:.4f}")

    return ibs_results


def plot_smoothed_cindex_by_report_count(
    df,
    window_size=5,
    reference_cindex=0.75,
    vertical_line_x=28,
    vertical_line_y=0.7,
    ci_level=0.95,
    n_folds=10,
    export_path=None
):
    """
    Plot a smoothed curve of the C-index on the validation set as a function of the maximum number of reports per patient.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the following columns:
        - "max_reports"
        - "Mean C-index"
        - "Std C-index"
    window_size : int, optional
        Window size for the moving average smoothing (default: 5).
    reference_cindex : float, optional
        Horizontal reference line for C-index (default: 0.75).
    vertical_line_x : int, optional
        X-position for a vertical reference line (e.g., report count) (default: 28).
    vertical_line_y : float, optional
        Y-position for the vertical reference annotation (default: 0.7).
    ci_level : float, optional
        Confidence interval level (default: 0.95).
    n_folds : int, optional
        Number of test sets used for estimating standard deviation (default: 10).
    export_path : str or None
        If provided, path to save the figure (e.g., './fig.png').

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract necessary columns
    max_reports = df["max_reports"]
    mean_cindex = df["Mean C-index"]
    std_cindex = df["Std C-index"]

    # Compute confidence intervals
    z = 1.96 if ci_level == 0.95 else 1.0  # Can be extended
    ci_upper = mean_cindex + z / np.sqrt(n_folds) * std_cindex
    ci_lower = mean_cindex - z / np.sqrt(n_folds) * std_cindex

    # Define moving average
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Smooth the curves
    max_reports_smooth = max_reports[window_size - 1:]
    mean_cindex_smooth = moving_average(mean_cindex, window_size)
    ci_upper_smooth = moving_average(ci_upper, window_size)
    ci_lower_smooth = moving_average(ci_lower, window_size)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.plot(max_reports_smooth, mean_cindex_smooth, linestyle="-", color="#07AAE4", linewidth=2, label="Mean C-index")
    plt.fill_between(max_reports_smooth, ci_lower_smooth, ci_upper_smooth, color="#07AAE4", alpha=0.2, label="95% CI")

    # Reference lines
    plt.axhline(y=reference_cindex, color="#B21D61", linestyle="--", linewidth=2, label=f"Reference C-index ({reference_cindex})")
    plt.plot([vertical_line_x, vertical_line_x], [plt.ylim()[0], vertical_line_y], color="#B21D61", linestyle="--", linewidth=1.2, label=f"{vertical_line_x} reports for C-index of {vertical_line_y}")

    # Plot formatting
    plt.xlabel("Maximum Number of Known Reports", fontsize=15)
    plt.ylabel("C-index on Validation Set", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim([0.62, 0.77])

    # Save if export path is given
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {export_path}")

    plt.show()


def plot_smoothed_cindex_by_variable(
    df,
    var_abs="max_reports",
    var_ord="Mean C-index",
    std_ord="Std C-index",
    window_size=5,
    reference_cindex=0.75,
    vertical_line_x=28,
    vertical_line_y=0.7,
    ci_level=0.95,
    n_folds=10,
    export_path=None
):
    """
    Plot a smoothed curve of a performance metric (e.g., C-index) as a function of a given variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the following columns:
        - var_abs (x-axis)
        - var_ord (mean metric)
        - std_ord (std deviation of the metric)
    var_abs : str, optional
        Column name to use as the x-axis variable (default: "max_reports").
    var_ord : str, optional
        Column name to use as the y-axis variable (default: "Mean C-index").
    std_ord : str, optional
        Column name for the standard deviation of the y-variable (default: "Std C-index").
    window_size : int, optional
        Window size for the moving average smoothing (default: 5).
    reference_cindex : float, optional
        Horizontal reference line (default: 0.75).
    vertical_line_x : int or float, optional
        X-position for a vertical reference line (default: 28).
    vertical_line_y : float, optional
        Y-position for the vertical reference annotation (default: 0.7).
    ci_level : float, optional
        Confidence interval level (default: 0.95).
    n_folds : int, optional
        Number of test sets used for estimating standard deviation (default: 10).
    export_path : str or None
        If provided, path to save the figure (e.g., './fig.png').

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Extract columns
    x = df[var_abs]
    y = df[var_ord]
    y_std = df[std_ord]

    # Compute confidence intervals
    z = 1.96 if ci_level == 0.95 else 1.0
    ci_upper = y + z / np.sqrt(n_folds) * y_std
    ci_lower = y - z / np.sqrt(n_folds) * y_std

    # Define moving average
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Smoothed values
    x_smooth = x[window_size - 1:]
    y_smooth = moving_average(y, window_size)
    ci_upper_smooth = moving_average(ci_upper, window_size)
    ci_lower_smooth = moving_average(ci_lower, window_size)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.plot(x_smooth, y_smooth, linestyle="-", color="#07AAE4", linewidth=2, label=var_ord)
    plt.fill_between(x_smooth, ci_lower_smooth, ci_upper_smooth, color="#07AAE4", alpha=0.2, label=f"{int(ci_level * 100)}% CI")

    # Reference lines
    plt.axhline(y=reference_cindex, color="#B21D61", linestyle="--", linewidth=2, label=f"Reference ({reference_cindex})")
    plt.plot([vertical_line_x, vertical_line_x], [plt.ylim()[0], vertical_line_y], color="#B21D61", linestyle="--", linewidth=1.2, label=f"{vertical_line_x} for value {vertical_line_y}")

    # Plot formatting
    plt.xlabel(var_abs.replace("_", " ").title(), fontsize=15)
    plt.ylabel(var_ord.replace("_", " ").title(), fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Export if path provided
    if export_path:
        plt.savefig(export_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {export_path}")

    plt.show()



def compute_integrated_auc(times, auc_values, t_min=None, t_max=None, verbose=True):
    """
    Compute the integrated time-dependent AUC (iAUC).

    Parameters
    ----------
    times : array-like
        Time points at which AUC(t) was evaluated.
    auc_values : array-like
        Corresponding AUC(t) values.
    t_min : float, optional
        Lower integration bound (default: min(times)).
    t_max : float, optional
        Upper integration bound (default: max(times)).
    verbose : bool, default=True
        Print the iAUC result.

    Returns
    -------
    iauc : float
        Integrated AUC over [t_min, t_max].
    """
    times = np.asarray(times)
    auc_values = np.asarray(auc_values)

    # Clean NaN values
    mask = ~np.isnan(times) & ~np.isnan(auc_values)
    times, auc_values = times[mask], auc_values[mask]
    if len(times) < 2:
        raise ValueError("Not enough valid AUC values for integration.")

    # Define integration limits
    t_min = t_min if t_min is not None else times.min()
    t_max = t_max if t_max is not None else times.max()

    # Restrict domain
    mask = (times >= t_min) & (times <= t_max)
    times, auc_values = times[mask], auc_values[mask]

    # Numerical integration (trapezoidal rule)
    area = np.trapz(auc_values, times)
    iauc = area / (t_max - t_min)

    if verbose:
        print(f"Integrated AUC (iAUC) over [{t_min:.1f}, {t_max:.1f}] = {iauc:.4f}")

    return iauc