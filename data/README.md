# Processed Data for SigBERT

- This directory contains the processed tabular datasets used as inputs for survival prediction models. Each file typically corresponds to a set of patient reports that have been cleaned, timestamped, and encoded with sentence-level embeddings.
- All identifiers must be fully anonymized and compliant with local data protection regulations.
- Sentence embeddings are typically obtained from clinical narratives and are designed to preserve semantic information in a compact numerical format.
 
## Required Columns

Each CSV file in this folder must contain the following columns:

| Column Name     | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `ID`             | Unique patient identifier. Must be anonymized.                             |
| `date_creation`  | Timestamp of the original medical report (format: `YYYY-MM-DD`).           |
| `DEATH`          | Binary target variable indicating death status (`0` = censored, `1` = death). |
| `date_death`     | Date of death when available (format: `YYYY-MM-DD`), `NaN` otherwise.      |
| `date_start`     | Earliest report date per patient. Used as the start of the follow-up period. |
| `date_end`       | Latest report date per patient. Used as the end of the follow-up period.     |
| `embeddings`     | Sentence embedding vector representing the medical report. Formally, a dense vector in $\mathbb{R}^p$. Stored as a stringified list (e.g., `"[0.01, 0.43, ...]"`). |

## Format Notes

- All date fields must be in ISO format (`YYYY-MM-DD`), and parseable as `datetime` objects in Python.
- The `embeddings` column stores sentence-level representations computed using a language model (e.g., OncoBERT) and optionally reduced in dimension (e.g., via PCA).
- `DEATH` must be strictly binary: `{0, 1}`.

## Usage

These files are typically produced by the notebook [`compute_sent_embd.ipynb`](../notebooks/compute_sent_embd.ipynb) 
and are consumed by downstream survival analysis code which expects these exact column names and formats.

## Example Rows

| ID     | date_creation | DEATH | date_death | date_start | date_end  | embeddings                           |
|--------|---------------|-------|------------|------------|-----------|--------------------------------------|
| 12345  | 2020-06-18    | 1     | 2021-09-30 | 2018-03-14 | 2021-06-20 | "[0.012, -0.345, ..., 0.098]"        |
| 12345  | 2020-10-22    | 1     | 2021-09-30 | 2018-03-14 | 2021-06-20 | "[0.812, -0.450, ..., 0.930]"        |
| 12345  | 2021-01-11    | 1     | 2021-09-30 | 2018-03-14 | 2021-06-20 | "[-0.188, -0.990, ..., 0.153]"        |




### 1. Landmark Cohort Definition
At each landmark $L$, we restrict the analysis to the risk set:
$$
\mathsf{cohort}_L = \{\, i : T_i \ge L \,\},
$$
that is, patients who are still under observation at the landmark. 
Individuals with events or censoring before $L$ are excluded to avoid conditioning on future information.

### 2. Feature Construction
For each remaining patient $i$, we extract temporal features from the time window 
$[L-w, L]$, where $w > 0$ is a backward window size:
$$
\mathbb{S}_i(L) = \operatorname{Sign}_{\text{order}}\big(v_t^i,\, t \in [L-w, L]\big).
$$
We then build the covariate vector
$$
\mathbb{X}_i(L) = 
\begin{bmatrix}
\mathbb{S}_i(L) \\
\gamma_i(L)
\end{bmatrix},
\quad
\text{where } 
\gamma_i(L) = \mathbb{1}\{L - t_{\text{diag},i} < w\}
$$
is the *history-short indicator* identifying patients with less than $w$ months of available history before $L$.


Landmark-Based Survival MultiModelling
---


| Patient | date $t$ | Sentence embedding $v \in \mathbb{R}^q$ | $D \in \{0,1\}$ | $T \ge 0$ |
|:--------|:-----------:|:------------------------------------------:|:-----------------:|:------------:|
| $i$ |$\begin{aligned} &t_1 \\ &\vdots \\ &t_N \end{aligned}$| $\begin{aligned} &v^i_{t_1} \\ &\vdots \\ &v^i_{t_N} \end{aligned}$ | $\delta_i$ | $T_i$ |


Let each patient $i$ be described by a sequence of sentence embeddings $v_t^i \in \mathbb{R}^p$ observed at times $t$, an event indicator $\delta_i \in \{0,1\}$, and a survival time $T_i \ge 0$.
We aim to estimate short-term survival probabilities at a fixed **landmark time** $L$ using information available up to $L$.

### Formal Table - Landmark-Specific Cohort at $L$ Months

| Patient $i$ | Report time $t$ | Embedding $v_t^i \in \mathbb{R}^p$ | Event indicator $\delta_i(L)$ | Total survival time $T_i \ge 0$ | Residual time $R_i = T_i - L$ | Days since start $(t - t_{\text{start},i})$ | Short-history flag $\gamma_i(L)$ | **Landmark $L$** |
|:--------------:|:-----------------:|:-------------------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------------------:|:----------------------------------:|:------------------:|
| $i$ | $t_1 \in [L-w,L]$ | $v_{t_1}^i$ | $\delta_i(L)$ | $T_i$ | $T_i - L$ | $(t_1 - t_{\text{start},i})$ | $\gamma_i(L)$ | $L$ |
|  | $t_2 \in [L-w,L]$ | $v_{t_2}^i$ | $\delta_i(L)$ | $T_i$ | $T_i - L$ | $(t_2 - t_{\text{start},i})$ | $\gamma_i(L)$ | $L$ |
|  | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $L$ |
|  | $t_N \in [L-w,L]$ | $v_{t_N}^i$ | $\delta_i(L)$ | $T_i$ | $T_i - L$ | $(t_N - t_{\text{start},i})$ | $\gamma_i(L)$ | $L$ |

---


### 3. Time Re-indexing and Model Fitting
To ensure temporal coherence, we re-index the survival time at the landmark:
$$
R_i = T_i - L \quad (\text{residual time after } L),
$$
and retain only patients with $R_i > 0$.  
The event indicator is redefined as
$$
\delta_i(L) = 
\begin{cases}
1, & \text{if an event occurs after } L,\\
0, & \text{otherwise.}
\end{cases}
$$
A Cox proportional hazards model is then fitted using these left-truncated data:
$$
h(u \mid \mathbb{X}_i(L)) = h_0(u)\,
\exp\!\big(\beta^\top \mathbb{X}_i(L)\big),
\qquad u \ge 0.
$$
Here $u = 0$ corresponds to the landmark time $L$.

| Patient | Covariates from Signature Transform $\in \mathbb{R}^{n+1}$ | $D \in \{0,1\}$ | $R \ge 0$ |
|:--------|:----------------------------------------------------------:|:-----------------:|:------------:|
| $i$ | $\mathbb{X}_i(L)$ | $\delta_i(L)$ | $R_i = T_i - L$ |

### 4. Prediction at a Fixed Horizon
From the fitted model, the conditional survival function at horizon $h$ months after the landmark is:
$$
\mathbb{P}(T_i > L + h \mid \mathbb{X}_i(L)) 
= \frac{S_0(L+h)^{\exp(\eta_i(L))}}{S_0(L)^{\exp(\eta_i(L))}}
= S_0^{*}(h)^{\exp(\eta_i(L))},
$$
where $S_0^{*}(h) = \exp\!\big(-\int_0^h h_0(u)\,du\big)$
is the baseline survival re-indexed at $L$,
and $\eta_i(L) = \beta^\top \mathbb{X}_i(L)$ is the patient’s risk score.

### 5. Evaluation
Predictions are evaluated dynamically at each landmark $L$ and horizon $h$ using:
- time-dependent AUC $\text{AUC}(L; h)$;
- Brier score or Integrated Brier Score (IBS) computed with inverse-probability-of-censoring weighting (IPCW);
- calibration and risk-stratification analyses.

#### Interpretation.
At each landmark $L$, the model estimates
$$
\mathbb{P}(T_i > L + h \mid \mathbb{X}_i(L))
$$ the probability that patient $i$ survives an additional $h$ months beyond $L$,
given all information available up to $L$.  
This formulation prevents temporal leakage and allows fair comparison of risk scores across patients at the same observation time.