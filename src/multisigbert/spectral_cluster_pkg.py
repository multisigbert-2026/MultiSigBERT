import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import pynndescent
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
from sklearn.cluster import SpectralClustering

from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import eigsh



def load_signature_dataset(csv_path, verbose=False):
    """
    Load and standardize a survival dataset containing signature-based covariates.

    The input CSV file is expected to follow the structure below:

    Columns
    -------
    - ID: patient identifier.
    - sig_1, ..., sig_p: signature coefficients used as covariates.
    - event: binary indicator (1 = death, 0 = censored).
    - time: survival time in days from hospital admission to last observation
            (death if event == 1, last known report otherwise).

    This function extracts signature features, applies z-score standardization,
    and returns the normalized design matrix.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    verbose : bool, default=False
        If True, prints dataset statistics.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Standardized signature feature matrix.
    scaler : StandardScaler
        Fitted scaler used for normalization.
    signature_columns : list of str
        Names of extracted signature columns.
    """

    # Load dataset
    dataframe = pd.read_csv(csv_path)

    # Extract signature columns
    signature_columns = [col for col in dataframe.columns if col.startswith("sig_")]

    feature_matrix = dataframe[signature_columns].values.astype(float)

    # Standardization (critical for spectral methods)
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)

    n_samples, n_features = X.shape

    if verbose:
        print(f"Number of patients: {n_samples}")
        print(f"Number of signature features: {n_features}")

    return X, scaler, signature_columns


# =====================================================
# graph kNN -> MDS -> Spectral clustering
# =====================================================

def build_knn_graph(indices, distances, n_nodes):
    """
    Construct a symmetric weighted k-Nearest Neighbor (kNN) graph.

    Given the kNN structure (neighbor indices and associated distances),
    this function builds a sparse weighted adjacency matrix in CSR format.
    The graph is symmetrized using edge union (i.e., an edge (i, j) is kept
    if it exists in either direction).

    Parameters
    ----------
    indices : np.ndarray of shape (n_samples, k)
        Matrix containing the indices of the k nearest neighbors for each node.
    distances : np.ndarray of shape (n_samples, k)
        Matrix containing the corresponding pairwise distances.
    n_nodes : int
        Total number of nodes in the graph.

    Returns
    -------
    G : scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Symmetric weighted adjacency matrix representing the kNN graph.
    """

    n_samples, n_neighbors = indices.shape

    row_indices = np.repeat(np.arange(n_samples), n_neighbors)
    col_indices = indices.ravel()
    edge_weights = distances.ravel().astype(float)

    # Remove self-loops
    mask = row_indices != col_indices
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    edge_weights = edge_weights[mask]

    adjacency_matrix = csr_matrix(
        (edge_weights, (row_indices, col_indices)),
        shape=(n_nodes, n_nodes)
    )

    # Symmetrize via edge union
    adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.T)

    return adjacency_matrix


def largest_connected_component(G):
    """
    Extract the largest connected component of an undirected graph.

    Parameters
    ----------
    G : scipy.sparse.csr_matrix of shape (n_nodes, n_nodes)
        Sparse adjacency matrix of the graph.

    Returns
    -------
    kept_indices : np.ndarray
        Indices of nodes belonging to the largest connected component.
    G_lcc : scipy.sparse.csr_matrix
        Adjacency matrix restricted to the largest connected component.
    n_components : int
        Total number of connected components in the original graph.
    """

    n_components, component_labels = connected_components(
        G, directed=False, return_labels=True
    )

    if n_components == 1:
        kept_indices = np.arange(G.shape[0])
    else:
        largest_label = np.bincount(component_labels).argmax()
        kept_indices = np.where(component_labels == largest_label)[0]

    G_lcc = G[kept_indices][:, kept_indices]

    return kept_indices, G_lcc, n_components


# =====================================================
# GLOBAL SPECTRAL LEARNING PIPELINE
# =====================================================

def _print_cluster_summary(labels_cc, df, verbose=True):
    """
    Display cluster distribution and preview relevant survival columns.
    """
    if not verbose:
        return

    print("\nCluster distribution:")
    print(pd.Series(labels_cc).value_counts().sort_index())

    print("\nPreview of key columns:")
    display(df[["ID", "cluster_spectral", "event", "time"]].head())


def _plot_mds_projections(Y, labels, verbose=True):
    """
    Generate pairwise 2D projections of the MDS embedding,
    colored by spectral cluster assignments.
    """
    if not verbose:
        return

    n_components = Y.shape[1]
    valid_mask = labels >= 0

    for i in range(n_components):
        for j in range(i + 1, n_components):
            plt.figure(figsize=(8, 6))
            plt.scatter(
                Y[valid_mask, i],
                Y[valid_mask, j],
                c=labels[valid_mask],
                s=18,
                alpha=0.85
            )
            plt.title("MDS embedding + Spectral clustering")
            plt.xlabel(f"MDS-{i+1}")
            plt.ylabel(f"MDS-{j+1}")
            plt.grid(alpha=0.2)
            plt.show()


def global_spectral_learning(
    X,
    df,
    n_neighbors=15,
    n_clusters=3,
    n_mds_components=5,
    random_state=177,
    verbose=True
):
    """
    Perform global spectral learning on standardized features.

    The pipeline includes:
    1) Approximate kNN graph construction (PyNNDescent),
    2) Extraction of the largest connected component,
    3) Geodesic distance computation,
    4) MDS embedding based on geodesic distances,
    5) Spectral clustering using an RBF affinity matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Standardized feature matrix.
    df : pd.DataFrame
        Original dataframe (must contain ID, event, time columns).
    n_neighbors : int, default=15
        Number of neighbors for kNN graph construction.
    n_clusters : int, default=3
        Number of clusters for spectral clustering.
    n_mds_components : int, default=5
        Dimensionality of the MDS embedding.
    random_state : int, default=177
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print diagnostics and display plots.

    Returns
    -------
    labels_spectral : np.ndarray
        Cluster labels for all samples (-1 outside main component).
    Y : np.ndarray
        Full MDS embedding (NaN outside main component).
    A : scipy.sparse.csr_matrix
        RBF affinity matrix used for spectral clustering.
    sigma : float
        RBF bandwidth parameter (median heuristic).
    df_clusters : pd.DataFrame
        DataFrame containing ID, event, time, and cluster_spectral.
    """

    n_patients = X.shape[0]

    if verbose:
        print("Building kNN graph...")

    index = pynndescent.NNDescent(
        X,
        n_neighbors=n_neighbors,
        random_state=random_state
    )

    indices, distances = index.query(X)

    # kNN graph
    G = build_knn_graph(indices, distances, n_patients)
    keep, G_cc, n_components = largest_connected_component(G)

    if verbose:
        print(f"Number of connected components: {n_components}")
        print(f"Largest component size: {len(keep)} / {n_patients}")

    # Geodesic distances
    D = shortest_path(G_cc, directed=False, unweighted=False)
    D = np.asarray(D, dtype=float)

    # MDS embedding
    mds = MDS(
        n_components=n_mds_components,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        max_iter=400,
        normalized_stress="auto"
    )

    Y_cc = mds.fit_transform(D)

    Y = np.full((n_patients, n_mds_components), np.nan, dtype=float)
    Y[keep] = Y_cc

    # RBF affinity matrix
    sigma = np.median(G_cc.data)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    A = G_cc.copy().tocsr()
    A.data = np.exp(-(A.data ** 2) / (2.0 * sigma ** 2))
    A = A.maximum(A.T)

    A.setdiag(1.0)
    A.eliminate_zeros()

    # Spectral clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
        n_init=20
    )

    labels_cc = spectral.fit_predict(A)

    labels_spectral = np.full(n_patients, -1, dtype=int)
    labels_spectral[keep] = labels_cc

    df["cluster_spectral"] = labels_spectral

    # Reduced dataframe with key survival columns
    df_clusters = df[["ID", "event", "time", "cluster_spectral"]].copy()

    # Diagnostics
    _print_cluster_summary(labels_cc, df, verbose)
    _plot_mds_projections(Y, labels_spectral, verbose)

    return labels_spectral, Y, A, sigma, df_clusters


# =====================================================
# MDS + SPECTRAL + CENTRALITY-CONFORMAL RELEVANCE INDEX
# =====================================================


def stationary_centrality(W):
    """
    Compute the stationary degree-based centrality of a weighted graph.

    This centrality corresponds to the normalized node degrees and can be
    interpreted as the stationary distribution of a random walk on the graph
    when transition probabilities are proportional to edge weights.

    Parameters
    ----------
    W : scipy.sparse matrix or np.ndarray of shape (n_nodes, n_nodes)
        Symmetric weighted adjacency matrix.

    Returns
    -------
    np.ndarray of shape (n_nodes,)
        Normalized stationary centrality vector.
    """
    deg = np.asarray(W.sum(axis=1)).ravel()
    s = deg.sum()
    if s <= 0:
        return np.full(W.shape[0], 1.0 / max(W.shape[0], 1))
    return deg / s


def pagerank_centrality(W, alpha=0.85, max_iter=300, tol=1e-10):
    """
    Compute PageRank centrality on a weighted graph.

    The method estimates the stationary distribution of a random walk with
    teleportation, controlled by damping factor alpha.

    Parameters
    ----------
    W : scipy.sparse matrix or np.ndarray
        Weighted adjacency matrix.
    alpha : float, default=0.85
        Damping factor.
    max_iter : int, default=300
        Maximum number of power iterations.
    tol : float, default=1e-10
        Convergence tolerance in L1 norm.

    Returns
    -------
    np.ndarray
        Normalized PageRank centrality scores.
    """
    n = W.shape[0]
    if n == 1:
        return np.array([1.0])

    deg = np.asarray(W.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(deg)
    nz = deg > 0
    inv_deg[nz] = 1.0 / deg[nz]

    P = diags(inv_deg) @ W

    v = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - alpha) / n)

    for _ in range(max_iter):
        v_next = alpha * (P.T @ v) + teleport
        if np.linalg.norm(v_next - v, 1) < tol:
            v = v_next
            break
        v = v_next

    v = np.maximum(v, 0)
    s = v.sum()
    return v / s if s > 0 else np.full(n, 1.0 / n)


def eigenvector_centrality(W):
    """
    Compute eigenvector centrality for a weighted graph.

    The centrality vector corresponds to the dominant eigenvector of the
    adjacency matrix. A robust fallback to stationary centrality is used
    if eigendecomposition fails.

    Parameters
    ----------
    W : scipy.sparse matrix or np.ndarray
        Weighted adjacency matrix.

    Returns
    -------
    np.ndarray
        Normalized eigenvector centrality scores.
    """
    n = W.shape[0]
    if n == 1:
        return np.array([1.0])

    try:
        vals, vecs = eigsh(W, k=1, which="LA")
        v = np.abs(vecs[:, 0])
    except Exception:
        v = stationary_centrality(W)

    s = v.sum()
    return v / s if s > 0 else np.full(n, 1.0 / n)


def compute_centrality(W, method="stationary"):
    """
    Unified interface for graph centrality computation.

    Parameters
    ----------
    W : scipy.sparse matrix or np.ndarray
        Weighted adjacency matrix.
    method : str, default="stationary"
        Centrality method in {"stationary", "pagerank", "eigenvector"}.

    Returns
    -------
    np.ndarray
        Normalized centrality scores.

    Raises
    ------
    ValueError
        If the method is not recognized.
    """
    method = method.lower()
    if method == "stationary":
        return stationary_centrality(W)
    if method == "pagerank":
        return pagerank_centrality(W)
    if method == "eigenvector":
        return eigenvector_centrality(W)
    raise ValueError(
        "CENTRALITY_METHOD must be 'stationary', 'pagerank' or 'eigenvector'."
    )


# =====================================================
# CENTRALITY-CONFORMAL SPLIT AND INJECTION PIPELINE
# =====================================================


def split_train_calibration(
    n_samples,
    calibration_fraction,
    random_state=177,
    stratify_labels=None
):
    """
    Split indices into training and calibration subsets.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    calibration_fraction : float
        Fraction of samples assigned to calibration.
    random_state : int, default=177
        Random seed for reproducibility.
    stratify_labels : np.ndarray or None
        Optional labels for stratified splitting.

    Returns
    -------
    train_idx : np.ndarray
        Training indices.
    cal_idx : np.ndarray
        Calibration indices.
    """
    indices_all = np.arange(n_samples)

    train_idx, cal_idx = train_test_split(
        indices_all,
        test_size=calibration_fraction,
        random_state=random_state,
        stratify=stratify_labels
    )

    return train_idx, cal_idx


def extract_train_subgraph(W_full, train_idx):
    """
    Extract the induced subgraph restricted to training nodes.

    Parameters
    ----------
    W_full : csr_matrix
        Full adjacency matrix.
    train_idx : np.ndarray
        Indices selected for training.

    Returns
    -------
    W_train : csr_matrix
        Induced training subgraph.
    """
    return W_full[train_idx][:, train_idx].tocsr()


def compute_train_centrality(W_train, method):
    """
    Compute centrality scores on the training subgraph.

    Returns
    -------
    cent_train : np.ndarray
        Centrality scores for training nodes.
    deg_sum_train : float
        Sum of node degrees (useful for stationary shortcut).
    """
    cent_train = compute_centrality(W_train, method)
    cent_train = np.asarray(cent_train, dtype=float).ravel()

    deg_sum_train = float(np.asarray(W_train.sum(axis=1)).ravel().sum())

    return cent_train, deg_sum_train


def centrality_of_injected_node(
    cal_global_idx,
    W_full,
    W_train,
    train_idx,
    X,
    sigma_aff,
    method,
    deg_sum_train,
    k_inject=10,
    use_full_row_neighbors=True
):
    """
    Estimate centrality of a calibration point injected into the training graph.

    The calibration node is connected to its nearest neighbors in the
    training set, either using existing graph edges or recomputed kNN
    in feature space.

    Returns
    -------
    float
        Centrality score of the injected node.
    """

    n_train = W_train.shape[0]
    global2local = {g: i for i, g in enumerate(train_idx)}

    g_neighbors = np.array([], dtype=int)
    w_neighbors = np.array([], dtype=float)

    # --- Use existing edges from full graph
    if use_full_row_neighbors:
        row = W_full.getrow(cal_global_idx)
        neigh_global = row.indices
        neigh_w = row.data

        mask = np.isin(neigh_global, train_idx)
        g_neighbors = neigh_global[mask]
        w_neighbors = neigh_w[mask]

        if w_neighbors.size > k_inject:
            keep = np.argpartition(w_neighbors, -k_inject)[-k_inject:]
            g_neighbors = g_neighbors[keep]
            w_neighbors = w_neighbors[keep]

    # --- Fallback: recompute neighbors in feature space
    if (not use_full_row_neighbors) or (w_neighbors.size == 0):
        Xt = X[train_idx]
        xc = X[cal_global_idx]
        d = np.linalg.norm(Xt - xc, axis=1)

        k = min(k_inject, len(d))
        nn_loc = np.argpartition(d, kth=k - 1)[:k]
        g_neighbors = train_idx[nn_loc]
        w_neighbors = np.exp(-(d[nn_loc] ** 2) / (2.0 * (sigma_aff ** 2 + 1e-12)))

    if w_neighbors.size == 0:
        return 0.0

    local_idx = np.array([global2local[g] for g in g_neighbors], dtype=int)
    local_w = np.array(w_neighbors, dtype=float)

    # --- Stationary shortcut
    if method.lower() == "stationary":
        d_new = float(np.sum(local_w))
        denom = deg_sum_train + 2.0 * d_new
        if denom <= 0:
            return 1.0 / (n_train + 1.0)
        return d_new / denom

    # --- Full recomputation on augmented graph
    col_new = csr_matrix(
        (local_w, (local_idx, np.zeros_like(local_idx))),
        shape=(n_train, 1)
    )
    row_new = csr_matrix(
        (local_w, (np.zeros_like(local_idx), local_idx)),
        shape=(1, n_train)
    )

    top = hstack([W_train, col_new], format="csr")
    bottom = hstack([row_new, csr_matrix((1, 1))], format="csr")
    W_aug = vstack([top, bottom], format="csr")

    cent_aug = compute_centrality(W_aug, method)
    cent_aug = np.asarray(cent_aug, dtype=float).ravel()

    return float(cent_aug[-1])


def compute_calibration_centralities(
    cal_idx,
    W_full,
    W_train,
    train_idx,
    X,
    sigma_aff,
    method,
    deg_sum_train,
    k_inject=50,
    use_full_row_neighbors=True
):
    """
    Compute injected centralities for all calibration points.
    """
    cent_cal = np.zeros(len(cal_idx), dtype=float)

    for j, gcal in enumerate(cal_idx):
        cent_cal[j] = centrality_of_injected_node(
            gcal,
            W_full,
            W_train,
            train_idx,
            X,
            sigma_aff,
            method,
            deg_sum_train,
            k_inject=k_inject,
            use_full_row_neighbors=use_full_row_neighbors
        )

    return cent_cal


def plot_train_vs_calibration_histograms(
    cent_train,
    cent_cal,
    method,
    k_inject
):
    """
    Plot overlaid histograms of training and injected calibration centralities.
    """
    plt.figure(figsize=(7, 4))

    plt.hist(cent_train, bins=40, alpha=0.5, label="TRAIN", density=True)
    plt.hist(
        cent_cal,
        bins=40,
        alpha=0.5,
        label=f"CAL injected (k={k_inject})",
        density=True
    )

    plt.title(f"Centralities: TRAIN vs CAL injected (method: {method})")
    plt.xlabel("centrality")
    plt.ylabel("density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()