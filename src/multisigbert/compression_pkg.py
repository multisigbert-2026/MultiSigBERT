import numpy as np
import torch

# PCA
def pca_compression(
    df_data,
    bar_p,
    var_embd='embeddings',
    verbose=False
):
    """
    Computes a linear compression of high-dimensional embeddings using PCA.

    Given a DataFrame containing p-dimensional embedding vectors, this function 
    returns their projection onto a lower-dimensional subspace of dimension bar_p 
    using the top principal components. The projection matrix R is computed via 
    truncated singular value decomposition (SVD).

    Parameters:
    - df_data: DataFrame containing a column with embedding vectors.
    - bar_p: Size of compression shape.
    - var_embd: Name of the column with embeddings (default: 'embeddings').
    - verbose: If True, prints explained variance ratio and dimension used.

    Returns:
    - V_proj: Compressed embeddings of shape (N, bar_p).
    - R_opt: Compression matrix of shape (bar_p, p) derived from PCA.
    """

    df = df_data.copy()

    embeddings = torch.tensor(np.stack(df[var_embd].values), dtype=torch.float32)
    V_numpy = embeddings.numpy()

    V_centered = V_numpy - V_numpy.mean(axis=0)
    V = torch.tensor(V_centered, dtype=torch.float32)
    N, p = V.shape

    # SVD
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    R_opt = Vh[:bar_p].numpy()

    # Explained variance
    total_variance = (S**2).sum().item()
    explained_variance = (S[:bar_p]**2).sum().item()
    explained_ratio = explained_variance / total_variance

    if verbose:
        print(f"Compression dimension (bar_p): {bar_p}")
        print(f"Explained variance ratio: {explained_ratio:.4%}")

    return V_numpy @ R_opt.T, R_opt


def apply_linear_projection(df_input, R, var_embd='embeddings'):
    """
    Applies a fixed linear projection matrix to compress high-dimensional embeddings
    into a lower-dimensional space.

    This is typically used at test time, where a projection matrix R
    (obtained during training) is used to map input embeddings to a reduced space.
    The method performs a linear transformation of the form:
        Z_projected = R @ Z_original.T

    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame containing the input embeddings in column `var_embd`.
    R : np.ndarray
        The linear projection matrix (shape [k, p]) used to compress the embeddings
        from original dimension p to compressed dimension k.
    var_embd : str, default='embeddings'
        Name of the column in `df_input` containing the original embedding vectors (as lists or arrays).

    Returns
    -------
    df_projected : pd.DataFrame
        Copy of the original DataFrame with compressed embeddings in `var_embd`.
    """

    df_projected = df_input.copy()

    # Stack embeddings into a matrix shape [n_samples, p]
    Z_original = np.vstack(df_projected[var_embd].values)

    # Apply the linear projection: shape [n_samples, k]
    Z_compressed = (R @ Z_original.T).T

    # Convert each row back to a list to store in the DataFrame
    Z_compressed_list = [list(row) for row in Z_compressed]

    # Replace the embedding column with the compressed version
    df_projected[var_embd] = Z_compressed_list

    return df_projected