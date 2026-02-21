########################################################################################
#                                                                                      #
#                                                                                      #
#                                                                                      #
#                                   SIF Embedding package                              #
#                                                                                      #
#                                                                                      #
#                                                                                      #
########################################################################################


"""
SIF Embedding by Arora et al. 2017
Please refer to the GitHub of the original authors                 

                           https://github.com/PrincetonML/SIF
                           
"""

import numpy as np
from tqdm import tqdm 
from sklearn.decomposition import TruncatedSVD
import torch

# Classe pour définir les paramètres
class Params:
    def __init__(self, rmpc):
        self.rmpc = rmpc


def get_weighted_average(We, x, w):
    """
    Compute the weighted average sentence embeddings using pre-trained word embeddings.

    This function implements the first step of the SIF (Smooth Inverse Frequency) method,
    as proposed by Arora et al. (2017). It computes a weighted average of word vectors 
    for each sentence in the input batch.

    Parameters
    ----------
    We : np.ndarray of shape (vocab_size, emb_dim)
        Matrix of pre-trained word embeddings. Each row We[i, :] corresponds to the embedding
        vector for the i-th word in the vocabulary.
    x : np.ndarray of shape (n_samples, max_len)
        Matrix of word indices. Each row x[i, :] contains the indices of words in sentence i,
        padded to a common length.
    w : np.ndarray of shape (n_samples, max_len)
        Matrix of word weights. Each entry w[i, j] corresponds to the weight for the j-th word
        in the i-th sentence. Typically, weights are based on word frequency (e.g., inverse frequency).

    Returns
    -------
    emb : np.ndarray of shape (n_samples, emb_dim)
        Matrix of sentence embeddings. Each row emb[i, :] is the weighted average of the 
        word vectors in sentence i.

    Notes
    -----
    - Sentences are assumed to be pre-tokenized and represented by word indices.
    - Zero weights are ignored in the averaging to prevent division by zero.
    - This is the first step of the SIF embedding method prior to principal component removal.
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i, :] = w[i, :].dot(We[x[i, :], :]) / np.count_nonzero(w[i, :])
    return emb


def compute_pc(X, npc=1):
    """
    Compute the top principal components of a matrix without centering the data.

    This function uses Truncated Singular Value Decomposition (SVD) to extract 
    the top `npc` principal components from the input matrix. Unlike traditional 
    PCA, the input data is not mean-centered before decomposition.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix. Each row corresponds to a sample.
    npc : int, default=1
        Number of principal components to compute.

    Returns
    -------
    components_ : np.ndarray of shape (npc, n_features)
        Matrix containing the top principal components as rows.

    Notes
    -----
    - NaNs and infinite values in `X` are replaced with 0 before SVD.
    - This approach is consistent with the SIF embedding method,
      which avoids centering the data before component removal.
    """
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Input matrix still contains non-numeric values after cleaning.")

    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove projection of data onto the top principal components.

    This function removes the contribution of the top `npc` principal components 
    from the input data matrix. It is typically used after computing 
    weighted averages in the SIF embedding method.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix.
    npc : int, default=1
        Number of principal components to remove.

    Returns
    -------
    X_denoised : np.ndarray of shape (n_samples, n_features)
        Matrix with projections onto the top components removed.
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        X_denoised = X - X.dot(pc.T) * pc
    else:
        X_denoised = X - X.dot(pc.T).dot(pc)
    return X_denoised


def SIF_embedding(We, x, w, params):
    """
    Compute sentence embeddings using the SIF method (Arora et al., 2017).

    This function performs the full SIF embedding procedure:
    1. Computes a weighted average of word embeddings.
    2. Optionally removes the projection onto the first principal components.

    Parameters
    ----------
    We : np.ndarray of shape (vocab_size, emb_dim)
        Word embedding matrix. Each row corresponds to a word vector.
    x : np.ndarray of shape (n_samples, max_len)
        Matrix of word indices for each sentence.
    w : np.ndarray of shape (n_samples, max_len)
        Weight matrix for each word in each sentence (typically inverse word frequency).
    params : object
        Object with attribute `rmpc` specifying how many principal components to remove.

    Returns
    -------
    emb : np.ndarray of shape (n_samples, emb_dim)
        Final sentence embeddings after component removal.
    """
    emb = get_weighted_average(We, x, w)
    if params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb


def get_word_embeddings(text, tokenizer, model, device):
    """
    Extract word-level embeddings from a RoBERTa-like transformer model (e.g., OncoBERT).

    This function tokenizes the input text, passes it through the transformer model,
    and retrieves contextual embeddings for each token. It returns a dictionary
    mapping each token (as string) to its corresponding embedding vector.

    Special tokens such as <s> and </s> (used in RoBERTa-based models) are removed
    from the returned dictionary.

    Parameters
    ----------
    text : str
        Input text string to be tokenized and embedded.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer compatible with the pre-trained model.
    model : transformers.PreTrainedModel
        RoBERTa-like transformer model (e.g., OncoBERT) providing hidden states.
    device : torch.device
        Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns
    -------
    word_embeddings : dict
        Dictionary mapping each token (as a string) to its corresponding embedding vector (np.ndarray).

    Notes
    -----
    - This method returns embeddings for subword tokens, as handled by the tokenizer.
    - It assumes the model returns all hidden states (`output_hidden_states=True`).
    - Batch size is set to 1; this function is not optimized for large-scale batching.
    """
    # Tokenize and send tensors to the correct device
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in tokens.items()}

    # Inference with no gradient tracking
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract last hidden state
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]

    # Get token IDs and convert them to tokens
    token_ids = inputs['input_ids'].squeeze()  # Remove batch dimension
    tokens_list = tokenizer.convert_ids_to_tokens(token_ids)

    # Build a dictionary mapping each token to its embedding vector
    word_embeddings = {
        token: last_hidden_state[0, idx].cpu().numpy()
        for idx, token in enumerate(tokens_list)
    }

    # Optionally remove special tokens (e.g., RoBERTa: <s>, </s>)
    word_embeddings.pop('<s>', None)
    word_embeddings.pop('</s>', None)

    return word_embeddings


def compute_sif_weights(word_occurrence, total_words, a=1e-3):
    """
    Compute Smooth Inverse Frequency (SIF) weights for words in a document.

    This function implements the weighting scheme from Arora et al. (2017),
    assigning each word a weight inversely proportional to its corpus frequency.

    Parameters
    ----------
    word_occurrence : dict
        Dictionary mapping each word to its count within the document.
    total_words : int
        Total number of words in the document (i.e., sum of all frequencies).
    a : float, default=1e-3
        Smoothing parameter for controlling the influence of word frequency.

    Returns
    -------
    dict
        Dictionary mapping each word to its computed SIF weight.

    Notes
    -----
    - Frequent words receive lower weights to reduce their dominance.
    - This is the first step in computing SIF-based sentence embeddings.
    """
    sif_weights = {
        word: a / (a + (freq / total_words))
        for word, freq in word_occurrence.items()
    }
    return sif_weights


def process_report(word_embeddings, a_sifembedding=1e-3):
    """
    Process a single report to prepare matrices for SIF embedding computation.

    Given a dictionary of word embeddings for a report, this function:
    - collects all word vectors,
    - computes word frequencies,
    - applies the SIF weighting scheme,
    - constructs input matrices `We`, `x`, and `w` compatible with the SIF embedding pipeline.

    Parameters
    ----------
    word_embeddings : dict
        Dictionary mapping each word (string) to its embedding vector (np.ndarray).
    a_sifembedding : float, default=1e-3
        Smoothing parameter used in SIF weighting.

    Returns
    -------
    We : np.ndarray of shape (n_words, emb_dim)
        Matrix of word embeddings.
    x : np.ndarray of shape (1, n_words)
        Matrix of word indices (used as placeholders for lookup in `We`).
    w : np.ndarray of shape (1, n_words)
        Matrix of corresponding SIF weights.

    Notes
    -----
    - This function is typically applied independently to each document before batch SIF computation.
    - The ordering in `x` and `w` corresponds to the order of embeddings in `We`.
    """
    all_words = []
    all_embeddings = []
    word_occurrence = {}

    # Collect embeddings and compute word frequencies
    for word, embedding in word_embeddings.items():
        all_words.append(word)
        all_embeddings.append(embedding)
        word_occurrence[word] = word_occurrence.get(word, 0) + 1

    # Stack embeddings into a matrix
    We = np.vstack(all_embeddings)

    # Compute SIF weights for this report
    total_words = len(all_words)
    sif_weights = compute_sif_weights(word_occurrence, total_words, a=a_sifembedding)

    # Build index and weight matrices for the sentence
    x = np.array([[i for i, _ in enumerate(all_words)]])
    w = np.array([[sif_weights[word] for word in all_words]])

    return We, x, w


def arora_methods(
    df,
    tokenizer, 
    model, 
    device,
    var_cible="text",
    a_sifembedding=1e-3
):
    """
    Compute sentence embeddings for a DataFrame using the SIF method proposed by Arora et al. (2017).

    This function applies a multi-step pipeline to generate sentence-level embeddings:
    1. Extract word-level embeddings for each sentence using a transformer model.
    2. Compute Smooth Inverse Frequency (SIF) weights based on word frequency.
    3. Perform weighted averaging of word embeddings.
    4. Optionally remove projection onto top principal components (rmpc = 0 by default).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the text data.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer compatible with the language model.
    model : transformers.PreTrainedModel
        Pre-trained language model (e.g., OncoBERT) used to generate embeddings.
    device : torch.device
        Target device for model inference (e.g., 'cuda' or 'cpu').
    var_cible : str, default="text"
        Name of the column in `df` containing the raw textual input.
    a_sifembedding : float, default=1e-3
        Smoothing parameter used for SIF weighting.

    Returns
    -------
    df : pd.DataFrame
        The original DataFrame with two additional columns:
        - 'word_embeddings': dictionary of word-level vectors for each row.
        - 'embeddings': final sentence embedding as a NumPy array.

    Notes
    -----
    - Uses the full SIF pipeline including weighted averaging and optional PCA projection removal.
    - The `Params` object with `rmpc=0` disables component removal by default.
    - Input text is processed row-by-row with progress bars shown via `tqdm`.
    """
    word_embeddings_list = []

    # Step 1: extract word embeddings for each text entry
    for text in tqdm(df[var_cible], desc="Computing word embeddings..."):
        word_embeddings = get_word_embeddings(text, tokenizer, model, device)
        word_embeddings_list.append(word_embeddings)

    df['word_embeddings'] = word_embeddings_list

    # Initialize SIF parameter object
    params = Params(rmpc=0)

    sentence_embeddings_list = []

    # Step 2: compute sentence embeddings using SIF
    for we in tqdm(df['word_embeddings'], desc="Computing sentence embeddings..."):
        We, x, w = process_report(we, a_sifembedding=a_sifembedding)
        sentence_embedding = SIF_embedding(We, x, w, params)
        sentence_embeddings_list.append(sentence_embedding)

    # Squeeze and store final sentence embeddings
    sentence_embeddings_list = [np.squeeze(embedding) for embedding in sentence_embeddings_list]
    df['embeddings'] = sentence_embeddings_list

    return df