import numpy as np

def precision_k(R, k):
    num_hits = np.sum(R <= k)
    prec_k = num_hits / k
    return prec_k

def recall_k(R, k):
    num_hits = np.sum(R <= k)
    recall_k = num_hits / len(R)
    return recall_k

def ndcg_k(R, k):
    # NDCG
    idcg = 0.0
    for i in range(min(len(R), k)):
        idcg += 1 / np.math.log2(i + 2)
    dcg = 0.0
    for i in range(k):
        if (i + 1) in R:
            dcg += 1 / np.math.log2(i + 2)
    ndcg_k = dcg / idcg
    return ndcg_k

def ap_k(R, k):
    ps = 0
    for i in range(k):
        if (i + 1) in R:
            ps += precision_k(R, i + 1)
    ap = ps / min(len(R), k)
    return ap

def rr_k(R, k):
    min_pos = min(R)
    rr = 1 / min_pos if min_pos <= k else 0.0
    return rr