from scipy import stats


def spearman(list_a, list_b):
    return stats.spearmanr(list_a, list_b).correlation


def pearson(list_a, list_b):
    c, _ = stats.pearsonr(list_a, list_b)
    return c
