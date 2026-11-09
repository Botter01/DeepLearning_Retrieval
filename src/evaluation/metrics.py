from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def calculateRecall_k(relevancies: list[bool], query: Query, k: int):
    return sum(relevancies[:k]) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def calculatePrecision_k(relevancies: list[bool], query: Query, k: int):
    return sum(relevancies[:k]) / (len(relevancies[:k]))

def calculateMRR(relevancies: list[bool], query: Query):
    rr = 0
    for idx, val in enumerate(relevancies):
        if val:
            rr = 1/(idx + 1)
            break
    return rr

def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None, 'recall_k': None, 'precision_k': None, 'mrr': None}
    n_queries = len(queries)
    total_recall = 0
    total_precision_k = 0
    total_recall_k = 0
    total_mrr = 0

    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_recall_k += calculateRecall_k(relevancies[i], queries[i], 6)
        total_precision_k += calculatePrecision_k(relevancies[i], queries[i], 6)
        total_mrr += calculateMRR(relevancies[i], queries[i])
    
    scores['recall'] = total_recall / n_queries
    scores['recall_k'] = total_recall_k / n_queries
    scores['precision_k'] = total_precision_k / n_queries
    scores['mrr'] = total_mrr / n_queries
    return scores
