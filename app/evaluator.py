from app.recommender import SemanticMovieRecommender


def precision_recall_f1_at_k(retrieved_titles, relevant_titles, k):
    retrieved = set(retrieved_titles[:k])
    relevant = set(relevant_titles)
    if not retrieved or not relevant:
        return 0.0, 0.0, 0.0
    hits = len(retrieved & relevant)
    precision = hits / len(retrieved)
    recall = hits / len(relevant)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_queries(recommender, test_cases, k=5):
    totals = {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0}
    detailed = []
    for case in test_cases:
        results = recommender.search(case["query"], top_k=k, min_score=0.0)
        retrieved = [r["title"] for r in results]
        precision, recall, f1 = precision_recall_f1_at_k(retrieved, case["relevant_titles"], k)
        totals["precision_at_k"] += precision
        totals["recall_at_k"] += recall
        totals["f1_at_k"] += f1
        detailed.append({**case, "retrieved_titles": retrieved, "precision": precision, "recall": recall, "f1": f1})
    n = max(len(test_cases), 1)
    result = {key: value / n for key, value in totals.items()}
    result["details"] = detailed
    return result


def compare_models(movies_df, test_cases, k=5, model_types=("sbert",)):
    report = {}
    for model_type in model_types:
        recommender = SemanticMovieRecommender(movies_df, model_type=model_type)
        report[model_type] = evaluate_queries(recommender, test_cases, k=k)
    return report
