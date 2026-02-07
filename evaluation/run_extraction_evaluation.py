"""
Evaluate all extraction methods across all documents.
Generates comprehensive comparison and rankings.
"""

import json
from pathlib import Path
from evaluation.extraction_quality import evaluate_all_methods, print_comparison_table, save_evaluation_results


def evaluate_all_documents():
    """Run evaluation for all 6 documents."""
    documents = [
        "Narkose",
        "Kaiserschnitt",
        "Geburtseinleitung",
        "Äußere Wendung",
        "Geburtshilfliche Maßnahmen",
        "DRK Geburtshilfe Infos"
    ]
    
    model_keys = ["qwen3-4b", "gpt-5-mini"]
    methods = ["naive", "uie", "atomic", "cot"]
    
    all_results = {}
    
    for doc in documents:
        print(f"\n{'='*100}")
        print(f"EVALUATING: {doc}")
        print(f"{'='*100}")
        
        results = evaluate_all_methods(
            document=doc,
            model_keys=model_keys,
            methods=methods,
            eval_model="gpt-5-mini",
            verbose=True
        )
        
        all_results[doc] = results
        
        print_comparison_table(results)
        save_evaluation_results(doc, results)
    
    # Save aggregate results
    output_path = Path("data/evaluation_results/extraction_quality_all.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*100}")
    print(f"✅ ALL EVALUATIONS COMPLETE")
    print(f"{'='*100}")
    print(f"\nAggregate results saved to: {output_path}")
    
    # Print overall rankings
    print_overall_rankings(all_results)


def print_overall_rankings(all_results: dict):
    """Print overall rankings across all documents."""
    print(f"\n{'='*100}")
    print("OVERALL RANKINGS (Average across all documents)")
    print(f"{'='*100}\n")
    
    # Aggregate scores by model/method
    from collections import defaultdict
    aggregates = defaultdict(lambda: {"f1_scores": [], "precisions": [], "recalls": []})
    
    for doc, doc_results in all_results.items():
        for model_key, methods_data in doc_results.items():
            for method, evaluation in methods_data.items():
                if evaluation.get("status") == "success":
                    key = f"{model_key}/{method}"
                    aggregates[key]["f1_scores"].append(evaluation["f1_score"])
                    aggregates[key]["precisions"].append(evaluation["precision"])
                    aggregates[key]["recalls"].append(evaluation["recall"])
    
    # Calculate averages
    rankings = []
    for key, data in aggregates.items():
        avg_f1 = sum(data["f1_scores"]) / len(data["f1_scores"]) if data["f1_scores"] else 0
        avg_precision = sum(data["precisions"]) / len(data["precisions"]) if data["precisions"] else 0
        avg_recall = sum(data["recalls"]) / len(data["recalls"]) if data["recalls"] else 0
        
        model, method = key.split("/")
        rankings.append({
            "model": model,
            "method": method,
            "avg_f1": avg_f1,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "n_docs": len(data["f1_scores"])
        })
    
    # Sort by average F1
    rankings.sort(key=lambda x: x["avg_f1"], reverse=True)
    
    # Print table
    print(f"{'Rank':<6}{'Model':<15}{'Method':<12}{'Avg Precision':<15}{'Avg Recall':<15}{'Avg F1':<12}{'Docs':<6}")
    print("-"*100)
    
    for rank, r in enumerate(rankings, 1):
        print(f"{rank:<6}"
              f"{r['model']:<15}"
              f"{r['method']:<12}"
              f"{r['avg_precision']:<15.2%}"
              f"{r['avg_recall']:<15.2%}"
              f"{r['avg_f1']:<12.3f}"
              f"{r['n_docs']:<6}")
    
    print("="*100)
    
    # Winner
    if rankings:
        winner = rankings[0]
        print(f"\n🏆 BEST OVERALL METHOD: {winner['model']} / {winner['method']}")
        print(f"   Average F1: {winner['avg_f1']:.3f}")
        print(f"   Average Precision: {winner['avg_precision']:.2%}")
        print(f"   Average Recall: {winner['avg_recall']:.2%}")
        print(f"   Evaluated on {winner['n_docs']} documents")


if __name__ == "__main__":
    evaluate_all_documents()
