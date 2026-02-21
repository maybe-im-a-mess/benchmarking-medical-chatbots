# Create evaluation script: evaluate_conversations.py
from evaluation_metrics import EvaluationMetrics
import json

def evaluate_conversation_log(conversation_log_path: str):
    """Evaluate a saved conversation using all three metrics"""
    
    with open(conversation_log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
    
    metrics = EvaluationMetrics()
    
    # Load ground truth
    procedure = log["metadata"]["procedure"]
    ground_truth = load_ground_truth_topics(procedure)  # Your IE results
    
    # Extract covered topics from conversation
    covered_topics = extract_covered_topics(log)  # You need to implement this
    
    # Compute metrics
    results = {
        "semantic_hit_rate": metrics.semantic_hit_rate(
            [t["content"] for t in ground_truth],
            covered_topics
        ),
        "weighted_critical_recall": metrics.weighted_critical_recall(
            ground_truth,  # Must have criticality_weight field
            covered_topics
        ),
        "llm_judge": metrics.llm_as_judge(
            log["conversation"],
            [t["content"] for t in ground_truth]
        )
    }
    
    return results