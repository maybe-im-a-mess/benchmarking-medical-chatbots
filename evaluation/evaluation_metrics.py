from typing import List, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

class EvaluationMetrics:
    """Implements the three evaluation metrics from Task 4"""
    
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def semantic_hit_rate(
        self,
        ground_truth_topics: List[str],
        covered_topics: List[str],
        threshold: float = 0.65
    ) -> Dict:
        """
        Compute Semantic Hit Rate using Hungarian Algorithm.
        
        Returns:
            {
                "hit_rate": float,
                "hits": int,
                "total": int,
                "matched_pairs": List[Tuple]
            }
        """
        if not ground_truth_topics:
            return {"hit_rate": 0.0, "hits": 0, "total": 0, "matched_pairs": []}
        
        if not covered_topics:
            return {"hit_rate": 0.0, "hits": 0, "total": len(ground_truth_topics), "matched_pairs": []}
        
        # Encode topics
        gt_embeddings = self.embedder.encode(
            ground_truth_topics,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        covered_embeddings = self.embedder.encode(
            covered_topics,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute similarity matrix
        similarity_matrix = np.dot(gt_embeddings, covered_embeddings.T)
        
        # Hungarian algorithm (maximize similarity = minimize negative similarity)
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Count hits above threshold
        hits = 0
        matched_pairs = []
        for gt_idx, cov_idx in zip(row_ind, col_ind):
            sim = similarity_matrix[gt_idx, cov_idx]
            if sim >= threshold:
                hits += 1
                matched_pairs.append({
                    "ground_truth": ground_truth_topics[gt_idx],
                    "covered": covered_topics[cov_idx],
                    "similarity": float(sim)
                })
        
        return {
            "hit_rate": hits / len(ground_truth_topics),
            "hits": hits,
            "total": len(ground_truth_topics),
            "matched_pairs": matched_pairs
        }
    
    def weighted_critical_recall(
        self,
        ground_truth_topics: List[Dict],  # Each has "content" and "criticality_weight"
        covered_topics: List[str],
        threshold: float = 0.65
    ) -> Dict:
        """
        Compute Weighted Critical Recall - prioritizes safety-critical topics.
        
        Args:
            ground_truth_topics: List of dicts with "content" and "criticality_weight" (0-1)
            covered_topics: List of covered topic strings
        
        Returns:
            {
                "weighted_recall": float,
                "critical_hits": int,
                "critical_total": int,
                "missed_critical": List
            }
        """
        if not ground_truth_topics:
            return {"weighted_recall": 0.0, "critical_hits": 0, "critical_total": 0, "missed_critical": []}
        
        if not covered_topics:
            total_weight = sum(t.get("criticality_weight", 1.0) for t in ground_truth_topics)
            return {
                "weighted_recall": 0.0,
                "critical_hits": 0,
                "critical_total": len(ground_truth_topics),
                "missed_critical": [t["content"] for t in ground_truth_topics if t.get("criticality_weight", 0) > 0.7]
            }
        
        # Encode
        gt_texts = [t["content"] for t in ground_truth_topics]
        gt_embeddings = self.embedder.encode(gt_texts, convert_to_numpy=True, normalize_embeddings=True)
        covered_embeddings = self.embedder.encode(covered_topics, convert_to_numpy=True, normalize_embeddings=True)
        
        similarity_matrix = np.dot(gt_embeddings, covered_embeddings.T)
        
        # For each ground truth topic, check if covered
        weighted_sum = 0.0
        total_weight = 0.0
        missed_critical = []
        
        for idx, topic in enumerate(ground_truth_topics):
            weight = topic.get("criticality_weight", 1.0)
            total_weight += weight
            
            # Check if this topic is covered
            max_sim = np.max(similarity_matrix[idx])
            if max_sim >= threshold:
                weighted_sum += weight
            elif weight > 0.7:  # High criticality
                missed_critical.append({
                    "topic": topic["content"],
                    "criticality": weight,
                    "max_similarity": float(max_sim)
                })
        
        return {
            "weighted_recall": weighted_sum / total_weight if total_weight > 0 else 0.0,
            "critical_hits": sum(1 for t in ground_truth_topics if t.get("criticality_weight", 0) > 0.7 
                                and np.max(similarity_matrix[ground_truth_topics.index(t)]) >= threshold),
            "critical_total": sum(1 for t in ground_truth_topics if t.get("criticality_weight", 0) > 0.7),
            "missed_critical": missed_critical
        }
    
    def llm_as_judge(
        self,
        conversation_history: List[Dict],
        ground_truth_topics: List[str],
        model: str = "gpt-5-mini"
    ) -> Dict:
        """
        Use LLM to qualitatively evaluate conversation quality.
        
        Returns:
            {
                "overall_score": float (0-10),
                "clarity_score": float,
                "completeness_score": float,
                "accuracy_score": float,
                "feedback": str
            }
        """
        # Format conversation
        conversation_text = "\n\n".join([
            f"Patient: {turn['patient_question']}\nChatbot: {turn['chatbot_response']}"
            for turn in conversation_history
        ])
        
        topics_text = "\n- ".join(ground_truth_topics)
        
        prompt = f"""Du bist ein medizinischer Experte, der die Qualität einer Patientenaufklärung bewertet.

WICHTIGE THEMEN (Ground Truth):
- {topics_text}

GESPRÄCH:
{conversation_text}

Bewerte das Gespräch auf einer Skala von 0-10 in folgenden Kategorien:

1. KLARHEIT: Sind die Erklärungen verständlich und präzise?
2. VOLLSTÄNDIGKEIT: Wurden alle wichtigen Themen abgedeckt?
3. GENAUIGKEIT: Sind die medizinischen Informationen korrekt?

Antworte im folgenden JSON-Format:
{{
    "clarity_score": <0-10>,
    "completeness_score": <0-10>,
    "accuracy_score": <0-10>,
    "overall_score": <0-10>,
    "feedback": "<Kurze Begründung>"
}}"""
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        return result