"""Evaluation script for recommendation models."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import yaml

from src.utils import (
    load_config,
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_coverage
)


class RecommenderEvaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self, ground_truth_path: str):
        """
        Initialize evaluator with ground truth.
        
        Args:
            ground_truth_path: Path to ground truth CSV file
                             Format: visit_id, product_ids (space-separated)
        """
        print(f"\n=== Loading Ground Truth ===")
        self.gt_df = pd.read_csv(ground_truth_path)
        
        # Parse ground truth
        self.ground_truth = {}
        for _, row in self.gt_df.iterrows():
            visit_id = row['visit_id']
            # Parse space-separated product IDs (handle floats)
            product_ids = [int(float(x)) for x in str(row['product_ids']).split()]
            # Ground truth is the first item (next item to predict)
            if product_ids:
                self.ground_truth[visit_id] = product_ids[0]
        
        print(f"✓ Loaded {len(self.ground_truth)} ground truth labels")
    
    def evaluate_submission(
        self,
        submission_path: str,
        k: int = 6
    ) -> Dict[str, float]:
        """
        Evaluate a submission CSV file.
        
        Args:
            submission_path: Path to submission CSV
                           Format: visit_id, product_ids (space-separated, top-6)
            k: Recall@K
            
        Returns:
            Dictionary with metrics
        """
        print(f"\n=== Evaluating Submission: {submission_path} ===")
        
        # Load submission
        sub_df = pd.read_csv(submission_path)
        
        predictions = []
        ground_truth_list = []
        valid_count = 0
        
        for _, row in sub_df.iterrows():
            visit_id = row['visit_id']
            
            if visit_id not in self.ground_truth:
                continue
            
            # Parse predictions (handle floats)
            pred_ids = [int(float(x)) for x in str(row['product_ids']).split()]
            
            predictions.append(pred_ids[:k])
            ground_truth_list.append(self.ground_truth[visit_id])
            valid_count += 1
        
        # Calculate metrics
        recall = calculate_recall_at_k(predictions, ground_truth_list, k)
        ndcg = calculate_ndcg_at_k(predictions, ground_truth_list, k)
        
        # Coverage
        all_predicted = set()
        for pred in predictions:
            all_predicted.update(pred)
        
        all_gt = set(self.ground_truth.values())
        coverage = len(all_predicted) / len(all_gt) if all_gt else 0.0
        
        metrics = {
            f'recall@{k}': recall,
            f'ndcg@{k}': ndcg,
            'coverage': coverage,
            'num_sessions': valid_count
        }
        
        # Print results
        print(f"\nResults:")
        print(f"  Recall@{k}: {recall:.5f}")
        print(f"  NDCG@{k}: {ndcg:.5f}")
        print(f"  Coverage: {coverage:.4f}")
        print(f"  Sessions: {valid_count}")
        
        return metrics
    
    def analyze_failures(
        self,
        submission_path: str,
        k: int = 6,
        num_examples: int = 10
    ):
        """
        Analyze failure cases.
        
        Args:
            submission_path: Path to submission CSV
            k: Top-K
            num_examples: Number of examples to show
        """
        print(f"\n=== Failure Analysis ===")
        
        # Load submission
        sub_df = pd.read_csv(submission_path)
        
        failures = []
        
        for _, row in sub_df.iterrows():
            visit_id = row['visit_id']
            
            if visit_id not in self.ground_truth:
                continue
            
            pred_ids = [int(float(x)) for x in str(row['product_ids']).split()][:k]
            gt_id = self.ground_truth[visit_id]
            
            if gt_id not in pred_ids:
                failures.append({
                    'visit_id': visit_id,
                    'ground_truth': gt_id,
                    'predictions': pred_ids
                })
        
        print(f"\nTotal failures: {len(failures)}")
        print(f"Showing first {num_examples} examples:\n")
        
        for i, failure in enumerate(failures[:num_examples]):
            print(f"{i+1}. Visit {failure['visit_id']}")
            print(f"   Ground Truth: {failure['ground_truth']}")
            print(f"   Predictions: {failure['predictions']}")
            print()
    
    def compare_submissions(
        self,
        submission_paths: List[str],
        labels: List[str],
        k: int = 6
    ):
        """
        Compare multiple submissions.
        
        Args:
            submission_paths: List of submission file paths
            labels: Labels for each submission
            k: Recall@K
        """
        print(f"\n=== Comparing Submissions ===")
        
        results = []
        
        for path, label in zip(submission_paths, labels):
            metrics = self.evaluate_submission(path, k)
            metrics['label'] = label
            results.append(metrics)
        
        # Create comparison table
        print(f"\n{'Model':<20} {'Recall@6':<12} {'NDCG@6':<12} {'Coverage':<12}")
        print("=" * 56)
        
        for result in results:
            print(f"{result['label']:<20} "
                  f"{result[f'recall@{k}']:<12.5f} "
                  f"{result[f'ndcg@{k}']:<12.5f} "
                  f"{result['coverage']:<12.4f}")


def evaluate_against_ground_truth(
    submission_csv: str,
    ground_truth_csv: str = "95)submission.csv",
    verbose: bool = True
) -> float:
    """
    Quick evaluation function.
    
    Args:
        submission_csv: Path to submission
        ground_truth_csv: Path to ground truth
        verbose: Print details
        
    Returns:
        Recall@6 score
    """
    evaluator = RecommenderEvaluator(ground_truth_csv)
    metrics = evaluator.evaluate_submission(submission_csv, k=6)
    
    return metrics['recall@6']


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate recommendation submissions")
    parser.add_argument("--submission", type=str, required=True,
                       help="Path to submission CSV")
    parser.add_argument("--ground_truth", type=str, default="95)submission.csv",
                       help="Path to ground truth CSV")
    parser.add_argument("--analyze", action="store_true",
                       help="Run failure analysis")
    parser.add_argument("--compare", type=str, nargs="+",
                       help="Compare multiple submissions")
    
    args = parser.parse_args()
    
    # Evaluate single submission
    if not args.compare:
        evaluator = RecommenderEvaluator(args.ground_truth)
        evaluator.evaluate_submission(args.submission, k=6)
        
        if args.analyze:
            evaluator.analyze_failures(args.submission, k=6, num_examples=20)
    
    # Compare multiple submissions
    else:
        evaluator = RecommenderEvaluator(args.ground_truth)
        
        all_submissions = [args.submission] + args.compare
        labels = [f"Model_{i+1}" for i in range(len(all_submissions))]
        
        evaluator.compare_submissions(all_submissions, labels, k=6)
