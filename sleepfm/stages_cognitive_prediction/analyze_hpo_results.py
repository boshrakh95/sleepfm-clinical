#!/usr/bin/env python3
"""
Analyze Hyperparameter Optimization Results
============================================

This script analyzes the results from multiple HPO trials and finds the best
hyperparameter configuration.

Usage:
    python analyze_hpo_results.py --results_dir /path/to/cognitive_models

Author: Generated for STAGES HPO analysis
Date: February 2026
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List


def extract_hpo_results(results_dir: Path) -> pd.DataFrame:
    """Extract results from all HPO experiment directories.
    
    Args:
        results_dir: Directory containing experiment subdirectories
    
    Returns:
        DataFrame with hyperparameters and metrics for each trial
    """
    results = []
    
    # Find all experiment directories
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    for exp_dir in exp_dirs:
        # Look for final_metrics.json
        metrics_file = exp_dir / "final_metrics.json"
        
        if not metrics_file.exists():
            print(f"Warning: No metrics found in {exp_dir.name}")
            continue
        
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract hyperparameters from log file (if available)
        log_files = list(exp_dir.glob("*.log"))
        hyperparams = {}
        
        if log_files:
            log_file = log_files[0]
            with open(log_file, 'r') as f:
                for line in f:
                    # Parse override lines
                    if "Overriding" in line:
                        parts = line.split("Overriding")[1].strip().split(":")
                        if len(parts) == 2:
                            param_name = parts[0].strip()
                            param_value = parts[1].strip()
                            try:
                                # Try to convert to float
                                hyperparams[param_name] = float(param_value)
                            except:
                                hyperparams[param_name] = param_value
        
        # Combine hyperparameters and metrics
        result = {
            'experiment': exp_dir.name,
            **hyperparams,
            **{f'train_{k}': v for k, v in metrics.get('train', {}).items()},
            **{f'val_{k}': v for k, v in metrics.get('val', {}).items()},
            **{f'test_{k}': v for k, v in metrics.get('test', {}).items()},
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def find_best_config(df: pd.DataFrame, metric: str = 'val_Accuracy', higher_is_better: bool = True):
    """Find the best hyperparameter configuration.
    
    Args:
        df: DataFrame with results
        metric: Metric to optimize
        higher_is_better: Whether higher values are better
    
    Returns:
        Best configuration row
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in results. Available: {list(df.columns)}")
    
    if higher_is_better:
        best_idx = df[metric].idxmax()
    else:
        best_idx = df[metric].idxmin()
    
    return df.loc[best_idx]


def main():
    parser = argparse.ArgumentParser(description="Analyze HPO results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/boshra95/scratch/stages/sleepfm_format/cognitive_models",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_Accuracy",
        help="Metric to optimize (e.g., val_Accuracy, val_AUC, test_F1)"
    )
    parser.add_argument(
        "--higher_is_better",
        type=bool,
        default=True,
        help="Whether higher metric values are better"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hpo_results.csv",
        help="Output CSV file for results"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print("="*80)
    print("Hyperparameter Optimization Analysis")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Optimization metric: {args.metric}")
    print(f"Higher is better: {args.higher_is_better}")
    print()
    
    # Extract results
    print("Extracting results from experiments...")
    df = extract_hpo_results(results_dir)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"Found {len(df)} completed trials")
    print()
    
    # Save all results
    df.to_csv(args.output, index=False)
    print(f"Saved all results to: {args.output}")
    print()
    
    # Find best configuration
    try:
        best = find_best_config(df, args.metric, args.higher_is_better)
        
        print("="*80)
        print("BEST CONFIGURATION")
        print("="*80)
        print(f"Experiment: {best['experiment']}")
        print(f"Best {args.metric}: {best[args.metric]:.4f}")
        print()
        
        print("Hyperparameters:")
        hyperparam_cols = [c for c in df.columns if c not in ['experiment'] and not c.startswith(('train_', 'val_', 'test_'))]
        for col in hyperparam_cols:
            if col in best.index and pd.notna(best[col]):
                print(f"  {col}: {best[col]}")
        print()
        
        print("Performance:")
        metric_cols = [c for c in df.columns if c.startswith(('train_', 'val_', 'test_'))]
        for col in sorted(metric_cols):
            if col in best.index and pd.notna(best[col]):
                print(f"  {col}: {best[col]:.4f}")
        print()
        
        # Print top 5 configurations
        print("="*80)
        print(f"TOP 5 CONFIGURATIONS (by {args.metric})")
        print("="*80)
        
        if args.higher_is_better:
            top5 = df.nlargest(5, args.metric)
        else:
            top5 = df.nsmallest(5, args.metric)
        
        for idx, row in top5.iterrows():
            print(f"\n{row['experiment']}")
            print(f"  {args.metric}: {row[args.metric]:.4f}")
            for col in hyperparam_cols:
                if col in row.index and pd.notna(row[col]):
                    print(f"  {col}: {row[col]}")
        
        print()
        print("="*80)
        
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available metrics: {[c for c in df.columns if c.startswith(('train_', 'val_', 'test_'))]}")


if __name__ == "__main__":
    main()
