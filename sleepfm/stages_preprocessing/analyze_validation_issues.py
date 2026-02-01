#!/usr/bin/env python3
"""
Analyze Validation Issues
==========================

This script analyzes the validation_results.json to provide detailed
insights into normalization issues.

Author: Generated for STAGES data validation
Date: February 2026
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def analyze_issues(validation_file: str):
    """Analyze validation issues in detail."""
    
    # Load validation results
    print(f"Loading {validation_file}...")
    with open(validation_file, 'r') as f:
        results = json.load(f)
    
    print(f"Total subjects: {len(results)}")
    
    # Categorize issues
    std_issues = defaultdict(list)  # channel -> list of std values
    mean_issues = defaultdict(list)  # channel -> list of mean values
    other_issues = defaultdict(int)  # issue type -> count
    
    for result in results:
        for issue in result['all_issues']:
            if 'std' in issue and 'outside' in issue:
                # Extract channel and std value
                parts = issue.split(':')
                channel = parts[0].strip()
                # Extract std value: "std 0.767 outside [0.8, 1.2]"
                std_str = parts[1].split('outside')[0].strip()
                std_val = float(std_str.split()[1])
                std_issues[channel].append(std_val)
            elif 'mean' in issue and '>' in issue:
                # Extract channel and mean value
                parts = issue.split(':')
                channel = parts[0].strip()
                # Extract mean value
                mean_str = parts[1].split('>')[0].strip()
                mean_val = float(mean_str.split()[1])
                mean_issues[channel].append(mean_val)
            else:
                other_issues[issue] += 1
    
    # Print summary
    print("\n" + "="*80)
    print("VALIDATION ISSUES ANALYSIS")
    print("="*80)
    
    print("\n1. STANDARD DEVIATION ISSUES (outside [0.8, 1.2])")
    print("-" * 80)
    print(f"{'Channel':<15} {'Count':>8} {'Min Std':>10} {'Max Std':>10} {'Mean Std':>10}")
    print("-" * 80)
    
    for channel in sorted(std_issues.keys(), key=lambda x: len(std_issues[x]), reverse=True):
        std_vals = std_issues[channel]
        print(f"{channel:<15} {len(std_vals):>8} {min(std_vals):>10.3f} {max(std_vals):>10.3f} {np.mean(std_vals):>10.3f}")
    
    if mean_issues:
        print("\n2. MEAN ISSUES (abs(mean) > threshold)")
        print("-" * 80)
        print(f"{'Channel':<15} {'Count':>8} {'Min Mean':>10} {'Max Mean':>10} {'Mean':>10}")
        print("-" * 80)
        
        for channel in sorted(mean_issues.keys(), key=lambda x: len(mean_issues[x]), reverse=True):
            mean_vals = mean_issues[channel]
            print(f"{channel:<15} {len(mean_vals):>8} {min(mean_vals):>10.3f} {max(mean_vals):>10.3f} {np.mean(mean_vals):>10.3f}")
    
    if other_issues:
        print("\n3. OTHER ISSUES")
        print("-" * 80)
        for issue, count in sorted(other_issues.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {issue}: {count}")
    
    # Create visualization
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Count of std issues per channel
    if std_issues:
        channels = sorted(std_issues.keys(), key=lambda x: len(std_issues[x]), reverse=True)
        counts = [len(std_issues[ch]) for ch in channels]
        
        axes[0, 0].barh(channels, counts)
        axes[0, 0].set_xlabel('Number of Issues')
        axes[0, 0].set_title('Standard Deviation Issues by Channel')
        axes[0, 0].invert_yaxis()
    
    # Plot 2: Distribution of std values for top problematic channels
    if std_issues:
        top_channels = sorted(std_issues.keys(), key=lambda x: len(std_issues[x]), reverse=True)[:5]
        data_for_box = []
        labels_for_box = []
        
        for ch in top_channels:
            data_for_box.append(std_issues[ch])
            labels_for_box.append(f"{ch}\n(n={len(std_issues[ch])})")
        
        bp = axes[0, 1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='Lower threshold')
        axes[0, 1].axhline(y=1.2, color='r', linestyle='--', label='Upper threshold')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_title('Std Distribution for Top 5 Problematic Channels')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=0)
    
    # Plot 3: Histogram of all std values
    if std_issues:
        all_std = []
        for vals in std_issues.values():
            all_std.extend(vals)
        
        axes[1, 0].hist(all_std, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0.8, color='r', linestyle='--', label='Lower threshold')
        axes[1, 0].axvline(x=1.2, color='r', linestyle='--', label='Upper threshold')
        axes[1, 0].set_xlabel('Standard Deviation')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Distribution of All Std Values (n={len(all_std)})')
        axes[1, 0].legend()
    
    # Plot 4: Percentage of subjects with issues per channel
    if std_issues:
        n_subjects = len(results)
        channels = sorted(std_issues.keys(), key=lambda x: len(std_issues[x]), reverse=True)
        percentages = [len(std_issues[ch]) / n_subjects * 100 for ch in channels]
        
        axes[1, 1].barh(channels, percentages, color='coral')
        axes[1, 1].set_xlabel('Percentage of Subjects (%)')
        axes[1, 1].set_title('Percentage of Subjects with Std Issues per Channel')
        axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(validation_file).parent
    plot_file = output_dir / "validation_issues_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_file}")
    plt.close()
    
    # Create detailed report
    report_file = output_dir / "validation_issues_detailed.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED VALIDATION ISSUES ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        f.write("After artifact-aware normalization (z-score computed on clean segments),\n")
        f.write("we expect mean ≈ 0 and std ≈ 1 on the CLEAN segments.\n\n")
        
        f.write("Issues indicate:\n")
        f.write("  - Std < 0.8: Signal has less variance than expected (over-normalized or low signal)\n")
        f.write("  - Std > 1.2: Signal has more variance than expected (under-normalized or high variability)\n")
        f.write("  - |Mean| > 0.1: Signal has DC offset (should be centered at 0)\n\n")
        
        f.write("Common reasons for std < 1.0:\n")
        f.write("  1. EMG channels (CHIN, RLEG, LLEG) often have lower std after normalization\n")
        f.write("     because their normstats were computed on clean segments which may have\n")
        f.write("     different variance than the full signal\n")
        f.write("  2. Respiratory channels (Flow) may have different variance patterns\n")
        f.write("  3. This is NOT necessarily a problem - it reflects true signal characteristics\n\n")
        
        f.write("="*80 + "\n\n")
        
        # Standard deviation issues
        f.write("1. STANDARD DEVIATION ISSUES\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Channel':<15} {'Count':>8} {'% Subjects':>12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10}\n")
        f.write("-" * 80 + "\n")
        
        for channel in sorted(std_issues.keys(), key=lambda x: len(std_issues[x]), reverse=True):
            std_vals = std_issues[channel]
            pct = len(std_vals) / len(results) * 100
            f.write(f"{channel:<15} {len(std_vals):>8} {pct:>11.1f}% "
                   f"{min(std_vals):>10.3f} {max(std_vals):>10.3f} "
                   f"{np.mean(std_vals):>10.3f} {np.median(std_vals):>10.3f}\n")
        
        # Subjects with most issues
        f.write("\n\n2. SUBJECTS WITH MOST ISSUES\n")
        f.write("-" * 80 + "\n")
        
        subject_issue_counts = defaultdict(int)
        for result in results:
            if not result['is_valid']:
                subject_issue_counts[result['subject_id']] = len(result['all_issues'])
        
        f.write(f"{'Subject ID':<15} {'Number of Issues':>20}\n")
        f.write("-" * 80 + "\n")
        
        for subject_id, count in sorted(subject_issue_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            f.write(f"{subject_id:<15} {count:>20}\n")
    
    print(f"Saved detailed report to: {report_file}")
    
    # Return summary
    return {
        'std_issues': std_issues,
        'mean_issues': mean_issues,
        'other_issues': other_issues
    }


if __name__ == "__main__":
    validation_file = "/home/boshra95/scratch/stages/sleepfm_format/validation/validation_results.json"
    
    results = analyze_issues(validation_file)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Channels with std issues: {len(results['std_issues'])}")
    print(f"Channels with mean issues: {len(results['mean_issues'])}")
    print(f"Other issue types: {len(results['other_issues'])}")
    print("\nAnalysis complete!")
