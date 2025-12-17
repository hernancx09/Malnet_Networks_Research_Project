#!/usr/bin/env python3
"""
Comprehensive Comparison Table and CV Fold Analysis
Generates comparison tables and analyzes CV fold performance across all configurations
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_all_results(results_dir: Path) -> Dict:
    """
    Scan results directory and load all JSON result files
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary with train/test and CV results organized by configuration
    """
    results_dir = Path(results_dir)
    all_results = {
        'train_test': {},
        'cv': {}
    }
    
    # Find all result JSON files
    result_files = list(results_dir.glob("*_results.json"))
    
    for result_file in result_files:
        filename = result_file.stem  # e.g., "rf_gcm_results" or "svm_coarse_cv_results"
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Determine if it's CV or train/test
            if '_cv_results' in filename:
                # CV result
                config_name = filename.replace('_cv_results', '')
                all_results['cv'][config_name] = data
            elif filename.endswith('_results'):
                # Train/test result (skip individual run files)
                if 'run_' not in filename:
                    config_name = filename.replace('_results', '')
                    all_results['train_test'][config_name] = data
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")
            continue
    
    return all_results


def extract_metrics(result_data: Dict, is_cv: bool = False) -> Dict:
    """
    Extract key metrics from result dictionary
    
    Args:
        result_data: Result dictionary from JSON file
        is_cv: Whether this is a CV result (affects structure)
        
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}
    
    if is_cv:
        # CV results have cv_results.fold_results and cv_results.consistency
        if 'cv_results' in result_data:
            cv_data = result_data['cv_results']
            
            # Extract consistency metrics
            if 'consistency' in cv_data:
                consistency = cv_data['consistency']
                for metric_name, stats in consistency.items():
                    if isinstance(stats, dict):
                        metrics[f'{metric_name}_mean'] = stats.get('mean', None)
                        metrics[f'{metric_name}_std'] = stats.get('std', None)
                        metrics[f'{metric_name}_min'] = stats.get('min', None)
                        metrics[f'{metric_name}_max'] = stats.get('max', None)
            
            # Extract fold results for detailed analysis
            if 'fold_results' in cv_data:
                metrics['fold_results'] = cv_data['fold_results']
            
            # Also extract final model metrics (trained on full dataset)
            metrics['final_accuracy'] = result_data.get('accuracy', None)
            metrics['final_macro_f1'] = result_data.get('macro_f1', None)
            metrics['final_macro_auc'] = result_data.get('macro_auc', None)
    else:
        # Train/test results - can be aggregated (mean/std) or single values
        for metric_name in ['accuracy', 'macro_f1', 'macro_auc', 'macro_aupr']:
            value = result_data.get(metric_name)
            
            if isinstance(value, dict):
                # Aggregated format (mean/std)
                metrics[f'{metric_name}_mean'] = value.get('mean', None)
                metrics[f'{metric_name}_std'] = value.get('std', None)
                metrics[f'{metric_name}_min'] = value.get('min', None)
                metrics[f'{metric_name}_max'] = value.get('max', None)
            else:
                # Single value
                metrics[f'{metric_name}'] = value
        
        # Get number of runs if available
        metrics['n_runs'] = result_data.get('n_runs', 1)
    
    return metrics


def generate_comparison_table(all_results: Dict) -> pd.DataFrame:
    """
    Generate comprehensive comparison table
    
    Args:
        all_results: Dictionary with train/test and CV results
        
    Returns:
        DataFrame with comparison metrics
    """
    rows = []
    
    # Get all unique configurations
    all_configs = set(all_results['train_test'].keys()) | set(all_results['cv'].keys())
    
    for config_name in sorted(all_configs):
        row = {'Configuration': config_name}
        
        # Extract train/test metrics
        if config_name in all_results['train_test']:
            tt_metrics = extract_metrics(all_results['train_test'][config_name], is_cv=False)
            
            # Format train/test metrics
            if 'accuracy_mean' in tt_metrics:
                row['TT_Accuracy'] = f"{tt_metrics['accuracy_mean']:.4f} ± {tt_metrics['accuracy_std']:.4f}"
                row['TT_Accuracy_Mean'] = tt_metrics['accuracy_mean']
            elif 'accuracy' in tt_metrics:
                row['TT_Accuracy'] = f"{tt_metrics['accuracy']:.4f}"
                row['TT_Accuracy_Mean'] = tt_metrics['accuracy']
            
            if 'macro_f1_mean' in tt_metrics:
                row['TT_Macro_F1'] = f"{tt_metrics['macro_f1_mean']:.4f} ± {tt_metrics['macro_f1_std']:.4f}"
            elif 'macro_f1' in tt_metrics:
                row['TT_Macro_F1'] = f"{tt_metrics['macro_f1']:.4f}"
            
            if 'macro_auc_mean' in tt_metrics:
                row['TT_Macro_AUC'] = f"{tt_metrics['macro_auc_mean']:.4f} ± {tt_metrics['macro_auc_std']:.4f}"
            elif 'macro_auc' in tt_metrics:
                row['TT_Macro_AUC'] = f"{tt_metrics['macro_auc']:.4f}"
            
            row['TT_N_Runs'] = tt_metrics.get('n_runs', 1)
        else:
            row['TT_Accuracy'] = 'N/A'
            row['TT_Macro_F1'] = 'N/A'
            row['TT_Macro_AUC'] = 'N/A'
            row['TT_Accuracy_Mean'] = None
        
        # Extract CV metrics
        if config_name in all_results['cv']:
            cv_metrics = extract_metrics(all_results['cv'][config_name], is_cv=True)
            
            # Format CV metrics
            if 'accuracy_mean' in cv_metrics:
                row['CV_Accuracy'] = f"{cv_metrics['accuracy_mean']:.4f} ± {cv_metrics['accuracy_std']:.4f}"
                row['CV_Accuracy_Mean'] = cv_metrics['accuracy_mean']
                row['CV_Accuracy_Std'] = cv_metrics['accuracy_std']
                
                # Calculate consistency (std as % of mean)
                if cv_metrics['accuracy_mean'] > 0:
                    row['CV_Consistency'] = f"{(cv_metrics['accuracy_std'] / cv_metrics['accuracy_mean'] * 100):.2f}%"
            else:
                row['CV_Accuracy'] = 'N/A'
                row['CV_Accuracy_Mean'] = None
            
            if 'macro_f1_mean' in cv_metrics:
                row['CV_Macro_F1'] = f"{cv_metrics['macro_f1_mean']:.4f} ± {cv_metrics['macro_f1_std']:.4f}"
            
            if 'macro_auc_mean' in cv_metrics:
                row['CV_Macro_AUC'] = f"{cv_metrics['macro_auc_mean']:.4f} ± {cv_metrics['macro_auc_std']:.4f}"
        else:
            row['CV_Accuracy'] = 'N/A'
            row['CV_Macro_F1'] = 'N/A'
            row['CV_Macro_AUC'] = 'N/A'
            row['CV_Accuracy_Mean'] = None
            row['CV_Consistency'] = 'N/A'
        
        # Calculate overfitting indicator (difference between TT and CV accuracy)
        if row.get('TT_Accuracy_Mean') is not None and row.get('CV_Accuracy_Mean') is not None:
            overfitting = row['TT_Accuracy_Mean'] - row['CV_Accuracy_Mean']
            row['Overfitting_Diff'] = f"{overfitting:.4f}"
            row['Overfitting_Diff_Value'] = overfitting
        else:
            row['Overfitting_Diff'] = 'N/A'
            row['Overfitting_Diff_Value'] = None
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Identify best configuration (by CV accuracy if available, else TT accuracy)
    if 'CV_Accuracy_Mean' in df.columns:
        best_idx = df['CV_Accuracy_Mean'].idxmax()
    elif 'TT_Accuracy_Mean' in df.columns:
        best_idx = df['TT_Accuracy_Mean'].idxmax()
    else:
        best_idx = None
    
    if best_idx is not None:
        df.loc[best_idx, 'Best'] = '★'
    
    # Reorder columns for better readability
    column_order = ['Configuration', 'TT_Accuracy', 'TT_Macro_F1', 'TT_Macro_AUC', 'TT_N_Runs',
                    'CV_Accuracy', 'CV_Macro_F1', 'CV_Macro_AUC', 'CV_Consistency',
                    'Overfitting_Diff', 'Best']
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    return df


def analyze_cv_folds(cv_results: Dict) -> Dict:
    """
    Analyze CV fold performance
    
    Args:
        cv_results: Dictionary with CV results for all configurations
        
    Returns:
        Dictionary with analysis for each configuration
    """
    analysis = {}
    
    for config_name, result_data in cv_results.items():
        config_analysis = {
            'config': config_name,
            'folds': [],
            'best_fold': None,
            'worst_fold': None,
            'fold_accuracy': [],
            'fold_f1': [],
            'fold_auc': [],
            'variance': {},
            'outliers': []
        }
        
        cv_metrics = extract_metrics(result_data, is_cv=True)
        
        if 'fold_results' in cv_metrics:
            folds = cv_metrics['fold_results']
            
            for fold_data in folds:
                fold_num = fold_data.get('fold', 0)
                accuracy = fold_data.get('accuracy', 0)
                f1 = fold_data.get('macro_f1', 0)
                auc = fold_data.get('macro_auc', 0)
                
                config_analysis['folds'].append({
                    'fold': fold_num,
                    'accuracy': accuracy,
                    'f1': f1,
                    'auc': auc
                })
                
                config_analysis['fold_accuracy'].append(accuracy)
                config_analysis['fold_f1'].append(f1)
                config_analysis['fold_auc'].append(auc)
            
            # Identify best and worst folds
            if config_analysis['fold_accuracy']:
                best_idx = np.argmax(config_analysis['fold_accuracy'])
                worst_idx = np.argmin(config_analysis['fold_accuracy'])
                
                config_analysis['best_fold'] = {
                    'fold': config_analysis['folds'][best_idx]['fold'],
                    'accuracy': config_analysis['fold_accuracy'][best_idx],
                    'f1': config_analysis['fold_f1'][best_idx],
                    'auc': config_analysis['fold_auc'][best_idx]
                }
                
                config_analysis['worst_fold'] = {
                    'fold': config_analysis['folds'][worst_idx]['fold'],
                    'accuracy': config_analysis['fold_accuracy'][worst_idx],
                    'f1': config_analysis['fold_f1'][worst_idx],
                    'auc': config_analysis['fold_auc'][worst_idx]
                }
                
                # Calculate variance
                config_analysis['variance'] = {
                    'accuracy': {
                        'std': np.std(config_analysis['fold_accuracy']),
                        'range': np.max(config_analysis['fold_accuracy']) - np.min(config_analysis['fold_accuracy']),
                        'cv': np.std(config_analysis['fold_accuracy']) / np.mean(config_analysis['fold_accuracy']) if np.mean(config_analysis['fold_accuracy']) > 0 else 0
                    },
                    'f1': {
                        'std': np.std(config_analysis['fold_f1']),
                        'range': np.max(config_analysis['fold_f1']) - np.min(config_analysis['fold_f1']),
                        'cv': np.std(config_analysis['fold_f1']) / np.mean(config_analysis['fold_f1']) if np.mean(config_analysis['fold_f1']) > 0 else 0
                    },
                    'auc': {
                        'std': np.std(config_analysis['fold_auc']),
                        'range': np.max(config_analysis['fold_auc']) - np.min(config_analysis['fold_auc']),
                        'cv': np.std(config_analysis['fold_auc']) / np.mean(config_analysis['fold_auc']) if np.mean(config_analysis['fold_auc']) > 0 else 0
                    }
                }
                
                # Identify outliers (folds more than 2 std from mean)
                mean_acc = np.mean(config_analysis['fold_accuracy'])
                std_acc = np.std(config_analysis['fold_accuracy'])
                
                for fold_data in config_analysis['folds']:
                    if abs(fold_data['accuracy'] - mean_acc) > 2 * std_acc:
                        config_analysis['outliers'].append({
                            'fold': fold_data['fold'],
                            'accuracy': fold_data['accuracy'],
                            'deviation': fold_data['accuracy'] - mean_acc
                        })
        
        analysis[config_name] = config_analysis
    
    return analysis


def create_cv_visualizations(cv_data: Dict, save_path: Path) -> None:
    """
    Create visualizations for CV fold analysis
    
    Args:
        cv_data: Dictionary with CV fold analysis data
        save_path: Path to save visualization
    """
    if not cv_data:
        print("No CV data available for visualization")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Validation Fold Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    config_names = []
    fold_accuracies = []
    colors = plt.cm.Set3(np.linspace(0, 1, len(cv_data)))
    
    for config_name, analysis in cv_data.items():
        if analysis['fold_accuracy']:
            config_names.append(config_name)
            fold_accuracies.append(analysis['fold_accuracy'])
    
    if not fold_accuracies:
        print("No fold data available for visualization")
        plt.close(fig)
        return
    
    # 1. Bar plot: Fold-by-fold accuracy for each configuration
    ax1 = axes[0, 0]
    x = np.arange(5)  # 5 folds
    width = 0.8 / len(config_names)
    
    for i, (config_name, accuracies) in enumerate(zip(config_names, fold_accuracies)):
        offset = (i - len(config_names) / 2 + 0.5) * width
        ax1.bar(x + offset, accuracies, width, label=config_name, alpha=0.8)
    
    ax1.set_xlabel('Fold Number', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Fold-by-Fold Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot: Distribution of fold performance
    ax2 = axes[0, 1]
    bp = ax2.boxplot(fold_accuracies, labels=config_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(config_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Distribution of Fold Performance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Line plot: Fold performance trends
    ax3 = axes[1, 0]
    for i, (config_name, accuracies) in enumerate(zip(config_names, fold_accuracies)):
        ax3.plot(range(1, 6), accuracies, marker='o', label=config_name, linewidth=2, markersize=8)
    
    ax3.set_xlabel('Fold Number', fontsize=11)
    ax3.set_ylabel('Accuracy', fontsize=11)
    ax3.set_title('Fold Performance Trends', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(1, 6))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparison: Mean and std across configurations
    ax4 = axes[1, 1]
    means = [np.mean(acc) for acc in fold_accuracies]
    stds = [np.std(acc) for acc in fold_accuracies]
    
    x_pos = np.arange(len(config_names))
    bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color=colors[:len(config_names)])
    ax4.set_xlabel('Configuration', fontsize=11)
    ax4.set_ylabel('Mean Accuracy ± Std', fontsize=11)
    ax4.set_title('Mean Performance with Variability', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(config_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax4.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CV visualization to: {save_path}")


def save_comparison_table(df: pd.DataFrame, save_dir: Path) -> None:
    """
    Save comparison table in both CSV and Markdown formats
    
    Args:
        df: DataFrame with comparison data
        save_dir: Directory to save files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = save_dir / "comprehensive_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison table (CSV) to: {csv_path}")
    
    # Save Markdown
    md_path = save_dir / "comprehensive_comparison.md"
    
    with open(md_path, 'w') as f:
        f.write("# Comprehensive Classification Results Comparison\n\n")
        f.write("This table compares train/test and cross-validation results across all configurations.\n\n")
        
        # Create markdown table
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    values.append("N/A")
                elif isinstance(val, (int, float)):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            f.write("| " + " | ".join(values) + " |\n")
        
        f.write("\n## Legend\n\n")
        f.write("- **TT_**: Train/Test split metrics\n")
        f.write("- **CV_**: Cross-validation metrics\n")
        f.write("- **Overfitting_Diff**: Difference between train/test and CV accuracy (positive = potential overfitting)\n")
        f.write("- **CV_Consistency**: Coefficient of variation (std/mean) as percentage\n")
        f.write("- **★**: Best performing configuration\n")
    
    print(f"Saved comparison table (Markdown) to: {md_path}")


def generate_cv_analysis_text(cv_analysis: Dict, save_path: Path) -> None:
    """
    Generate text summary of CV fold analysis
    
    Args:
        cv_analysis: Dictionary with CV analysis data
        save_path: Path to save text file
    """
    save_path = Path(save_path)
    
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Cross-Validation Fold Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for config_name, analysis in sorted(cv_analysis.items()):
            f.write(f"\n## {config_name.upper()}\n")
            f.write("-" * 80 + "\n\n")
            
            if not analysis['folds']:
                f.write("No fold data available.\n\n")
                continue
            
            # Best and worst folds
            if analysis['best_fold']:
                best = analysis['best_fold']
                f.write(f"**Best Fold**: Fold {best['fold']}\n")
                f.write(f"  - Accuracy: {best['accuracy']:.4f}\n")
                f.write(f"  - Macro F1: {best['f1']:.4f}\n")
                f.write(f"  - Macro AUC: {best['auc']:.4f}\n\n")
            
            if analysis['worst_fold']:
                worst = analysis['worst_fold']
                f.write(f"**Worst Fold**: Fold {worst['fold']}\n")
                f.write(f"  - Accuracy: {worst['accuracy']:.4f}\n")
                f.write(f"  - Macro F1: {worst['f1']:.4f}\n")
                f.write(f"  - Macro AUC: {worst['auc']:.4f}\n\n")
            
            # Variance metrics
            if analysis['variance']:
                var = analysis['variance']
                f.write("**Variance Metrics**:\n")
                for metric_name in ['accuracy', 'f1', 'auc']:
                    if metric_name in var:
                        f.write(f"  - {metric_name.capitalize()}:\n")
                        f.write(f"    - Std: {var[metric_name]['std']:.4f}\n")
                        f.write(f"    - Range: {var[metric_name]['range']:.4f}\n")
                        f.write(f"    - CV (std/mean): {var[metric_name]['cv']:.4f}\n")
                f.write("\n")
            
            # Outliers
            if analysis['outliers']:
                f.write(f"**Outliers** (folds >2 std from mean): {len(analysis['outliers'])} found\n")
                for outlier in analysis['outliers']:
                    f.write(f"  - Fold {outlier['fold']}: Accuracy = {outlier['accuracy']:.4f} "
                           f"(deviation: {outlier['deviation']:+.4f})\n")
                f.write("\n")
            else:
                f.write("**Outliers**: None detected\n\n")
            
            # Fold-by-fold breakdown
            f.write("**Fold-by-Fold Performance**:\n")
            f.write("| Fold | Accuracy | Macro F1 | Macro AUC |\n")
            f.write("|------|----------|----------|-----------|\n")
            for fold_data in sorted(analysis['folds'], key=lambda x: x['fold']):
                f.write(f"| {fold_data['fold']} | {fold_data['accuracy']:.4f} | "
                       f"{fold_data['f1']:.4f} | {fold_data['auc']:.4f} |\n")
            f.write("\n")
        
        # Summary comparison
        f.write("\n" + "=" * 80 + "\n")
        f.write("Summary Comparison\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("| Configuration | Mean Accuracy | Std | Consistency (CV) |\n")
        f.write("|---------------|---------------|-----|------------------|\n")
        
        for config_name, analysis in sorted(cv_analysis.items()):
            if analysis['fold_accuracy']:
                mean_acc = np.mean(analysis['fold_accuracy'])
                std_acc = np.std(analysis['fold_accuracy'])
                cv_coef = std_acc / mean_acc if mean_acc > 0 else 0
                f.write(f"| {config_name} | {mean_acc:.4f} | {std_acc:.4f} | {cv_coef:.4f} |\n")
    
    print(f"Saved CV analysis text to: {save_path}")


def main():
    """Main function to orchestrate comparison generation"""
    import os
    import sys
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    results_dir = project_root / "project" / "data" / "results" / "classification_results"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("=" * 80)
    print("Comprehensive Comparison Table and CV Fold Analysis")
    print("=" * 80)
    print(f"\nLoading results from: {results_dir}\n")
    
    # Load all results
    all_results = load_all_results(results_dir)
    
    print(f"Found {len(all_results['train_test'])} train/test configurations")
    print(f"Found {len(all_results['cv'])} CV configurations\n")
    
    # Generate comparison table
    print("Generating comparison table...")
    comparison_df = generate_comparison_table(all_results)
    
    # Save comparison table
    save_comparison_table(comparison_df, results_dir)
    
    # Analyze CV folds
    if all_results['cv']:
        print("\nAnalyzing CV folds...")
        cv_analysis = analyze_cv_folds(all_results['cv'])
        
        # Generate text summary
        cv_text_path = results_dir / "cv_fold_analysis.txt"
        generate_cv_analysis_text(cv_analysis, cv_text_path)
        
        # Generate visualizations
        cv_viz_path = results_dir / "cv_fold_analysis.png"
        create_cv_visualizations(cv_analysis, cv_viz_path)
    else:
        print("\nNo CV results found, skipping CV fold analysis")
    
    print("\n" + "=" * 80)
    print("Comparison generation complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - comprehensive_comparison.csv")
    print(f"  - comprehensive_comparison.md")
    if all_results['cv']:
        print(f"  - cv_fold_analysis.txt")
        print(f"  - cv_fold_analysis.png")
    print(f"\nAll files saved to: {results_dir}")


if __name__ == "__main__":
    main()
