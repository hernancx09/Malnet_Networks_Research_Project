#!/usr/bin/env python3
"""
Evaluation Module
Comprehensive evaluation metrics, confusion matrices, and result saving
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List] = None
) -> Dict:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities of shape (n_samples, n_classes)
        labels: List of label names for per-class metrics
    
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted'))
    metrics['micro_f1'] = float(f1_score(y_true, y_pred, average='micro'))
    
    metrics['macro_precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['weighted_precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    
    metrics['macro_recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['weighted_recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # Per-class metrics
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=unique_labels, zero_division=0)
    
    metrics['per_class'] = {}
    for i, label in enumerate(unique_labels):
        label_name = labels[int(label)] if labels is not None else str(int(label))
        metrics['per_class'][label_name] = {
            'f1': float(per_class_f1[i]),
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i])
        }
    
    # AUC metrics (if probabilities available)
    if y_proba is not None:
        try:
            # Multi-class AUC
            if len(np.unique(y_true)) > 2:
                # Multi-class: use one-vs-rest
                metrics['macro_auc'] = float(roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr'))
                metrics['weighted_auc'] = float(roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr'))
                
                # Per-class AUC
                per_class_auc = {}
                for i, label in enumerate(unique_labels):
                    label_name = labels[int(label)] if labels is not None else str(int(label))
                    # One-vs-rest AUC for this class
                    y_true_binary = (y_true == label).astype(int)
                    if len(np.unique(y_true_binary)) > 1:  # Check if both classes present
                        auc = float(roc_auc_score(y_true_binary, y_proba[:, int(label)]))
                        per_class_auc[label_name] = auc
                    else:
                        per_class_auc[label_name] = None
                
                metrics['per_class_auc'] = per_class_auc
            else:
                # Binary classification
                metrics['auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except Exception as e:
            print(f"Warning: Could not compute AUC metrics: {e}")
            metrics['macro_auc'] = None
            metrics['weighted_auc'] = None
    
    # Classification report
    if labels is not None:
        target_names = [labels[int(i)] for i in unique_labels]
    else:
        target_names = [str(int(i)) for i in unique_labels]
    
    metrics['classification_report'] = classification_report(
        y_true, y_pred, labels=unique_labels, target_names=target_names, output_dict=True
    )
    
    return metrics


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: Optional[Path] = None,
    save_plot: bool = False
) -> np.ndarray:
    """
    Generate confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        save_path: Path to save confusion matrix CSV
        save_plot: Whether to save visualization plot
    
    Returns:
        Confusion matrix array
    """
    # Get unique labels
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    label_names = [labels[int(i)] for i in unique_labels]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Save as CSV
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([''] + label_names)
            # Rows
            for i, label_name in enumerate(label_names):
                writer.writerow([label_name] + cm[i].tolist())
        
        print(f"Saved confusion matrix to: {save_path}")
        
        # Save visualization if requested
        if save_plot:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                plot_path = save_path.with_suffix('.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved confusion matrix plot to: {plot_path}")
            except ImportError:
                print("Warning: matplotlib/seaborn not available, skipping plot generation")
    
    return cm


def save_evaluation_results(
    results: Dict,
    save_path: Path,
    format: str = "json"
) -> None:
    """
    Save evaluation results to file
    
    Args:
        results: Dictionary with evaluation results
        save_path: Path to save file
        format: Format to save ('json', 'csv', or 'both')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format in ['json', 'both']:
        json_path = save_path.with_suffix('.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved evaluation results (JSON) to: {json_path}")
    
    if format in ['csv', 'both']:
        csv_path = save_path.with_suffix('.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Overall metrics
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Accuracy', results.get('accuracy', '')])
            writer.writerow(['Macro F1', results.get('macro_f1', '')])
            writer.writerow(['Weighted F1', results.get('weighted_f1', '')])
            writer.writerow(['Macro Precision', results.get('macro_precision', '')])
            writer.writerow(['Macro Recall', results.get('macro_recall', '')])
            
            if 'macro_auc' in results and results['macro_auc'] is not None:
                writer.writerow(['Macro AUC', results.get('macro_auc', '')])
            
            writer.writerow([])
            
            # Per-class metrics
            if 'per_class' in results:
                writer.writerow(['Class', 'F1', 'Precision', 'Recall'])
                for class_name, class_metrics in results['per_class'].items():
                    writer.writerow([
                        class_name,
                        class_metrics.get('f1', ''),
                        class_metrics.get('precision', ''),
                        class_metrics.get('recall', '')
                    ])
        
        print(f"Saved evaluation results (CSV) to: {csv_path}")


def print_evaluation_summary(results: Dict) -> None:
    """
    Print formatted summary of evaluation results
    
    Args:
        results: Dictionary with evaluation results
    """
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    
    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  Accuracy:        {results.get('accuracy', 0):.4f}")
    print(f"  Macro F1:        {results.get('macro_f1', 0):.4f}")
    print(f"  Weighted F1:     {results.get('weighted_f1', 0):.4f}")
    print(f"  Macro Precision: {results.get('macro_precision', 0):.4f}")
    print(f"  Macro Recall:    {results.get('macro_recall', 0):.4f}")
    
    if 'macro_auc' in results and results['macro_auc'] is not None:
        print(f"  Macro AUC:       {results.get('macro_auc', 0):.4f}")
    
    # Per-class metrics
    if 'per_class' in results:
        print("\nPer-Class Metrics:")
        print(f"  {'Class':<20} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print("  " + "-"*50)
        for class_name, class_metrics in results['per_class'].items():
            print(f"  {class_name:<20} "
                  f"{class_metrics.get('f1', 0):<10.4f} "
                  f"{class_metrics.get('precision', 0):<10.4f} "
                  f"{class_metrics.get('recall', 0):<10.4f}")
    
    print("="*60 + "\n")


def generate_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: List[str],
    save_path: Optional[Path] = None,
    multi_class: str = 'ovr'
) -> Dict:
    """
    Generate ROC curves for multi-class classification
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities of shape (n_samples, n_classes)
        labels: List of label names
        save_path: Path to save ROC curve plot
        multi_class: Multi-class strategy ('ovr' for one-vs-rest)
    
    Returns:
        Dictionary with AUROC values per class and macro-averaged
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping ROC curve generation")
        return {}
    
    unique_labels = np.unique(y_true)
    n_classes = len(unique_labels)
    
    # Compute ROC curve for each class (one-vs-rest)
    roc_data = {}
    auc_scores = {}
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    for i, label in enumerate(unique_labels):
        label_name = labels[int(label)] if labels is not None else f"Class {int(label)}"
        
        # Binary labels for this class (one-vs-rest)
        y_true_binary = (y_true == label).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:  # Both classes present
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, int(label)])
            auc = roc_auc_score(y_true_binary, y_proba[:, int(label)])
            
            roc_data[label_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
            auc_scores[label_name] = float(auc)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors(i), lw=2,
                    label=f'{label_name} (AUC = {auc:.3f})')
    
    # Compute macro-averaged ROC curve
    all_fpr = np.unique(np.concatenate([roc_data[label]['fpr'] for label in roc_data.keys()]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for label_name in roc_data.keys():
        fpr = roc_data[label_name]['fpr']
        tpr = roc_data[label_name]['tpr']
        # Interpolate to common FPR values
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    
    mean_tpr /= len(roc_data)
    macro_auc = roc_auc_score(y_true, y_proba, average='macro', multi_class=multi_class)
    
    # Plot macro-averaged ROC
    plt.plot(all_fpr, mean_tpr, color='black', linestyle='--', lw=2,
            label=f'Macro-average (AUC = {macro_auc:.3f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (Multi-class)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to: {save_path}")
    
    plt.close()
    
    # Return AUROC values
    result = {
        'per_class_auc': auc_scores,
        'macro_auc': float(macro_auc)
    }
    
    return result


def generate_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: List[str],
    save_path: Optional[Path] = None
) -> Dict:
    """
    Generate Precision-Recall curves for multi-class classification
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities of shape (n_samples, n_classes)
        labels: List of label names
        save_path: Path to save PR curve plot
    
    Returns:
        Dictionary with AUPR values per class and macro-averaged
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping PR curve generation")
        return {}
    
    unique_labels = np.unique(y_true)
    n_classes = len(unique_labels)
    
    # Compute PR curve for each class (one-vs-rest)
    pr_data = {}
    aupr_scores = {}
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    for i, label in enumerate(unique_labels):
        label_name = labels[int(label)] if labels is not None else f"Class {int(label)}"
        
        # Binary labels for this class (one-vs-rest)
        y_true_binary = (y_true == label).astype(int)
        
        if len(np.unique(y_true_binary)) > 1:  # Both classes present
            # Compute PR curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, int(label)])
            aupr = average_precision_score(y_true_binary, y_proba[:, int(label)])
            
            pr_data[label_name] = {'precision': precision, 'recall': recall, 'aupr': aupr}
            aupr_scores[label_name] = float(aupr)
            
            # Plot PR curve
            plt.plot(recall, precision, color=colors(i), lw=2,
                    label=f'{label_name} (AUPR = {aupr:.3f})')
    
    # Compute macro-averaged AUPR
    macro_aupr = average_precision_score(y_true, y_proba, average='macro')
    
    # Plot baseline (random classifier)
    baseline = np.sum([(y_true == label).sum() for label in unique_labels]) / len(y_true)
    plt.axhline(y=baseline, color='gray', linestyle=':', lw=1, label='Baseline (random)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (Multi-class)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curves to: {save_path}")
    
    plt.close()
    
    # Return AUPR values
    result = {
        'per_class_aupr': aupr_scores,
        'macro_aupr': float(macro_aupr)
    }
    
    return result


def extract_feature_importances(
    pipeline,
    classifier_name: str = 'rf',
    feature_type: str = 'gcm',
    top_n: int = 50,
    save_path: Optional[Path] = None
) -> Dict:
    """
    Extract feature importances from Random Forest classifier
    
    Args:
        pipeline: ClassificationPipeline instance with trained classifier
        classifier_name: Name of classifier ('rf' only, SVM doesn't have importances)
        feature_type: Type of features ('gcm', 'coarse', or 'both')
        top_n: Number of top features to return
        save_path: Path to save feature importance CSV and plot
    
    Returns:
        Dictionary with feature importances sorted by importance
    """
    if classifier_name != 'rf':
        print(f"Warning: Feature importances only available for Random Forest, not {classifier_name}")
        return {}
    
    if 'rf' not in pipeline.classifiers:
        print("Warning: Random Forest classifier not found in pipeline")
        return {}
    
    rf_classifier = pipeline.classifiers['rf']
    
    if not hasattr(rf_classifier, 'feature_importances_'):
        print("Warning: Classifier not trained yet, cannot extract feature importances")
        return {}
    
    importances = rf_classifier.feature_importances_
    n_features = len(importances)
    
    # Create feature names based on feature type
    if feature_type == 'gcm':
        # GCM features: correlation indices
        # For n orbits, we have n*(n-1)/2 correlations
        # We can't easily map back to orbit pairs without knowing n_orbits
        # So we'll just use indices
        feature_names = [f'GCM_{i}' for i in range(n_features)]
    elif feature_type == 'coarse':
        # Coarse features: orbit statistics
        # Format: mean_0, std_0, sum_0, min_0, max_0, mean_1, ...
        n_orbits = n_features // 5
        stats = ['mean', 'std', 'sum', 'min', 'max']
        feature_names = []
        for orbit_idx in range(n_orbits):
            for stat in stats:
                feature_names.append(f'Orbit_{orbit_idx}_{stat}')
    else:  # both
        # Combined features - use generic names
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Sort by importance (descending)
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]
    
    # Get top N
    top_importances = sorted_importances[:top_n]
    top_names = sorted_names[:top_n]
    
    # Create result dictionary
    result = {
        'top_features': [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(top_names, top_importances)
        ],
        'total_features': n_features,
        'mean_importance': float(np.mean(importances)),
        'std_importance': float(np.std(importances))
    }
    
    # Save to CSV
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        csv_path = save_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Feature', 'Importance'])
            for rank, (name, imp) in enumerate(zip(top_names, top_importances), 1):
                writer.writerow([rank, name, float(imp)])
        
        print(f"Saved feature importances to: {csv_path}")
        
        # Create visualization
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, max(8, top_n * 0.3)))
            y_pos = np.arange(len(top_names))
            
            plt.barh(y_pos, top_importances, align='center')
            plt.yticks(y_pos, top_names)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()  # Top feature at top
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plot_path = save_path.with_suffix('.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved feature importance plot to: {plot_path}")
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")
    
    return result


def analyze_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: Optional[Path] = None
) -> Dict:
    """
    Analyze misclassification patterns
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        save_path: Path to save misclassification analysis
    
    Returns:
        Dictionary with misclassification patterns and statistics
    """
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    label_names = [labels[int(i)] if labels is not None else f"Class {int(i)}" for i in unique_labels]
    
    # Find misclassified samples
    misclassified_mask = y_true != y_pred
    n_misclassified = misclassified_mask.sum()
    n_total = len(y_true)
    
    # Count misclassification patterns
    misclassification_patterns = {}
    misclassified_indices = {}
    
    for i in range(n_total):
        if misclassified_mask[i]:
            true_label = int(y_true[i])
            pred_label = int(y_pred[i])
            true_name = label_names[list(unique_labels).index(true_label)]
            pred_name = label_names[list(unique_labels).index(pred_label)]
            
            pattern = f"{true_name} → {pred_name}"
            
            if pattern not in misclassification_patterns:
                misclassification_patterns[pattern] = 0
                misclassified_indices[pattern] = []
            
            misclassification_patterns[pattern] += 1
            misclassified_indices[pattern].append(int(i))
    
    # Sort by frequency
    sorted_patterns = sorted(misclassification_patterns.items(), key=lambda x: x[1], reverse=True)
    
    # Create result dictionary
    result = {
        'total_samples': int(n_total),
        'misclassified_samples': int(n_misclassified),
        'accuracy': float(1.0 - (n_misclassified / n_total)),
        'misclassification_rate': float(n_misclassified / n_total),
        'patterns': [
            {
                'pattern': pattern,
                'count': count,
                'percentage': float(count / n_misclassified * 100),
                'sample_indices': misclassified_indices[pattern][:10]  # First 10 indices
            }
            for pattern, count in sorted_patterns
        ],
        'top_patterns': [
            {'pattern': pattern, 'count': count}
            for pattern, count in sorted_patterns[:10]
        ]
    }
    
    # Save to CSV
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        csv_path = save_path.with_suffix('.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Pattern', 'Count', 'Percentage', 'Sample Indices'])
            for rank, (pattern, count) in enumerate(sorted_patterns, 1):
                percentage = count / n_misclassified * 100
                indices_str = ', '.join(map(str, misclassified_indices[pattern][:10]))
                writer.writerow([rank, pattern, count, f"{percentage:.2f}%", indices_str])
        
        print(f"Saved misclassification analysis to: {csv_path}")
        
        # Save to JSON
        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Saved misclassification analysis (JSON) to: {json_path}")
        
        # Create visualization (heatmap of error patterns)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create misclassification matrix (only errors)
            n_classes = len(unique_labels)
            error_matrix = np.zeros((n_classes, n_classes))
            
            for i, true_label in enumerate(unique_labels):
                for j, pred_label in enumerate(unique_labels):
                    if true_label != pred_label:
                        pattern = f"{label_names[i]} → {label_names[j]}"
                        if pattern in misclassification_patterns:
                            error_matrix[i, j] = misclassification_patterns[pattern]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(error_matrix, annot=True, fmt='.0f', cmap='Reds',
                       xticklabels=label_names, yticklabels=label_names,
                       cbar_kws={'label': 'Count'})
            plt.title('Misclassification Patterns', fontsize=14, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            plot_path = save_path.with_suffix('.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved misclassification heatmap to: {plot_path}")
        except ImportError:
            print("Warning: matplotlib/seaborn not available, skipping plot generation")
    
    return result


def detailed_cv_analysis(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str = "rf",
    cv: int = 5,
    labels: Optional[List] = None
) -> Dict:
    """
    Perform detailed cross-validation analysis with per-fold metrics
    
    Args:
        pipeline: ClassificationPipeline instance
        X: Feature matrix
        y: Label array
        classifier_name: Name of classifier ('rf' or 'svm')
        cv: Number of folds
        labels: List of label names
    
    Returns:
        Dictionary with detailed per-fold metrics and consistency analysis
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    
    # Get random_state from pipeline if available
    random_state = getattr(pipeline, 'random_state', 42)
    
    # Scale features
    X_scaled = pipeline.feature_scaler.fit_transform(X)
    
    # Get classifier
    classifier = pipeline.classifiers[classifier_name]
    
    # Perform cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    fold_results = []
    fold_metrics = {
        'accuracy': [],
        'macro_f1': [],
        'weighted_f1': [],
        'macro_auc': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train_fold = X_scaled[train_idx]
        X_val_fold = X_scaled[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Train on fold
        classifier.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        y_pred_fold = classifier.predict(X_val_fold)
        y_proba_fold = classifier.predict_proba(X_val_fold)
        
        # Compute metrics
        acc = accuracy_score(y_val_fold, y_pred_fold)
        macro_f1 = f1_score(y_val_fold, y_pred_fold, average='macro')
        weighted_f1 = f1_score(y_val_fold, y_pred_fold, average='weighted')
        
        # AUC
        try:
            if len(np.unique(y_val_fold)) > 2:
                macro_auc = roc_auc_score(y_val_fold, y_proba_fold, average='macro', multi_class='ovr')
            else:
                macro_auc = roc_auc_score(y_val_fold, y_proba_fold[:, 1])
        except:
            macro_auc = None
        
        fold_metrics['accuracy'].append(float(acc))
        fold_metrics['macro_f1'].append(float(macro_f1))
        fold_metrics['weighted_f1'].append(float(weighted_f1))
        if macro_auc is not None:
            fold_metrics['macro_auc'].append(float(macro_auc))
        
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_auc': float(macro_auc) if macro_auc is not None else None,
            'n_train': len(train_idx),
            'n_val': len(val_idx)
        })
    
    # Compute consistency metrics
    consistency = {}
    for metric_name, values in fold_metrics.items():
        if values:
            consistency[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values))
            }
    
    result = {
        'fold_results': fold_results,
        'consistency': consistency,
        'cv_folds': cv
    }
    
    return result


def save_cv_analysis(
    cv_results: Dict,
    save_path: Path,
    create_plot: bool = True
) -> None:
    """
    Save detailed CV analysis results
    
    Args:
        cv_results: Dictionary from detailed_cv_analysis
        save_path: Path to save CSV file
        create_plot: Whether to create visualization
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save fold-by-fold results to CSV
    csv_path = save_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Fold', 'Accuracy', 'Macro F1', 'Weighted F1', 'Macro AUC', 'N Train', 'N Val'])
        for fold_result in cv_results['fold_results']:
            writer.writerow([
                fold_result['fold'],
                fold_result['accuracy'],
                fold_result['macro_f1'],
                fold_result['weighted_f1'],
                fold_result['macro_auc'] if fold_result['macro_auc'] is not None else '',
                fold_result['n_train'],
                fold_result['n_val']
            ])
        
        writer.writerow([])
        writer.writerow(['Consistency Metrics'])
        writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max', 'Range'])
        for metric_name, stats in cv_results['consistency'].items():
            writer.writerow([
                metric_name,
                stats['mean'],
                stats['std'],
                stats['min'],
                stats['max'],
                stats['range']
            ])
    
    print(f"Saved CV analysis to: {csv_path}")
    
    # Create visualization
    if create_plot:
        try:
            import matplotlib.pyplot as plt
            
            metrics_to_plot = ['accuracy', 'macro_f1', 'macro_auc']
            available_metrics = [m for m in metrics_to_plot if m in cv_results['consistency']]
            
            if available_metrics:
                fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 6))
                if len(available_metrics) == 1:
                    axes = [axes]
                
                for ax, metric_name in zip(axes, available_metrics):
                    values = [fold[metric_name] for fold in cv_results['fold_results'] 
                             if fold[metric_name] is not None]
                    folds = [fold['fold'] for fold in cv_results['fold_results'] 
                            if fold[metric_name] is not None]
                    
                    ax.plot(folds, values, 'o-', linewidth=2, markersize=8)
                    ax.axhline(y=cv_results['consistency'][metric_name]['mean'], 
                              color='r', linestyle='--', label='Mean')
                    ax.fill_between(folds, 
                                   cv_results['consistency'][metric_name]['mean'] - 
                                   cv_results['consistency'][metric_name]['std'],
                                   cv_results['consistency'][metric_name]['mean'] + 
                                   cv_results['consistency'][metric_name]['std'],
                                   alpha=0.2, label='±1 Std')
                    ax.set_xlabel('Fold', fontsize=12)
                    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
                    ax.set_title(f'{metric_name.replace("_", " ").title()} Across Folds', fontsize=12)
                    ax.legend()
                    ax.grid(alpha=0.3)
                    ax.set_xticks(folds)
                
                plt.tight_layout()
                plot_path = save_path.with_suffix('.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved CV analysis plot to: {plot_path}")
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")

