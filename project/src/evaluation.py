#!/usr/bin/env python3
"""
Evaluation Module
Comprehensive evaluation metrics, visualizations, and reporting
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve,
                            precision_recall_curve, auc, confusion_matrix,
                            classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class Evaluator:
    """
    Comprehensive evaluation for classification results
    """
    
    def __init__(self, output_dir: str = 'data/results'):
        """
        Initialize evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_pred_proba: Optional[np.ndarray] = None,
                       class_names: Optional[List[str]] = None) -> Dict:
        """
        Compute comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            class_names: Names of classes (optional)
        
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
        metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro'))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted'))
        
        # Per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = []
        recall_per_class = []
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        for cls in classes:
            cls_mask = (y_true == cls)
            pred_mask = (y_pred == cls)
            
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
        
        metrics['f1_per_class'] = {str(cls): float(f1) for cls, f1 in zip(classes, f1_per_class)}
        metrics['precision_per_class'] = {str(cls): float(p) for cls, p in zip(classes, precision_per_class)}
        metrics['recall_per_class'] = {str(cls): float(r) for cls, r in zip(classes, recall_per_class)}
        
        # AUC metrics
        if y_pred_proba is not None:
            n_classes = len(classes)
            
            if n_classes == 2:
                # Binary classification
                try:
                    auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
                    metrics['auc'] = float(auc_score)
                except:
                    pass
            else:
                # Multi-class: compute macro-averaged AUC
                try:
                    # One-vs-rest approach
                    auc_scores = []
                    for i, cls in enumerate(classes):
                        y_true_binary = (y_true == cls).astype(int)
                        if len(np.unique(y_true_binary)) > 1:  # Need both classes
                            y_proba_binary = y_pred_proba[:, i]
                            try:
                                auc_cls = roc_auc_score(y_true_binary, y_proba_binary)
                                auc_scores.append(auc_cls)
                            except:
                                pass
                    
                    if auc_scores:
                        metrics['auc_macro'] = float(np.mean(auc_scores))
                        metrics['auc_per_class'] = {str(cls): float(auc) for cls, auc in zip(classes[:len(auc_scores)], auc_scores)}
                except Exception as e:
                    metrics['auc_error'] = str(e)
        
        # Classification report
        if class_names:
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            metrics['classification_report'] = report
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              title: str = 'Confusion Matrix',
                              save_path: Optional[str] = None,
                              normalize: bool = True) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            save_path: Path to save figure
            normalize: Whether to normalize the matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            label = 'Normalized'
        else:
            fmt = 'd'
            label = 'Count'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names if class_names else 'auto',
                   yticklabels=class_names if class_names else 'auto')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       title: str = 'ROC Curves',
                       save_path: Optional[str] = None) -> Dict:
        """
        Plot ROC curves
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Dictionary with AUC scores
        """
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        plt.figure(figsize=(10, 8))
        
        auc_scores = {}
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            auc_score = auc(fpr, tpr)
            auc_scores['class_1'] = float(auc_score)
            
            plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
        else:
            # Multi-class: one-vs-rest
            for i, cls in enumerate(classes):
                y_true_binary = (y_true == cls).astype(int)
                if len(np.unique(y_true_binary)) > 1:
                    y_proba_binary = y_pred_proba[:, i]
                    fpr, tpr, _ = roc_curve(y_true_binary, y_proba_binary)
                    auc_score = auc(fpr, tpr)
                    auc_scores[str(cls)] = float(auc_score)
                    
                    cls_name = class_names[i] if class_names and i < len(class_names) else str(cls)
                    plt.plot(fpr, tpr, label=f'{cls_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROC curves to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return auc_scores
    
    def plot_pr_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      class_names: Optional[List[str]] = None,
                      title: str = 'Precision-Recall Curves',
                      save_path: Optional[str] = None) -> Dict:
        """
        Plot Precision-Recall curves
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            Dictionary with AUPR scores
        """
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        plt.figure(figsize=(10, 8))
        
        aupr_scores = {}
        
        if n_classes == 2:
            # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
            aupr_score = auc(recall, precision)
            aupr_scores['class_1'] = float(aupr_score)
            
            plt.plot(recall, precision, label=f'PR (AUPR = {aupr_score:.3f})')
        else:
            # Multi-class: one-vs-rest
            for i, cls in enumerate(classes):
                y_true_binary = (y_true == cls).astype(int)
                if len(np.unique(y_true_binary)) > 1:
                    y_proba_binary = y_pred_proba[:, i]
                    precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_binary)
                    aupr_score = auc(recall, precision)
                    aupr_scores[str(cls)] = float(aupr_score)
                    
                    cls_name = class_names[i] if class_names and i < len(class_names) else str(cls)
                    plt.plot(recall, precision, label=f'{cls_name} (AUPR = {aupr_score:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved PR curves to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return aupr_scores
    
    def plot_feature_importance(self, importances: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               top_n: int = 20,
                               title: str = 'Feature Importance',
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance
        
        Args:
            importances: Feature importance values
            feature_names: Names of features
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure
        """
        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        
        top_importances = importances[indices]
        if feature_names:
            top_names = [feature_names[i] for i in indices]
        else:
            top_names = [f'Feature {i}' for i in indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_importances)), top_importances)
        plt.yticks(range(len(top_importances)), top_names)
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, results: Dict, experiment_name: str = 'experiment') -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Dictionary with evaluation results
            experiment_name: Name of the experiment
        
        Returns:
            Dictionary with full report
        """
        report = {
            'experiment_name': experiment_name,
            'metrics': results.get('metrics', {}),
            'confusion_matrix': results.get('confusion_matrix', []),
            'n_samples': results.get('n_samples', 0),
            'n_classes': results.get('n_classes', 0),
            'feature_type': results.get('feature_type', 'unknown'),
            'classifier_type': results.get('classifier_type', 'unknown')
        }
        
        # Save report
        report_file = self.output_dir / f'{experiment_name}_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Saved evaluation report to: {report_file}")
        
        return report
    
    def compare_experiments(self, experiment_results: List[Dict],
                          save_path: Optional[str] = None) -> None:
        """
        Compare multiple experiments
        
        Args:
            experiment_results: List of experiment result dictionaries
            save_path: Path to save comparison plot
        """
        # Extract metrics for comparison
        experiments = []
        accuracies = []
        f1_macros = []
        auc_macros = []
        
        for exp in experiment_results:
            experiments.append(exp.get('name', 'Unknown'))
            metrics = exp.get('metrics', {})
            accuracies.append(metrics.get('accuracy', 0))
            f1_macros.append(metrics.get('f1_macro', 0))
            auc_macros.append(metrics.get('auc_macro', metrics.get('auc', 0)))
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x = np.arange(len(experiments))
        width = 0.35
        
        axes[0].bar(x, accuracies, width, label='Accuracy')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(experiments, rotation=45, ha='right')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        
        axes[1].bar(x, f1_macros, width, label='F1 Macro', color='orange')
        axes[1].set_ylabel('Score')
        axes[1].set_title('F1 Macro Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(experiments, rotation=45, ha='right')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        axes[2].bar(x, auc_macros, width, label='AUC Macro', color='green')
        axes[2].set_ylabel('Score')
        axes[2].set_title('AUC Macro Comparison')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(experiments, rotation=45, ha='right')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    print("Evaluation module - use from run_analysis.py for full pipeline")

