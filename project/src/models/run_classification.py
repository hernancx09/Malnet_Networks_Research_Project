#!/usr/bin/env python3
"""
Main Classification Script
Orchestrates the complete classification pipeline
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.classification import ClassificationPipeline, train_test_split_graphs, cross_validate
from src.models.feature_extraction import load_features_from_files
from src.models.evaluation import (
    compute_metrics,
    generate_confusion_matrix,
    save_evaluation_results,
    print_evaluation_summary,
    generate_roc_curves,
    generate_pr_curves,
    extract_feature_importances,
    analyze_misclassifications,
    detailed_cv_analysis,
    save_cv_analysis
)


def main():
    """Main classification pipeline"""
    parser = argparse.ArgumentParser(
        description='Malware Family Classification using DGDV/GCM Features'
    )
    
    # Feature selection
    parser.add_argument(
        '--features',
        type=str,
        choices=['gcm', 'coarse', 'both'],
        required=True,
        help='Feature type to use: gcm, coarse, or both'
    )
    
    # Classifier selection
    parser.add_argument(
        '--classifier',
        type=str,
        choices=['rf', 'svm', 'all'],
        default='all',
        help='Classifier to use: rf, svm, or all (default: all)'
    )
    
    # Train/test split method
    parser.add_argument(
        '--split',
        type=str,
        choices=['train_test', 'cv'],
        default='train_test',
        help='Split method: train_test (70/15/15) or cv (5-fold cross-validation)'
    )
    
    # Input files (combined files)
    parser.add_argument(
        '--dgdv_file',
        type=str,
        default=None,
        help='Path to combined DGDV file (for coarse features). Mutually exclusive with --dgdv_dir'
    )
    
    parser.add_argument(
        '--gcm_file',
        type=str,
        default=None,
        help='Path to combined GCM file. Mutually exclusive with --gcm_dir'
    )
    
    # Input directories (individual files)
    parser.add_argument(
        '--dgdv_dir',
        type=str,
        default=None,
        help='Directory with individual DGDV files (graph_XXXXX.pkl). Mutually exclusive with --dgdv_file'
    )
    
    parser.add_argument(
        '--gcm_dir',
        type=str,
        default=None,
        help='Directory with individual GCM files (graph_XXXXX.pkl). Mutually exclusive with --gcm_file'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/malnet_tiny',
        help='Path to MalNet-Tiny dataset directory (for labels)'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/results',
        help='Output directory for results and models'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Change to project root (already set above)
    os.chdir(project_root)
    
    print("="*60)
    print("Malware Family Classification Pipeline")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Features:      {args.features}")
    print(f"  Classifier(s): {args.classifier}")
    print(f"  Split method:  {args.split}")
    print(f"  Random state:  {args.random_state}")
    print("="*60)
    
    # Set up output directories
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    results_dir = output_dir / "classification_results"
    confusion_dir = results_dir / "confusion_matrices"
    roc_dir = results_dir / "roc_curves"
    pr_dir = results_dir / "pr_curves"
    importance_dir = results_dir / "feature_importances"
    misclassification_dir = results_dir / "misclassifications"
    cv_analysis_dir = results_dir / "cv_analysis"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    roc_dir.mkdir(parents=True, exist_ok=True)
    pr_dir.mkdir(parents=True, exist_ok=True)
    importance_dir.mkdir(parents=True, exist_ok=True)
    misclassification_dir.mkdir(parents=True, exist_ok=True)
    cv_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which files/directories to load
    dgdv_file = None
    gcm_file = None
    dgdv_dir = None
    gcm_dir = None
    
    if args.features in ['coarse', 'both']:
        if args.dgdv_dir:
            dgdv_dir = args.dgdv_dir
            if not Path(dgdv_dir).exists():
                print(f"\n[ERROR] DGDV directory not found: {dgdv_dir}")
                return
        elif args.dgdv_file:
            dgdv_file = args.dgdv_file
            if not Path(dgdv_file).exists():
                print(f"\n[ERROR] DGDV file not found: {dgdv_file}")
                return
        else:
            # Default to individual files directory
            dgdv_dir = 'data/DGDVs/individual'
            if not Path(dgdv_dir).exists():
                print(f"\n[ERROR] DGDV directory not found: {dgdv_dir}")
                print("  Please specify --dgdv_file or --dgdv_dir")
                return
    
    if args.features in ['gcm', 'both']:
        if args.gcm_dir:
            gcm_dir = args.gcm_dir
            if not Path(gcm_dir).exists():
                print(f"\n[ERROR] GCM directory not found: {gcm_dir}")
                return
        elif args.gcm_file:
            gcm_file = args.gcm_file
            if not Path(gcm_file).exists():
                print(f"\n[ERROR] GCM file not found: {gcm_file}")
                return
        else:
            # Default to individual files directory
            gcm_dir = 'data/GCMs/individual'
            if not Path(gcm_dir).exists():
                print(f"\n[ERROR] GCM directory not found: {gcm_dir}")
                print("  Please specify --gcm_file or --gcm_dir")
                return
    
    # Load features and labels
    print(f"\n[1/5] Loading features and labels...")
    try:
        features, labels, label_mapping, feature_type = load_features_from_files(
            dgdv_file=dgdv_file,
            gcm_file=gcm_file,
            dgdv_dir=dgdv_dir,
            gcm_dir=gcm_dir,
            data_dir=args.data_dir
        )
        print(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")
        print(f"  Feature type: {feature_type}")
        print(f"  Number of classes: {len(label_mapping)}")
    except Exception as e:
        print(f"\n[ERROR] Failed to load features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Determine which classifiers to run
    classifiers_to_run = []
    if args.classifier == 'all':
        classifiers_to_run = ['rf', 'svm']
    else:
        classifiers_to_run = [args.classifier]
    
    # Store all results for comparison
    all_results = {}
    
    # Train and evaluate each classifier
    for classifier_name in classifiers_to_run:
        print(f"\n[2/5] Training {classifier_name.upper()} classifier...")
        
        # Initialize pipeline
        pipeline = ClassificationPipeline(
            models_dir=str(models_dir),
            random_state=args.random_state
        )
        pipeline.label_encoder = label_mapping  # Store label mapping
        
        if args.split == 'train_test':
            # Train/test split (70/15/15, but we'll do 70/30 then split 30 into 15/15)
            print(f"\n[3/5] Train/Test Split (70/15/15)...")
            X_train, X_temp, y_train, y_temp = train_test_split_graphs(
                features, labels,
                test_size=0.3,
                random_state=args.random_state,
                stratify=True
            )
            
            # Split temp into validation and test
            X_val, X_test, y_val, y_test = train_test_split_graphs(
                X_temp, y_temp,
                test_size=0.5,
                random_state=args.random_state,
                stratify=True
            )
            
            print(f"  Train: {X_train.shape[0]} samples")
            print(f"  Validation: {X_val.shape[0]} samples")
            print(f"  Test: {X_test.shape[0]} samples")
            
            # Train on training set
            pipeline.fit(X_train, y_train, classifier_name=classifier_name)
            
            # Evaluate on test set
            print(f"\n[4/5] Evaluating on test set...")
            y_pred = pipeline.predict(X_test, classifier_name=classifier_name)
            y_proba = pipeline.predict_proba(X_test, classifier_name=classifier_name)
            
            # Compute metrics
            metrics = compute_metrics(
                y_test, y_pred, y_proba,
                labels=[label_mapping[i] for i in sorted(label_mapping.keys())]
            )
            
            # Generate confusion matrix
            label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
            cm = generate_confusion_matrix(
                y_test, y_pred, label_names,
                save_path=confusion_dir / f"{classifier_name}_{feature_type}_cm.csv",
                save_plot=True
            )
            
            # Generate ROC curves
            print(f"  Generating ROC curves...")
            roc_results = generate_roc_curves(
                y_test, y_proba, label_names,
                save_path=roc_dir / f"{classifier_name}_{feature_type}_roc.png"
            )
            if roc_results:
                metrics['roc_curves'] = roc_results
            
            # Generate PR curves
            print(f"  Generating PR curves...")
            pr_results = generate_pr_curves(
                y_test, y_proba, label_names,
                save_path=pr_dir / f"{classifier_name}_{feature_type}_pr.png"
            )
            if pr_results:
                metrics['pr_curves'] = pr_results
                metrics['macro_aupr'] = pr_results.get('macro_aupr')
            
            # Extract feature importances (if Random Forest)
            if classifier_name == 'rf':
                print(f"  Extracting feature importances...")
                importance_results = extract_feature_importances(
                    pipeline, classifier_name='rf', feature_type=feature_type,
                    save_path=importance_dir / f"{classifier_name}_{feature_type}_importances"
                )
                if importance_results:
                    metrics['feature_importances'] = importance_results
            
            # Analyze misclassifications
            print(f"  Analyzing misclassifications...")
            misclassification_results = analyze_misclassifications(
                y_test, y_pred, label_names,
                save_path=misclassification_dir / f"{classifier_name}_{feature_type}_analysis"
            )
            if misclassification_results:
                metrics['misclassifications'] = misclassification_results
            
            # Save model
            pipeline.save_model(classifier_name, feature_type, save_dir=models_dir)
            
            # Store results
            all_results[f"{classifier_name}_{feature_type}"] = {
                'metrics': metrics,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'feature_type': feature_type,
                'classifier': classifier_name
            }
            
            # Print summary
            print(f"\n{classifier_name.upper()} - {feature_type.upper()} Results:")
            print_evaluation_summary(metrics)
            
            # Save results
            save_evaluation_results(
                metrics,
                results_dir / f"{classifier_name}_{feature_type}_results",
                format='both'
            )
        
        elif args.split == 'cv':
            # Cross-validation
            print(f"\n[3/5] Cross-Validation (5-fold)...")
            
            # Perform detailed CV analysis
            label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
            detailed_cv = detailed_cv_analysis(
                pipeline, features, labels,
                classifier_name=classifier_name,
                cv=5,
                labels=label_names
            )
            
            # Save CV analysis
            save_cv_analysis(
                detailed_cv,
                save_path=cv_analysis_dir / f"{classifier_name}_{feature_type}_cv",
                create_plot=True
            )
            
            print(f"  Cross-Validation Results:")
            for metric_name, stats in detailed_cv['consistency'].items():
                print(f"    {metric_name}: {stats['mean']:.4f} (+/- {stats['std']:.4f})")
            
            # Train on full dataset for final model
            pipeline.fit(features, labels, classifier_name=classifier_name)
            
            # Also evaluate on full dataset for confusion matrix and curves
            y_pred = pipeline.predict(features, classifier_name=classifier_name)
            y_proba = pipeline.predict_proba(features, classifier_name=classifier_name)
            
            # Compute metrics
            metrics = compute_metrics(
                labels, y_pred, y_proba,
                labels=label_names
            )
            
            # Add CV results to metrics
            metrics['cv_results'] = detailed_cv
            
            # Generate confusion matrix
            cm = generate_confusion_matrix(
                labels, y_pred, label_names,
                save_path=confusion_dir / f"{classifier_name}_{feature_type}_cv_cm.csv",
                save_plot=True
            )
            
            # Generate ROC curves
            print(f"  Generating ROC curves...")
            roc_results = generate_roc_curves(
                labels, y_proba, label_names,
                save_path=roc_dir / f"{classifier_name}_{feature_type}_cv_roc.png"
            )
            if roc_results:
                metrics['roc_curves'] = roc_results
            
            # Generate PR curves
            print(f"  Generating PR curves...")
            pr_results = generate_pr_curves(
                labels, y_proba, label_names,
                save_path=pr_dir / f"{classifier_name}_{feature_type}_cv_pr.png"
            )
            if pr_results:
                metrics['pr_curves'] = pr_results
                metrics['macro_aupr'] = pr_results.get('macro_aupr')
            
            # Extract feature importances (if Random Forest)
            if classifier_name == 'rf':
                print(f"  Extracting feature importances...")
                importance_results = extract_feature_importances(
                    pipeline, classifier_name='rf', feature_type=feature_type,
                    save_path=importance_dir / f"{classifier_name}_{feature_type}_cv_importances"
                )
                if importance_results:
                    metrics['feature_importances'] = importance_results
            
            # Analyze misclassifications
            print(f"  Analyzing misclassifications...")
            misclassification_results = analyze_misclassifications(
                labels, y_pred, label_names,
                save_path=misclassification_dir / f"{classifier_name}_{feature_type}_cv_analysis"
            )
            if misclassification_results:
                metrics['misclassifications'] = misclassification_results
            
            # Save model
            pipeline.save_model(classifier_name, feature_type, save_dir=models_dir)
            
            # Store results
            all_results[f"{classifier_name}_{feature_type}"] = {
                'metrics': metrics,
                'n_samples': len(features),
                'feature_type': feature_type,
                'classifier': classifier_name
            }
            
            # Print summary
            print(f"\n{classifier_name.upper()} - {feature_type.upper()} Results:")
            print_evaluation_summary(metrics)
            
            # Save results
            save_evaluation_results(
                metrics,
                results_dir / f"{classifier_name}_{feature_type}_cv_results",
                format='both'
            )
    
    # Generate comparison report
    print(f"\n[5/5] Generating comparison report...")
    comparison_path = results_dir / "comparison_report.txt"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Classification Results Comparison\n")
        f.write("="*60 + "\n\n")
        
        for result_name, result_data in all_results.items():
            f.write(f"{result_name.upper()}:\n")
            f.write(f"  Accuracy: {result_data['metrics']['accuracy']:.4f}\n")
            f.write(f"  Macro F1: {result_data['metrics']['macro_f1']:.4f}\n")
            f.write(f"  Weighted F1: {result_data['metrics']['weighted_f1']:.4f}\n")
            if 'macro_auc' in result_data['metrics'] and result_data['metrics']['macro_auc'] is not None:
                f.write(f"  Macro AUC (AUROC): {result_data['metrics']['macro_auc']:.4f}\n")
            if 'macro_aupr' in result_data['metrics'] and result_data['metrics']['macro_aupr'] is not None:
                f.write(f"  Macro AUPR: {result_data['metrics']['macro_aupr']:.4f}\n")
            
            # Feature importances (if available)
            if 'feature_importances' in result_data['metrics']:
                f.write(f"\n  Top 5 Feature Importances:\n")
                top_features = result_data['metrics']['feature_importances'].get('top_features', [])[:5]
                for feat in top_features:
                    f.write(f"    {feat['feature']}: {feat['importance']:.4f}\n")
            
            # Misclassification patterns (if available)
            if 'misclassifications' in result_data['metrics']:
                misclass = result_data['metrics']['misclassifications']
                f.write(f"\n  Misclassification Analysis:\n")
                f.write(f"    Total misclassified: {misclass.get('misclassified_samples', 0)} / {misclass.get('total_samples', 0)}\n")
                f.write(f"    Misclassification rate: {misclass.get('misclassification_rate', 0):.2%}\n")
                top_patterns = misclass.get('top_patterns', [])[:3]
                if top_patterns:
                    f.write(f"    Top 3 error patterns:\n")
                    for pattern in top_patterns:
                        f.write(f"      {pattern['pattern']}: {pattern['count']} times\n")
            
            # CV consistency (if available)
            if 'cv_results' in result_data['metrics']:
                cv = result_data['metrics']['cv_results']
                f.write(f"\n  Cross-Validation Consistency:\n")
                for metric_name, stats in cv.get('consistency', {}).items():
                    f.write(f"    {metric_name}: {stats['mean']:.4f} (+/- {stats['std']:.4f})\n")
                    f.write(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            
            f.write("\n")
        
        f.write("="*60 + "\n")
    
    print(f"  Saved comparison report to: {comparison_path}")
    
    print("\n" + "="*60)
    print("Classification pipeline completed successfully!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print(f"Models saved to: {models_dir}")


if __name__ == "__main__":
    main()

