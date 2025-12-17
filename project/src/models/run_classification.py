#!/usr/bin/env python3
"""
Main Classification Script
Orchestrates the complete classification pipeline
"""

import os
import sys
import argparse
import csv
import time
import numpy as np
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
    save_cv_analysis,
    aggregate_metrics_across_runs
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
    
    parser.add_argument(
        '--n_runs',
        type=int,
        default=5,
        help='Number of runs to perform (default: 5). Each run uses different classifier random state.'
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
    print(f"  Number of runs: {args.n_runs}")
    print("="*60)
    
    # Initialize timing
    pipeline_start_time = time.time()
    timing_info = {}
    
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
    load_start = time.time()
    try:
        features, labels, label_mapping, feature_type = load_features_from_files(
            dgdv_file=dgdv_file,
            gcm_file=gcm_file,
            dgdv_dir=dgdv_dir,
            gcm_dir=gcm_dir,
            data_dir=args.data_dir
        )
        load_time = time.time() - load_start
        timing_info['data_loading'] = load_time
        print(f"  Loaded {features.shape[0]} samples with {features.shape[1]} features")
        print(f"  Feature type: {feature_type}")
        print(f"  Number of classes: {len(label_mapping)}")
        print(f"  Loading time: {load_time:.2f} seconds")
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
        
        # Initialize timing for this classifier
        classifier_timing = {}
        
        # Initialize pipeline
        pipeline = ClassificationPipeline(
            models_dir=str(models_dir),
            random_state=args.random_state
        )
        pipeline.label_encoder = label_mapping  # Store label mapping
        
        if args.split == 'train_test':
            # Train/test split (70/15/15, but we'll do 70/30 then split 30 into 15/15)
            print(f"\n[3/5] Train/Test Split (70/15/15)...")
            split_start = time.time()
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
            split_time = time.time() - split_start
            classifier_timing['train_test_split'] = split_time
            print(f"  Train: {X_train.shape[0]} samples")
            print(f"  Validation: {X_val.shape[0]} samples")
            print(f"  Test: {X_test.shape[0]} samples")
            print(f"  Time: {split_time:.2f} seconds")
            
            label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
            
            # Run multiple times
            print(f"\n[4/5] Running {args.n_runs} iterations...")
            all_metrics = []
            all_y_pred = []
            all_y_proba = []
            all_importance_results = []
            last_pipeline = None
            
            training_times = []
            evaluation_times = []
            metrics_times = []
            run_times = []
            
            for run_idx in range(args.n_runs):
                print(f"\n  Run {run_idx + 1}/{args.n_runs}...")
                run_start = time.time()
                
                # Initialize pipeline with different random state for each run
                pipeline = ClassificationPipeline(
                    models_dir=str(models_dir),
                    random_state=args.random_state + run_idx
                )
                pipeline.label_encoder = label_mapping
                last_pipeline = pipeline  # Keep reference to last pipeline
                
                # Train on training set
                train_start = time.time()
                pipeline.fit(X_train, y_train, classifier_name=classifier_name)
                train_time = time.time() - train_start
                training_times.append(train_time)
                print(f"    [Training] Time: {train_time:.2f} seconds")
                
                # Evaluate on test set
                eval_start = time.time()
                y_pred = pipeline.predict(X_test, classifier_name=classifier_name)
                y_proba = pipeline.predict_proba(X_test, classifier_name=classifier_name)
                eval_time = time.time() - eval_start
                evaluation_times.append(eval_time)
                print(f"    [Evaluation] Time: {eval_time:.2f} seconds")
                
                # Compute metrics
                metrics_start = time.time()
                metrics = compute_metrics(
                    y_test, y_pred, y_proba,
                    labels=label_names
                )
                metrics_time = time.time() - metrics_start
                metrics_times.append(metrics_time)
                print(f"    [Metrics] Time: {metrics_time:.2f} seconds")
                
                all_metrics.append(metrics)
                all_y_pred.append(y_pred)
                all_y_proba.append(y_proba)
                
                run_time = time.time() - run_start
                run_times.append(run_time)
                print(f"    Total run time: {run_time:.2f} seconds")
            
            # Store timing info
            classifier_timing['training'] = {
                'total': sum(training_times),
                'per_run': training_times,
                'average': np.mean(training_times)
            }
            classifier_timing['evaluation'] = {
                'total': sum(evaluation_times),
                'per_run': evaluation_times,
                'average': np.mean(evaluation_times)
            }
            classifier_timing['metrics_computation'] = {
                'total': sum(metrics_times),
                'per_run': metrics_times,
                'average': np.mean(metrics_times)
            }
            classifier_timing['runs'] = {
                'total': sum(run_times),
                'per_run': run_times,
                'average': np.mean(run_times)
            }
            
            print(f"\n  Average time per run: {np.mean(run_times):.2f} seconds")
            print(f"  Total time for all runs: {sum(run_times):.2f} seconds")
                
                # Extract feature importances (if Random Forest) - collect from all runs
            if classifier_name == 'rf':
                importance_results = extract_feature_importances(
                    pipeline, classifier_name='rf', feature_type=feature_type,
                    save_path=None  # Don't save individual run importances
                )
                if importance_results:
                    all_importance_results.append(importance_results)
            
            # Aggregate metrics across runs
            print(f"\n  Aggregating results across {args.n_runs} runs...")
            agg_start = time.time()
            aggregated_metrics = aggregate_metrics_across_runs(all_metrics)
            agg_time = time.time() - agg_start
            print(f"  Time: {agg_time:.2f} seconds")
            
            # Generate confusion matrix (averaged across runs)
            print(f"  Generating confusion matrix...")
            cm_start = time.time()
            cm = generate_confusion_matrix(
                y_test, all_y_pred[0], label_names,
                save_path=confusion_dir / f"{classifier_name}_{feature_type}_cm.csv",
                save_plot=True,
                y_pred_list=all_y_pred
            )
            cm_time = time.time() - cm_start
            classifier_timing['plotting'] = classifier_timing.get('plotting', {})
            classifier_timing['plotting']['confusion_matrix'] = cm_time
            print(f"  Time: {cm_time:.2f} seconds")
            
            # Generate ROC curves (with std deviation)
            print(f"  Generating ROC curves...")
            roc_start = time.time()
            roc_results = generate_roc_curves(
                y_test, all_y_proba[0], label_names,
                save_path=roc_dir / f"{classifier_name}_{feature_type}_roc.png",
                y_proba_list=all_y_proba,
                show_std=True
            )
            roc_time = time.time() - roc_start
            classifier_timing['plotting']['roc_curves'] = roc_time
            print(f"  Time: {roc_time:.2f} seconds")
            if roc_results:
                aggregated_metrics['roc_curves'] = roc_results
            
            # Generate PR curves (with std deviation)
            print(f"  Generating PR curves...")
            pr_start = time.time()
            pr_results = generate_pr_curves(
                y_test, all_y_proba[0], label_names,
                save_path=pr_dir / f"{classifier_name}_{feature_type}_pr.png",
                y_proba_list=all_y_proba,
                show_std=True
            )
            pr_time = time.time() - pr_start
            classifier_timing['plotting']['pr_curves'] = pr_time
            print(f"  Time: {pr_time:.2f} seconds")
            if pr_results:
                aggregated_metrics['pr_curves'] = pr_results
                if 'macro_aupr' in pr_results:
                    aggregated_metrics['macro_aupr'] = {
                        'mean': pr_results['macro_aupr'],
                        'std': pr_results.get('macro_aupr_std', 0.0)
                    }
            
            # Average feature importances (if Random Forest)
            if classifier_name == 'rf' and all_importance_results:
                print(f"  Averaging feature importances...")
                imp_start = time.time()
                # Average importances across runs
                if 'top_features' in all_importance_results[0]:
                    # Get all feature names
                    all_feature_names = set()
                    for imp_result in all_importance_results:
                        all_feature_names.update([f['feature'] for f in imp_result.get('top_features', [])])
                    
                    # Average importances
                    avg_importances = {}
                    for feat_name in all_feature_names:
                        values = []
                        for imp_result in all_importance_results:
                            for f in imp_result.get('top_features', []):
                                if f['feature'] == feat_name:
                                    values.append(f['importance'])
                        if values:
                            avg_importances[feat_name] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values))
                            }
                    
                    # Create top features list sorted by mean importance
                    top_features = sorted(
                        [{'feature': k, 'importance': v['mean'], 'std': v['std']} 
                         for k, v in avg_importances.items()],
                        key=lambda x: x['importance'],
                        reverse=True
                    )
                    
                    aggregated_metrics['feature_importances'] = {
                        'top_features': top_features[:50],  # Top 50
                        'n_runs': args.n_runs
                    }
                    
                    # Save averaged feature importances
                    save_path = importance_dir / f"{classifier_name}_{feature_type}_importances"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save CSV
                    csv_path = save_path.with_suffix('.csv')
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Feature', 'Importance (Mean)', 'Std'])
                        for feat in top_features[:50]:
                            writer.writerow([feat['feature'], feat['importance'], feat['std']])
                    
                    # Save plot
                    try:
                        import matplotlib.pyplot as plt
                        
                        top_n = min(50, len(top_features))
                        top_importances = [f['importance'] for f in top_features[:top_n]]
                        top_stds = [f['std'] for f in top_features[:top_n]]
                        top_names = [f['feature'] for f in top_features[:top_n]]
                        y_pos = np.arange(len(top_names))
                        
                        plt.figure(figsize=(10, max(8, top_n * 0.3)))
                        plt.barh(y_pos, top_importances, xerr=top_stds, align='center', capsize=3)
                        plt.yticks(y_pos, top_names)
                        plt.xlabel('Feature Importance (Mean ± Std)', fontsize=12)
                        plt.title(f'Top {top_n} Feature Importances (Random Forest, {args.n_runs} runs)', fontsize=14, fontweight='bold')
                        plt.gca().invert_yaxis()
                        plt.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        
                        plot_path = save_path.with_suffix('.png')
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"  Saved averaged feature importances to: {csv_path} and {plot_path}")
                    except ImportError:
                        print("  Warning: matplotlib not available, skipping feature importance plot")
                imp_time = time.time() - imp_start
                classifier_timing['plotting']['feature_importances'] = imp_time
                print(f"  Time: {imp_time:.2f} seconds")
            
            # Analyze misclassifications (use predictions from first run)
            print(f"  Analyzing misclassifications...")
            misclass_start = time.time()
            misclassification_results = analyze_misclassifications(
                y_test, all_y_pred[0], label_names,
                save_path=misclassification_dir / f"{classifier_name}_{feature_type}_analysis"
            )
            misclass_time = time.time() - misclass_start
            classifier_timing['plotting']['misclassifications'] = misclass_time
            print(f"  Time: {misclass_time:.2f} seconds")
            if misclassification_results:
                aggregated_metrics['misclassifications'] = misclassification_results
            
            # Save model (from last run)
            if last_pipeline:
                last_pipeline.save_model(classifier_name, feature_type, save_dir=models_dir)
            
            # Calculate total plotting time
            classifier_timing['plotting']['total'] = sum([
                classifier_timing['plotting'].get('confusion_matrix', 0),
                classifier_timing['plotting'].get('roc_curves', 0),
                classifier_timing['plotting'].get('pr_curves', 0),
                classifier_timing['plotting'].get('feature_importances', 0),
                classifier_timing['plotting'].get('misclassifications', 0)
            ])
            
            # Store results
            all_results[f"{classifier_name}_{feature_type}"] = {
                'metrics': aggregated_metrics,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'feature_type': feature_type,
                'classifier': classifier_name,
                'n_runs': args.n_runs,
                'timing': classifier_timing.copy()
            }
            
            # Print summary
            print(f"\n{classifier_name.upper()} - {feature_type.upper()} Results (Mean ± Std across {args.n_runs} runs):")
            print_evaluation_summary(aggregated_metrics)
            
            # Save results (include timing in metrics)
            aggregated_metrics['timing'] = classifier_timing.copy()
            save_evaluation_results(
                aggregated_metrics,
                results_dir / f"{classifier_name}_{feature_type}_results",
                format='both'
            )
            
            # Save individual run results
            for run_idx, metrics in enumerate(all_metrics):
                save_evaluation_results(
                    metrics,
                    results_dir / f"{classifier_name}_{feature_type}_run_{run_idx+1}_results",
                    format='json'
                )
        
        elif args.split == 'cv':
            # Cross-validation
            print(f"\n[3/5] Cross-Validation (5-fold)...")
            cv_start = time.time()
            
            # Perform detailed CV analysis
            label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
            cv_analysis_start = time.time()
            detailed_cv = detailed_cv_analysis(
                pipeline, features, labels,
                classifier_name=classifier_name,
                cv=5,
                labels=label_names
            )
            cv_analysis_time = time.time() - cv_analysis_start
            classifier_timing['cv_analysis'] = cv_analysis_time
            print(f"  CV Analysis Time: {cv_analysis_time:.2f} seconds")
            
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
            train_start = time.time()
            pipeline.fit(features, labels, classifier_name=classifier_name)
            train_time = time.time() - train_start
            classifier_timing['training'] = {'total': train_time}
            print(f"  Training Time: {train_time:.2f} seconds")
            
            # Also evaluate on full dataset for confusion matrix and curves
            eval_start = time.time()
            y_pred = pipeline.predict(features, classifier_name=classifier_name)
            y_proba = pipeline.predict_proba(features, classifier_name=classifier_name)
            eval_time = time.time() - eval_start
            classifier_timing['evaluation'] = {'total': eval_time}
            print(f"  Evaluation Time: {eval_time:.2f} seconds")
            
            # Compute metrics
            metrics_start = time.time()
            metrics = compute_metrics(
                labels, y_pred, y_proba,
                labels=label_names
            )
            metrics_time = time.time() - metrics_start
            classifier_timing['metrics_computation'] = {'total': metrics_time}
            
            # Add CV results to metrics
            metrics['cv_results'] = detailed_cv
            
            # Generate confusion matrix
            cm_start = time.time()
            cm = generate_confusion_matrix(
                labels, y_pred, label_names,
                save_path=confusion_dir / f"{classifier_name}_{feature_type}_cv_cm.csv",
                save_plot=True
            )
            cm_time = time.time() - cm_start
            classifier_timing['plotting'] = classifier_timing.get('plotting', {})
            classifier_timing['plotting']['confusion_matrix'] = cm_time
            
            # Generate ROC curves
            print(f"  Generating ROC curves...")
            roc_start = time.time()
            roc_results = generate_roc_curves(
                labels, y_proba, label_names,
                save_path=roc_dir / f"{classifier_name}_{feature_type}_cv_roc.png"
            )
            roc_time = time.time() - roc_start
            classifier_timing['plotting']['roc_curves'] = roc_time
            print(f"  Time: {roc_time:.2f} seconds")
            if roc_results:
                metrics['roc_curves'] = roc_results
            
            # Generate PR curves
            print(f"  Generating PR curves...")
            pr_start = time.time()
            pr_results = generate_pr_curves(
                labels, y_proba, label_names,
                save_path=pr_dir / f"{classifier_name}_{feature_type}_cv_pr.png"
            )
            pr_time = time.time() - pr_start
            classifier_timing['plotting']['pr_curves'] = pr_time
            print(f"  Time: {pr_time:.2f} seconds")
            if pr_results:
                metrics['pr_curves'] = pr_results
                metrics['macro_aupr'] = pr_results.get('macro_aupr')
            
            # Extract feature importances (if Random Forest)
            if classifier_name == 'rf':
                print(f"  Extracting feature importances...")
                imp_start = time.time()
                importance_results = extract_feature_importances(
                    pipeline, classifier_name='rf', feature_type=feature_type,
                    save_path=importance_dir / f"{classifier_name}_{feature_type}_cv_importances"
                )
                imp_time = time.time() - imp_start
                classifier_timing['plotting']['feature_importances'] = imp_time
                print(f"  Time: {imp_time:.2f} seconds")
                if importance_results:
                    metrics['feature_importances'] = importance_results
            
            # Analyze misclassifications
            print(f"  Analyzing misclassifications...")
            misclass_start = time.time()
            misclassification_results = analyze_misclassifications(
                labels, y_pred, label_names,
                save_path=misclassification_dir / f"{classifier_name}_{feature_type}_cv_analysis"
            )
            misclass_time = time.time() - misclass_start
            classifier_timing['plotting']['misclassifications'] = misclass_time
            print(f"  Time: {misclass_time:.2f} seconds")
            if misclassification_results:
                metrics['misclassifications'] = misclassification_results
            
            # Calculate total plotting time
            classifier_timing['plotting']['total'] = sum([
                classifier_timing['plotting'].get('confusion_matrix', 0),
                classifier_timing['plotting'].get('roc_curves', 0),
                classifier_timing['plotting'].get('pr_curves', 0),
                classifier_timing['plotting'].get('feature_importances', 0),
                classifier_timing['plotting'].get('misclassifications', 0)
            ])
            
            cv_time = time.time() - cv_start
            classifier_timing['cv_total'] = cv_time
            print(f"  Total CV Time: {cv_time:.2f} seconds")
            
            # Save model
            pipeline.save_model(classifier_name, feature_type, save_dir=models_dir)
            
            # Store results
            all_results[f"{classifier_name}_{feature_type}"] = {
                'metrics': metrics,
                'n_samples': len(features),
                'feature_type': feature_type,
                'classifier': classifier_name,
                'timing': classifier_timing.copy()
            }
            
            # Print summary
            print(f"\n{classifier_name.upper()} - {feature_type.upper()} Results:")
            print_evaluation_summary(metrics)
            
            # Save results (include timing in metrics)
            metrics['timing'] = classifier_timing.copy()
            save_evaluation_results(
                metrics,
                results_dir / f"{classifier_name}_{feature_type}_cv_results",
                format='both'
            )
    
    # Generate comparison report
    print(f"\n[5/5] Generating comparison report...")
    report_start = time.time()
    comparison_path = results_dir / "comparison_report.txt"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Classification Results Comparison\n")
        f.write("="*60 + "\n\n")
        
        for result_name, result_data in all_results.items():
            f.write(f"{result_name.upper()}:\n")
            metrics = result_data['metrics']
            
            # Handle aggregated metrics (mean ± std) or single run metrics
            def get_metric_str(metric_name):
                if metric_name in metrics:
                    val = metrics[metric_name]
                    if isinstance(val, dict) and 'mean' in val:
                        return f"{val['mean']:.4f} ± {val['std']:.4f}"
                    elif val is not None:
                        return f"{val:.4f}"
                return "N/A"
            
            f.write(f"  Accuracy: {get_metric_str('accuracy')}\n")
            f.write(f"  Macro F1: {get_metric_str('macro_f1')}\n")
            f.write(f"  Weighted F1: {get_metric_str('weighted_f1')}\n")
            
            macro_auc_str = get_metric_str('macro_auc')
            if macro_auc_str != "N/A":
                f.write(f"  Macro AUC (AUROC): {macro_auc_str}\n")
            
            macro_aupr_str = get_metric_str('macro_aupr')
            if macro_aupr_str != "N/A":
                f.write(f"  Macro AUPR: {macro_aupr_str}\n")
            
            if 'n_runs' in result_data:
                f.write(f"  Number of runs: {result_data['n_runs']}\n")
            
            # Feature importances (if available)
            if 'feature_importances' in result_data['metrics']:
                f.write(f"\n  Top 5 Feature Importances:\n")
                top_features = result_data['metrics']['feature_importances'].get('top_features', [])[:5]
                for feat in top_features:
                    if 'std' in feat:
                        # Aggregated format (mean ± std)
                        f.write(f"    {feat['feature']}: {feat['importance']:.4f} ± {feat['std']:.4f}\n")
                    else:
                        # Single run format
                        f.write(f"    {feat['feature']}: {feat.get('importance', 0):.4f}\n")
            
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
            
            # Timing information (if available)
            if 'timing' in result_data:
                timing = result_data['timing']
                f.write(f"\n  Timing Information:\n")
                if 'data_loading' in timing:
                    f.write(f"    Data Loading: {timing['data_loading']:.2f} seconds\n")
                if 'train_test_split' in timing:
                    f.write(f"    Train/Test Split: {timing['train_test_split']:.2f} seconds\n")
                if 'training' in timing:
                    if isinstance(timing['training'], dict) and 'total' in timing['training']:
                        f.write(f"    Training: {timing['training']['total']:.2f} seconds")
                        if 'average' in timing['training']:
                            f.write(f" (avg: {timing['training']['average']:.2f}s/run)")
                        f.write("\n")
                if 'evaluation' in timing:
                    if isinstance(timing['evaluation'], dict) and 'total' in timing['evaluation']:
                        f.write(f"    Evaluation: {timing['evaluation']['total']:.2f} seconds")
                        if 'average' in timing['evaluation']:
                            f.write(f" (avg: {timing['evaluation']['average']:.2f}s/run)")
                        f.write("\n")
                if 'plotting' in timing and 'total' in timing['plotting']:
                    f.write(f"    Plotting: {timing['plotting']['total']:.2f} seconds\n")
                if 'cv_total' in timing:
                    f.write(f"    CV Total: {timing['cv_total']:.2f} seconds\n")
            
            f.write("\n")
        
        f.write("="*60 + "\n")
    
    report_time = time.time() - report_start
    print(f"  Time: {report_time:.2f} seconds")
    print(f"  Saved comparison report to: {comparison_path}")
    
    # Calculate total pipeline time
    total_pipeline_time = time.time() - pipeline_start_time
    timing_info['total_pipeline'] = total_pipeline_time
    
    print("\n" + "="*60)
    print("Classification pipeline completed successfully!")
    print("="*60)
    print(f"\nTiming Summary:")
    if 'data_loading' in timing_info:
        print(f"  Data Loading: {timing_info['data_loading']:.2f} seconds")
    
    # Aggregate timing across all classifiers
    total_training = 0
    total_evaluation = 0
    total_plotting = 0
    for result_data in all_results.values():
        if 'timing' in result_data:
            t = result_data['timing']
            if 'training' in t and isinstance(t['training'], dict):
                total_training += t['training'].get('total', 0)
            if 'evaluation' in t and isinstance(t['evaluation'], dict):
                total_evaluation += t['evaluation'].get('total', 0)
            if 'plotting' in t and 'total' in t['plotting']:
                total_plotting += t['plotting']['total']
    
    if total_training > 0:
        print(f"  Total Training: {total_training:.2f} seconds")
    if total_evaluation > 0:
        print(f"  Total Evaluation: {total_evaluation:.2f} seconds")
    if total_plotting > 0:
        print(f"  Total Plotting: {total_plotting:.2f} seconds")
    print(f"  Total Pipeline: {total_pipeline_time:.2f} seconds")
    print(f"\nResults saved to: {results_dir}")
    print(f"Models saved to: {models_dir}")


if __name__ == "__main__":
    main()

