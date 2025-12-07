#!/usr/bin/env python3
"""
Classification Pipeline
Supports multiple feature types and classifiers for malware family classification
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                            classification_report, confusion_matrix)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class ClassificationPipeline:
    """
    Pipeline for malware family classification using various feature types and classifiers
    """
    
    def __init__(self,
                 feature_type: str = 'gcm',
                 classifier_type: str = 'rf',
                 scale_features: bool = True,
                 random_state: int = 42):
        """
        Initialize classification pipeline
        
        Args:
            feature_type: Type of features ('gcm', 'gcn', 'coarse')
            classifier_type: Type of classifier ('rf', 'gbt', 'svm', 'xgb')
            scale_features: Whether to scale features
            random_state: Random seed for reproducibility
        """
        self.feature_type = feature_type
        self.classifier_type = classifier_type
        self.scale_features = scale_features
        self.random_state = random_state
        
        self.scaler = StandardScaler() if scale_features else None
        self.label_encoder = LabelEncoder()
        self.classifier = None
        self.feature_names = None
        
    def _create_classifier(self, **kwargs):
        """Create classifier based on type"""
        if self.classifier_type == 'rf':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.classifier_type == 'gbt':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 3),
                random_state=self.random_state
            )
        elif self.classifier_type == 'xgb':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 3),
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        elif self.classifier_type == 'svm':
            return SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                probability=True,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def _extract_coarse_features(self, dgdvs: List[np.ndarray]) -> np.ndarray:
        """
        Extract coarse features from DGDVs (aggregate statistics)
        
        Args:
            dgdvs: List of DGDV matrices
        
        Returns:
            Feature matrix of shape (n_graphs, n_features)
        """
        features = []
        
        for dgdv in dgdvs:
            if dgdv.size == 0:
                # Empty graph - use zeros
                if len(features) > 0:
                    features.append(np.zeros(features[0].shape))
                else:
                    # First graph is empty - need to know orbit count
                    # Use a default (will be handled later)
                    continue
            else:
                # Aggregate statistics per orbit
                orbit_stats = []
                for orbit_idx in range(dgdv.shape[1]):
                    orbit_counts = dgdv[:, orbit_idx]
                    orbit_stats.extend([
                    np.mean(orbit_counts),
                    np.std(orbit_counts),
                    np.sum(orbit_counts),
                    np.max(orbit_counts),
                    np.min(orbit_counts)
                ])
                features.append(np.array(orbit_stats))
        
        if not features:
            return np.array([])
        
        # Handle case where first graph was empty
        if len(features) > 0 and features[0].size == 0:
            # Use second graph's shape
            if len(features) > 1:
                target_shape = features[1].shape
                features[0] = np.zeros(target_shape)
            else:
                return np.array([])
        
        return np.array(features)
    
    def _load_features(self, feature_type: str, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        Load features based on type
        
        Args:
            feature_type: Type of features ('gcm', 'gcn', 'coarse')
            **kwargs: Additional arguments (file paths, etc.)
        
        Returns:
            Tuple of (feature_matrix, labels)
        """
        if feature_type == 'gcm':
            # Load GCM vectors
            gcm_file = kwargs.get('gcm_file', 'data/gdvs/gcms/all_gcms_3_4node_reduced.pkl')
            with open(gcm_file, 'rb') as f:
                gcms = pickle.load(f)
            
            # Convert to feature matrix
            # Handle variable-length GCMs by padding or using max size
            max_size = max([gcm.size for gcm in gcms if gcm.size > 0], default=0)
            if max_size == 0:
                raise ValueError("No valid GCMs found")
            
            features = []
            for gcm in gcms:
                if gcm.size == 0:
                    features.append(np.zeros(max_size))
                elif gcm.size < max_size:
                    # Pad with zeros
                    padded = np.zeros(max_size)
                    padded[:gcm.size] = gcm
                    features.append(padded)
                else:
                    features.append(gcm[:max_size])  # Truncate if larger
            
            return np.array(features), kwargs.get('labels', [])
            
        elif feature_type == 'gcn':
            # Load GCN embeddings
            embedding_file = kwargs.get('embedding_file', 'data/embeddings/gcn_embeddings.pkl')
            with open(embedding_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            if isinstance(embeddings, list):
                # Handle variable-length embeddings
                max_size = max([emb.size for emb in embeddings if emb.size > 0], default=0)
                features = []
                for emb in embeddings:
                    if emb.size == 0:
                        features.append(np.zeros(max_size))
                    elif emb.size < max_size:
                        padded = np.zeros(max_size)
                        padded[:emb.size] = emb
                        features.append(padded)
                    else:
                        features.append(emb[:max_size])
                return np.array(features), kwargs.get('labels', [])
            else:
                return embeddings, kwargs.get('labels', [])
            
        elif feature_type == 'coarse':
            # Load DGDVs and extract coarse features
            dgdv_file = kwargs.get('dgdv_file', 'data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl')
            with open(dgdv_file, 'rb') as f:
                dgdvs = pickle.load(f)
            
            features = self._extract_coarse_features(dgdvs)
            return features, kwargs.get('labels', [])
            
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def prepare_features(self, X: np.ndarray, y: Optional[List[str]] = None,
                        fit_scaler: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for training/testing
        
        Args:
            X: Feature matrix
            y: Labels (optional)
            fit_scaler: Whether to fit the scaler (True for training, False for testing)
        
        Returns:
            Tuple of (scaled_features, encoded_labels)
        """
        # Handle NaN and inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if self.scale_features and self.scaler is not None:
            if fit_scaler:
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        # Encode labels
        y_encoded = None
        if y is not None:
            if fit_scaler:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
        
        return X, y_encoded
    
    def train(self, X: np.ndarray, y: List[str], **classifier_kwargs) -> Dict:
        """
        Train the classifier
        
        Args:
            X: Feature matrix
            y: Labels
            **classifier_kwargs: Additional arguments for classifier
        
        Returns:
            Dictionary with training results
        """
        # Prepare features
        X_scaled, y_encoded = self.prepare_features(X, y, fit_scaler=True)
        
        # Create and train classifier
        self.classifier = self._create_classifier(**classifier_kwargs)
        self.classifier.fit(X_scaled, y_encoded)
        
        # Predictions on training set
        y_pred = self.classifier.predict(X_scaled)
        y_pred_proba = self.classifier.predict_proba(X_scaled) if hasattr(self.classifier, 'predict_proba') else None
        
        # Compute metrics
        train_accuracy = accuracy_score(y_encoded, y_pred)
        train_f1 = f1_score(y_encoded, y_pred, average='macro')
        
        results = {
            'train_accuracy': train_accuracy,
            'train_f1_macro': train_f1,
            'n_samples': len(y),
            'n_features': X_scaled.shape[1],
            'n_classes': len(self.label_encoder.classes_)
        }
        
        # Add AUC if probabilities available
        if y_pred_proba is not None and len(self.label_encoder.classes_) > 2:
            try:
                train_auc = roc_auc_score(y_encoded, y_pred_proba, average='macro', multi_class='ovr')
                results['train_auc_macro'] = train_auc
            except:
                pass
        
        return results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        X_scaled, _ = self.prepare_features(X, fit_scaler=False)
        y_pred = self.classifier.predict(X_scaled)
        y_pred_proba = self.classifier.predict_proba(X_scaled) if hasattr(self.classifier, 'predict_proba') else None
        
        return y_pred, y_pred_proba
    
    def evaluate(self, X: np.ndarray, y: List[str]) -> Dict:
        """
        Evaluate classifier on test set
        
        Args:
            X: Feature matrix
            y: True labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred, y_pred_proba = self.predict(X)
        _, y_encoded = self.prepare_features(X, y, fit_scaler=False)
        
        # Compute metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        f1_macro = f1_score(y_encoded, y_pred, average='macro')
        f1_micro = f1_score(y_encoded, y_pred, average='micro')
        f1_weighted = f1_score(y_encoded, y_pred, average='weighted')
        
        # Per-class F1
        f1_per_class = f1_score(y_encoded, y_pred, average=None)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(self.label_encoder.classes_, f1_per_class)),
            'confusion_matrix': confusion_matrix(y_encoded, y_pred).tolist(),
            'n_samples': len(y),
            'predictions': y_pred.tolist(),
            'true_labels': y_encoded.tolist()
        }
        
        # Add AUC if probabilities available
        if y_pred_proba is not None:
            try:
                if len(self.label_encoder.classes_) == 2:
                    auc = roc_auc_score(y_encoded, y_pred_proba[:, 1])
                    results['auc'] = auc
                else:
                    auc_macro = roc_auc_score(y_encoded, y_pred_proba, average='macro', multi_class='ovr')
                    results['auc_macro'] = auc_macro
            except Exception as e:
                results['auc_error'] = str(e)
        
        return results
    
    def cross_validate(self, X: np.ndarray, y: List[str], cv: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
        
        Returns:
            Dictionary with CV results
        """
        X_scaled, y_encoded = self.prepare_features(X, y, fit_scaler=True)
        
        # Create classifier
        classifier = self._create_classifier()
        
        # Perform cross-validation
        cv_scores_accuracy = cross_val_score(classifier, X_scaled, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_scores_f1 = cross_val_score(classifier, X_scaled, y_encoded, cv=cv, scoring='f1_macro', n_jobs=-1)
        
        results = {
            'cv_folds': cv,
            'accuracy_mean': np.mean(cv_scores_accuracy),
            'accuracy_std': np.std(cv_scores_accuracy),
            'accuracy_scores': cv_scores_accuracy.tolist(),
            'f1_macro_mean': np.mean(cv_scores_f1),
            'f1_macro_std': np.std(cv_scores_f1),
            'f1_macro_scores': cv_scores_f1.tolist()
        }
        
        return results
    
    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance (for tree-based models)
        
        Returns:
            Dictionary with feature importances or None
        """
        if self.classifier is None:
            return None
        
        if hasattr(self.classifier, 'feature_importances_'):
            importances = self.classifier.feature_importances_
            # Sort by importance
            indices = np.argsort(importances)[::-1]
            
            return {
                'importances': importances.tolist(),
                'sorted_indices': indices.tolist(),
                'top_features': indices[:min(20, len(indices))].tolist()
            }
        
        return None


def train_test_split_stratified(X: np.ndarray, y: List[str],
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Tuple:
    """
    Stratified train/test split
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classification pipeline')
    parser.add_argument('--feature-type', type=str, choices=['gcm', 'gcn', 'coarse'], default='gcm',
                       help='Feature type')
    parser.add_argument('--classifier', type=str, choices=['rf', 'gbt', 'svm', 'xgb'], default='rf',
                       help='Classifier type')
    parser.add_argument('--gcm-file', type=str, default='data/gdvs/gcms/all_gcms_3_4node_reduced.pkl',
                       help='GCM file path')
    parser.add_argument('--dgdv-file', type=str, default='data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl',
                       help='DGDV file path')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Load labels (this would need to be implemented based on your data structure)
    print("Classification pipeline - use from run_analysis.py for full pipeline")

