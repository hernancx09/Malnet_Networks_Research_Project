#!/usr/bin/env python3
"""
Classification Module
Random Forest and Support Vector Machine classifiers for malware family classification
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold


class ClassificationPipeline:
    """
    Pipeline for training and evaluating classifiers
    """
    
    def __init__(self, models_dir: str = "data/models", random_state: int = 42):
        """
        Initialize classification pipeline
        
        Args:
            models_dir: Directory for saving models
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize classifiers
        self.classifiers = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,  # Required for predict_proba and AUC
                random_state=random_state
            )
        }
        
        # Feature scaler
        self.feature_scaler = StandardScaler()
        
        # Label encoder (will be set when fitting)
        self.label_encoder = None
        
        # Track which classifier is currently trained
        self.current_classifier_name = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, classifier_name: str = "rf") -> None:
        """
        Fit classifier on training data
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            classifier_name: Name of classifier ('rf' or 'svm')
        """
        if classifier_name not in self.classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}. Must be 'rf' or 'svm'")
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Fit classifier
        self.classifiers[classifier_name].fit(X_scaled, y)
        self.current_classifier_name = classifier_name
        
        print(f"Fitted {classifier_name.upper()} classifier on {X.shape[0]} samples with {X.shape[1]} features")
    
    def predict(self, X: np.ndarray, classifier_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions on test data
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            classifier_name: Name of classifier ('rf' or 'svm'). If None, uses current classifier
        
        Returns:
            Predicted labels
        """
        if classifier_name is None:
            classifier_name = self.current_classifier_name
        
        if classifier_name is None:
            raise ValueError("No classifier specified and no current classifier set")
        
        if classifier_name not in self.classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}. Must be 'rf' or 'svm'")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict
        return self.classifiers[classifier_name].predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray, classifier_name: Optional[str] = None) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            classifier_name: Name of classifier ('rf' or 'svm'). If None, uses current classifier
        
        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        if classifier_name is None:
            classifier_name = self.current_classifier_name
        
        if classifier_name is None:
            raise ValueError("No classifier specified and no current classifier set")
        
        if classifier_name not in self.classifiers:
            raise ValueError(f"Unknown classifier: {classifier_name}. Must be 'rf' or 'svm'")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Predict probabilities
        return self.classifiers[classifier_name].predict_proba(X_scaled)
    
    def save_model(self, classifier_name: str, feature_type: str, save_dir: Optional[Path] = None) -> None:
        """
        Save trained model components
        
        Args:
            classifier_name: Name of classifier ('rf' or 'svm')
            feature_type: Type of features used ('coarse', 'gcm', or 'both')
            save_dir: Directory to save model. If None, uses self.models_dir
        """
        if save_dir is None:
            save_dir = self.models_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{classifier_name}_{feature_type}_model.pkl"
        model_path = save_dir / model_filename
        
        model_data = {
            'classifier': self.classifiers[classifier_name],
            'scaler': self.feature_scaler,
            'label_encoder': self.label_encoder,
            'classifier_name': classifier_name,
            'feature_type': feature_type,
            'random_state': self.random_state
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved model to: {model_path}")
    
    def load_model(self, model_path: Path) -> None:
        """
        Load saved model components
        
        Args:
            model_path: Path to saved model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier_name = model_data['classifier_name']
        self.classifiers[classifier_name] = model_data['classifier']
        self.feature_scaler = model_data['scaler']
        self.label_encoder = model_data.get('label_encoder')
        self.random_state = model_data.get('random_state', 42)
        self.current_classifier_name = classifier_name
        
        print(f"Loaded model from: {model_path}")


def train_test_split_graphs(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Graph-level train/test split (stratified)
    
    Args:
        X: Feature matrix of shape (n_graphs, n_features)
        y: Label array of shape (n_graphs,)
        test_size: Proportion of data for test set
        random_state: Random seed
        stratify: Whether to preserve class proportions
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
    
    return X_train, X_test, y_train, y_test


def cross_validate(
    pipeline: ClassificationPipeline,
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str = "rf",
    cv: int = 5,
    scoring: Optional[list] = None
) -> Dict:
    """
    Perform stratified K-fold cross-validation
    
    Args:
        pipeline: ClassificationPipeline instance
        X: Feature matrix
        y: Label array
        classifier_name: Name of classifier ('rf' or 'svm')
        cv: Number of folds
        scoring: List of scoring metrics. If None, uses default metrics
    
    Returns:
        Dictionary with mean/std scores for each metric
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, accuracy_score, f1_score
    
    if scoring is None:
        scoring = ['accuracy', 'f1_macro', 'f1_weighted']
    
    # Scale features
    X_scaled = pipeline.feature_scaler.fit_transform(X)
    
    # Get classifier
    classifier = pipeline.classifiers[classifier_name]
    
    # Perform cross-validation
    results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=pipeline.random_state)
    
    for metric_name in scoring:
        if metric_name == 'accuracy':
            scorer = make_scorer(accuracy_score)
        elif metric_name == 'f1_macro':
            scorer = make_scorer(f1_score, average='macro')
        elif metric_name == 'f1_weighted':
            scorer = make_scorer(f1_score, average='weighted')
        elif metric_name == 'f1_micro':
            scorer = make_scorer(f1_score, average='micro')
        else:
            continue
        
        scores = cross_val_score(
            classifier,
            X_scaled,
            y,
            cv=skf,
            scoring=scorer,
            n_jobs=-1
        )
        
        results[metric_name] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'scores': scores.tolist()
        }
    
    return results

