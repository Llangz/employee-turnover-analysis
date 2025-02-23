from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import shap

def train_model(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train the employee turnover prediction model
    
    Args:
        X: Feature DataFrame
        y: Target Series (1 for turnover, 0 for retained)
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple containing:
        - Trained model
        - Training features
        - Test features 
        - Training labels
        - Test labels
    """
    # Split data with stratification to handle imbalanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42,
        stratify=y
    )
    
    # Initialize model with HR-specific parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list
) -> Dict:
    """
    Evaluate model performance with HR-specific metrics
    
    Args:
        model: Trained RandomForestClassifier
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X_test, y_test, cv=5)
    
    print("\n=== Employee Turnover Model Performance ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create output directory if it doesn't exist
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Factors Influencing Employee Turnover')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('outputs/figures/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP values for detailed feature impact
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Impact Analysis')
    plt.tight_layout()
    plt.savefig('outputs/figures/shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'feature_importance': feature_importance,
        'cv_scores': cv_scores,
        'predictions': y_pred,
        'probabilities': y_prob
    }