# credit_scoring_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def split_data(df, target_col, test_size=0.3, random_state=42,stratify=True):
    """
    Splits the DataFrame into training and testing sets.
    
    Parameters:
    - df: The input DataFrame
    - target_col: The column name of the target variable
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Random state for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_models(X_train, y_train):
    """
    Trains Logistic Regression, Decision Tree, Random Forest, and GBM models.
    
    Parameters:
    - X_train: Features for training
    - y_train: Target for training
    
    Returns:
    - models: A dictionary containing trained models
    """
    models = {
        'Logistic Regression': LogisticRegression(C=1, penalty='l1', random_state=42, solver='liblinear'),
        'Decision Tree':DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2,
                       min_samples_split=10, random_state=42)}
    mdls=[]
    for name, model in models.items():
        model.fit(X_train, y_train)
        mdls.append(model)
    return mdls

def hyperparameter_tuning(model, param_grid, X_train, y_train, search_type='grid'):
    """
    Performs hyperparameter tuning using Grid Search or Random Search.
    
    Parameters:
    - model: The model to tune
    - param_grid: The parameter grid for tuning
    - X_train: Features for training
    - y_train: Target for training
    - search_type: Type of search ('grid' or 'random')
    
    Returns:
    - best_model: The best model after tuning
    """
    if search_type == 'grid':
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    elif search_type == 'random':
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

def evaluate_models(models, X_test, y_test):
    """
    Evaluates models and returns their performance metrics.
    
    Parameters:
    - models: A dictionary of models
    - X_test: Features for testing
    - y_test: Target for testing
    
    Returns:
    - results_df: DataFrame with performance metrics
    """
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC-AUC': []
    }
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results['Model'].append(name)
        results['Accuracy'].append(accuracy_score(y_test, y_pred))
        results['Precision'].append(precision_score(y_test, y_pred))
        results['Recall'].append(recall_score(y_test, y_pred))
        results['F1 Score'].append(f1_score(y_test, y_pred))
        results['ROC-AUC'].append(roc_auc_score(y_test, y_pred))
    
    return pd.DataFrame(results)

    