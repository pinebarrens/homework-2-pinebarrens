"""
Homework 1: Regression and Classification Models

Stencil for:
1. Linear Regression (for genomic methylation data)
2. Logistic Regression (for heart disease data)

Students will implement model training, evaluation, and K-fold cross-validation.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class GenomicAgeRegressor:
    def __init__(
        self,
        model_type: str = "linear",
        random_state: int = 42,
        selected_features: Optional[List[str]] = None,
        alpha: float = 1.0,
    ):
        """
        Initialize the regressor with specified parameters.

        Args:
            model_type: Type of regression model ('linear', 'ridge', 'lasso')
            random_state: Random seed for reproducibility
            selected_features: Optional list of feature names to use
            alpha: Regularization strength for ridge/lasso
        """
        self.model_type = model_type
        self.random_state = random_state
        self.selected_features = selected_features
        self.alpha = alpha

        self.model = None
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Select features and fit the regression model. Save the model as self.model.
        """
        # TODO: Implement model fitting
        # 1. Select features if specified (self.selected_features)
        # 2. Scale features using self.scaler
        # 3. Initialize the model based on self.model_type:
        #       - "linear" -> LinearRegression()
        #       - "ridge"  -> Ridge(alpha=self.alpha, random_state=self.random_state)
        #       - "lasso"  -> Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=10000)
        #    If an unknown model_type is provided, raise a ValueError.
        # 4. Fit the model and store in self.model
        if self.selected_features is None:
            X_scaled = self.scaler.fit_transform(X)
        elif len(self.selected_features) > 0:
            self.scaler.fit(X[self.selected_features])
            X_scaled = self.scaler.transform(X[self.selected_features])
        
        if self.model_type == "linear":
            lin_model = LinearRegression()
            self.model = lin_model.fit(X_scaled, y)
        elif self.model_type == "ridge":
            ridge_model = Ridge(alpha=self.alpha, random_state=self.random_state)
            self.model = ridge_model.fit(X_scaled, y)
        elif self.model_type == "lasso":
            lasso_model = Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=10000)
            self.model = lasso_model.fit(X_scaled, y)
        else:
            raise ValueError("Model type is invalid")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        # TODO: Implement prediction
        # 1. Ensure self.model is trained (raise ValueError if not)
        # 2. Select the same features as used in training (self.selected_features)
        # 3. Scale using self.scaler.transform
        # 4. Return predictions from self.model
        if not self.model:
            raise ValueError("Model is not trained")
        
        if self.selected_features is None:
            X_scaled = self.scaler.transform(X)
        elif len(self.selected_features) > 0:
            X_scaled = self.scaler.transform(X[self.selected_features])

        return self.model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance and return metrics.
        """
        # TODO: Implement model evaluation
        # Use self.predict(X) then compute:
        #   - mse: mean_squared_error(y, y_pred)
        #   - rmse: sqrt(mse)
        #   - r2:  r2_score(y, y_pred)
        #   - mae: mean_absolute_error(y, y_pred)
        # Return a dict with keys: "mse", "rmse", "r2", "mae"
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return {"mse": mse, "rmse": rmse, "r2": r2, "mae": mae}


    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform K-fold cross-validation and return metrics for n_splits folds.

        Returns:
            cv_results: Dictionary with lists of metrics for each fold
        """
        # TODO: Implement K-fold cross-validation using KFold
        # For each fold:
        #   1. Split data into train/val
        #   2. Fit on train
        #   3. Evaluate on val
        #   4. Append metrics to cv_results
        
        KFolds = KFold(n_splits=n_splits)
        cv_results = {}
        for train_index, val_index in KFolds.split(X):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_val = X.iloc[val_index]
            y_val = y.iloc[val_index]
            self.fit(X_train, y_train)
            metrics = self.evaluate(X_val, y_val)

            for x in metrics:
                if x not in cv_results:
                    cv_results[x] = []
                cv_results[x].append(metrics[x])

        return cv_results


class LogisticClassifier:
    def __init__(
        self,
        C: float = 1.0,
        random_state: int = 42,
    ):
        """
        Initialize the classifier with specified parameters. Uses the lbfgs solver.

        Args:
            C: Inverse of regularization strength
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.scaling = False

    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for the model.

        Returns:
            X_processed: Processed feature matrix
        """
        # TODO: Implement feature preprocessing
        # Default behavior should return X unchanged
        X = X.fillna(X.mean(), inplace=True) # Impute mean for missing values
        return X.copy()

    def fit(self, X: pd.DataFrame, y: pd.Series, Scaling = False) -> None:
        """
        Preprocess features and fit the classification model.
        Save the fitted model in self.model.

        Args:
            X: Feature matrix
            y: Target variable (heart disease presence)
        """
        # TODO: Implement model fitting
        # 1. Preprocess features via self.preprocess_features
        # 2. Scale features using self.scaler
        # 3. Initialize LogisticRegression
        # 4. Fit the model and store in self.model
        self.scaling = Scaling
        X = self.preprocess_features(X)

        if self.scaling == True:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        
        log_reg = LogisticRegression(max_iter=1000)
        self.model = log_reg.fit(X, y)

    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Make binary predictions using the trained model (self.model).

        Args:
            X: Feature matrix
            return_proba: If True, return probability of class 1 instead of hard labels

        Returns:
            y_pred: Binary predictions (0 or 1) if return_proba=False
            y_proba: Probability predictions for class 1 if return_proba=True
        """
        # TODO: Implement prediction
        # 1. Ensure self.model is trained
        # 2. Preprocess features
        # 3. Scale using self.scaler.transform
        # 4. If return_proba: return self.model.predict_proba(X_scaled)[:, 1]
        #    else: return self.model.predict(X_scaled)
        X = self.preprocess_features(X)
        
        if self.scaling == True:
            X = self.scaler.transform(X)

        if return_proba: 
            return self.model.predict_proba(X)
        else: 
            return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # TODO: Implement model evaluation
        # Compute:
        #   y_pred  = self.predict(X)
        #   y_proba = self.predict(X, return_proba=True)
        #
        # Use zero_division=0 to avoid crashes on edge cases where a fold has no predicted positives:
        #   precision_score(y, y_pred, zero_division=0)
        #   recall_score(y, y_pred, zero_division=0)
        #   f1_score(y, y_pred, zero_division=0)
        #
        # ROC-AUC safeguard:
        #   Only compute roc_auc_score(y, y_proba) if BOTH classes appear in y (i.e., len(np.unique(y)) == 2).
        #   Otherwise, set "auc" to np.nan (or your chosen sentinel).
        #
        # Return dict with keys: "accuracy", "precision", "recall", "f1", "auc"
        y_pred = self.predict(X)
        y_proba = self.predict(X, return_proba=True)
        precision = precision_score(y, y_pred, zero_division=0, average='weighted')
        recall = recall_score(y, y_pred, zero_division=0, average='weighted')
        f1 = f1_score(y, y_pred, zero_division=0, average='weighted')
        accuracy = accuracy_score(y, y_pred)
        if len(np.unique(y)) > 2:
            auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
        else:
            auc = np.nan
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, Scaling = False
    ) -> Dict[str, List[float]]:
        """
        Perform K-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            n_splits: Number of folds for cross-validation

        Returns:
            cv_results: Dictionary with lists of metrics for each fold
        """
        # TODO: Implement K-fold cross-validation using KFold
        # For each fold:
        #   1. Split data into train/val
        #   2. Fit on train
        #   3. Evaluate on val
        #   4. Append metrics to cv_results
        #
        # Note: The ROC-AUC should effectively apply per fold:
        #   If y_val contains only one class, "auc" for that fold should be np.nan (or your chosen sentinel).
        KFolds = KFold(n_splits=n_splits)
        cv_results = {}
        for train_index, val_index in KFolds.split(X):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_val = X.iloc[val_index]
            y_val = y.iloc[val_index]

            self.fit(X_train, y_train, Scaling)
            metrics = self.evaluate(X_val, y_val)

            for x in metrics:
                if x not in cv_results:
                    cv_results[x] = []
                cv_results[x].append(metrics[x])
        
        return cv_results


