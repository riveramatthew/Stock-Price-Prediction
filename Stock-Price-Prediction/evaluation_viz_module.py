"""
Evaluation and Visualization Module
Comprehensive performance analysis and plotting utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and visualizations
    """
    
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         name: str = "Model") -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            name: Model name for logging
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Direction Accuracy (up/down prediction)
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(true_direction == pred_direction) * 100
        
        # Max Error
        max_error = np.max(np.abs(y_true - y_pred))
        
        # Bias (average error)
        bias = np.mean(y_pred - y_true)
        
        metrics = {
            'Model': name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'Max_Error': max_error,
            'Bias': bias
        }
        
        self.results[name] = metrics
        return metrics
    
    def evaluate_by_horizon(self, df: pd.DataFrame, predictions: Dict[str, pd.Series],
                           ticker: str, horizons: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
        """
        Evaluate predictions at different time horizons
        
        Args:
            df: DataFrame with actual prices
            predictions: Dictionary of predictions
            ticker: Stock ticker
            horizons: List of day horizons to evaluate
            
        Returns:
            DataFrame with results by horizon
        """
        results = []
        
        for horizon in horizons:
            # Get actual prices at horizon
            actual = df[ticker]['Adj Close'].shift(-horizon)
            pred = predictions[ticker]
            
            # Align and drop NaN
            mask = ~(actual.isna() | pred.isna())
            actual_aligned = actual[mask].values
            pred_aligned = pred[mask].values
            
            if len(actual_aligned) > 0:
                metrics = self.calculate_metrics(
                    actual_aligned, 
                    pred_aligned,
                    f"{horizon}-day"
                )
                metrics['Horizon'] = horizon
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models side by side
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with comparison
        """
        comparison = pd.DataFrame(model_results).T
        comparison = comparison.sort_values('RMSE')
        return comparison
    
    def plot_predictions_vs_actual(self, dates: pd.DatetimeIndex, 
                                   y_true: np.ndarray, y_pred: np.ndarray,
                                   ticker: str, save_path: str = None):
        """
        Plot predicted vs actual prices over time
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Actual vs Predicted
        ax1.plot(dates, y_true, label='Actual', color='blue', linewidth=2, alpha=0.7)
        ax1.plot(dates, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.7)
        ax1.fill_between(dates, y_true, y_pred, alpha=0.2, color='gray')
        ax1.set_title(f'{ticker} Stock Price: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        error = y_pred - y_true
        ax2.plot(dates, error, color='purple', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(dates, 0, error, where=(error > 0), alpha=0.3, color='red', label='Over-prediction')
        ax2.fill_between(dates, 0, error, where=(error < 0), alpha=0.3, color='green', label='Under-prediction')
        ax2.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Error ($)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               ticker: str, save_path: str = None):
        """
        Plot distribution of prediction errors
        """
        error = y_pred - y_true
        percent_error = (error / y_true) * 100
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Histogram of absolute errors
        axes[0].hist(error, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Error ($)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of percentage errors
        axes[1].hist(percent_error, bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Distribution of Percentage Errors', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Error (%)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Scatter plot: Actual vs Predicted
        axes[2].scatter(y_true, y_pred, alpha=0.5, s=30, color='darkgreen')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[2].set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Actual Price ($)', fontsize=11)
        axes[2].set_ylabel('Predicted Price ($)', fontsize=11)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{ticker} - Error Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """
        Plot comparison of multiple models
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        colors = ['steelblue', 'coral', 'green', 'purple']
        
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[idx // 2, idx % 2]
            
            comparison_df[metric].plot(kind='barh', ax=ax, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric} by Model', fontsize=12, fontweight='bold')
            ax.set_xlabel(metric, fontsize=11)
            ax.set_ylabel('Model', fontsize=11)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(comparison_df[metric]):
                ax.text(v, i, f' {v:.3f}', va='center', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_horizon_performance(self, horizon_df: pd.DataFrame, save_path: str = None):
        """
        Plot performance metrics across different prediction horizons
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['MAPE', 'RMSE', 'Direction_Accuracy', 'R2']
        titles = ['MAPE by Horizon', 'RMSE by Horizon', 
                 'Direction Accuracy by Horizon', 'R² by Horizon']
        colors = ['red', 'blue', 'green', 'purple']
        
        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[idx // 2, idx % 2]
            
            ax.plot(horizon_df['Horizon'], horizon_df[metric], 
                   marker='o', linewidth=2, markersize=8, color=color)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Prediction Horizon (days)', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for x, y in zip(horizon_df['Horizon'], horizon_df[metric]):
                ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                          xytext=(0, 10), ha='center', fontsize=9)
        
        plt.suptitle('Performance vs Prediction Horizon', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                               top_n: int = 15, save_path: str = None):
        """
        Plot feature importance for tree-based models
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), importances[indices], color='steelblue', alpha=0.7, edgecolor='black')
            plt.yticks(range(top_n), [feature_names[i] for i in indices])
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Model does not support feature importance")
    
    def generate_report(self, ticker: str, save_path: str = None) -> str:
        """
        Generate a text report of evaluation results
        """
        report = []
        report.append("="*70)
        report.append(f"STOCK PRICE PREDICTION EVALUATION REPORT - {ticker}")
        report.append("="*70)
        report.append("")
        
        if self.results:
            for model_name, metrics in self.results.items():
                report.append(f"Model: {model_name}")
                report.append("-"*50)
                report.append(f"  RMSE: ${metrics['RMSE']:.2f}")
                report.append(f"  MAE: ${metrics['MAE']:.2f}")
                report.append(f"  MAPE: {metrics['MAPE']:.2f}%")
                report.append(f"  R²: {metrics['R2']:.4f}")
                report.append(f"  Direction Accuracy: {metrics['Direction_Accuracy']:.2f}%")
                report.append(f"  Max Error: ${metrics['Max_Error']:.2f}")
                report.append(f"  Bias: ${metrics['Bias']:.2f}")
                report.append("")
        
        report.append("="*70)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


# Example usage
if __name__ == "__main__":
    print("Model Evaluation and Visualization Module")
    print("Use this module to evaluate model performance and create visualizations")
