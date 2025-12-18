"""
Feature Selection and Classification for ParData Datasets
Author: [Your Names Here]
Matriculation Numbers: [Your Numbers Here]

This script solves the feature selection problem for three high-dimensional datasets.
It uses multiple approaches to identify relevant features and build accurate classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    """
    Comprehensive feature selection and classification system.
    Uses multiple methods to identify relevant features.
    """
    
    def __init__(self, dataset_path, n_features):
        """
        Initialize with dataset path and number of features.
        
        Args:
            dataset_path: Path to CSV file
            n_features: Total number of features in dataset
        """
        self.dataset_path = dataset_path
        self.n_features = n_features
        self.data = None
        self.X = None
        self.y = None
        self.selected_features = None
        self.best_model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the dataset."""
        print(f"\n{'='*70}")
        print(f"Loading dataset: {self.dataset_path}")
        print(f"{'='*70}")
        
        # Load data
        self.data = pd.read_csv(self.dataset_path, header=None)
        
        # Separate features and labels
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Number of samples: {len(self.X)}")
        print(f"Number of features: {self.n_features}")
        print(f"Number of classes: {len(np.unique(self.y))}")
        print(f"Class distribution: {np.bincount(self.y.astype(int))}")
        
    def method1_random_forest_importance(self, n_top=20):
        """Method 1: Random Forest Feature Importance."""
        print("\n[METHOD 1] Random Forest Feature Importance")
        print("-" * 70)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                    random_state=42, n_jobs=-1)
        rf.fit(self.X, self.y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select top features
        top_indices = np.argsort(importances)[-n_top:][::-1]
        
        print(f"Top {n_top} features by importance:")
        for i, idx in enumerate(top_indices[:10]):
            print(f"  Feature {idx+1}: {importances[idx]:.6f}")
        
        return top_indices, importances
    
    def method2_mutual_information(self, n_top=20):
        """Method 2: Mutual Information."""
        print("\n[METHOD 2] Mutual Information")
        print("-" * 70)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
        
        # Select top features
        top_indices = np.argsort(mi_scores)[-n_top:][::-1]
        
        print(f"Top {n_top} features by MI:")
        for i, idx in enumerate(top_indices[:10]):
            print(f"  Feature {idx+1}: {mi_scores[idx]:.6f}")
        
        return top_indices, mi_scores
    
    def method3_anova_f_test(self, n_top=20):
        """Method 3: ANOVA F-test."""
        print("\n[METHOD 3] ANOVA F-test")
        print("-" * 70)
        
        # Calculate F-scores
        selector = SelectKBest(f_classif, k=n_top)
        selector.fit(self.X, self.y)
        
        f_scores = selector.scores_
        top_indices = np.argsort(f_scores)[-n_top:][::-1]
        
        print(f"Top {n_top} features by F-score:")
        for i, idx in enumerate(top_indices[:10]):
            print(f"  Feature {idx+1}: {f_scores[idx]:.2f}")
        
        return top_indices, f_scores
    
    def method4_correlation_analysis(self, threshold=0.1):
        """Method 4: Correlation with target."""
        print("\n[METHOD 4] Correlation Analysis")
        print("-" * 70)
        
        correlations = []
        for i in range(self.X.shape[1]):
            corr = np.corrcoef(self.X[:, i], self.y)[0, 1]
            correlations.append(abs(corr))
        
        correlations = np.array(correlations)
        top_indices = np.where(correlations > threshold)[0]
        top_indices = top_indices[np.argsort(correlations[top_indices])[::-1]]
        
        print(f"Features with |correlation| > {threshold}: {len(top_indices)}")
        if len(top_indices) > 0:
            for i, idx in enumerate(top_indices[:10]):
                print(f"  Feature {idx+1}: {correlations[idx]:.6f}")
        
        return top_indices, correlations
    
    def combine_feature_rankings(self, method_results, n_final=10):
        """Combine results from multiple methods using voting."""
        print("\n[ENSEMBLE] Combining Feature Rankings")
        print("-" * 70)
        
        feature_votes = np.zeros(self.n_features)
        
        for features, scores in method_results:
            # Give votes based on ranking
            for rank, idx in enumerate(features[:20]):
                feature_votes[idx] += (20 - rank)
        
        # Select top features by votes
        top_indices = np.argsort(feature_votes)[-n_final:][::-1]
        
        print(f"Top {n_final} features by ensemble voting:")
        for i, idx in enumerate(top_indices):
            print(f"  Feature {idx+1}: votes={feature_votes[idx]:.1f}")
        
        return top_indices
    
    def evaluate_features(self, feature_indices, test_size=0.2):
        """Evaluate classification performance using selected features."""
        print("\n[EVALUATION] Testing Selected Features")
        print("-" * 70)
        
        # Extract selected features
        X_selected = self.X[:, feature_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Test multiple classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, 
                                                   random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, 
                                                           max_depth=5, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        results = {}
        best_acc = 0
        
        for name, clf in classifiers.items():
            # Train
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_scaled)
            
            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'accuracy': acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"\n{name}:")
            print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            if acc > best_acc:
                best_acc = acc
                self.best_model = clf
        
        return results, best_acc
    
    def optimize_feature_count(self):
        """Find optimal number of features."""
        print("\n[OPTIMIZATION] Finding Optimal Feature Count")
        print("-" * 70)
        
        # Get all feature rankings
        rf_features, rf_scores = self.method1_random_forest_importance(n_top=50)
        
        best_n_features = 4
        best_accuracy = 0
        accuracy_history = []
        
        # Test different numbers of features
        for n in [3, 4, 5, 6, 8, 10, 12, 15, 20]:
            if n > len(rf_features):
                break
                
            top_n = rf_features[:n]
            X_selected = self.X[:, top_n]
            
            # Quick evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train_scaled, y_train)
            acc = clf.score(X_test_scaled, y_test)
            
            accuracy_history.append((n, acc))
            print(f"  n={n:2d} features: accuracy={acc:.4f} ({acc*100:.2f}%)")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_n_features = n
        
        print(f"\nOptimal: {best_n_features} features with {best_accuracy:.4f} accuracy")
        return best_n_features, accuracy_history
    
    def run_complete_analysis(self):
        """Run complete feature selection pipeline."""
        # Load data
        self.load_data()
        
        # Run all feature selection methods
        print("\n" + "="*70)
        print("RUNNING FEATURE SELECTION METHODS")
        print("="*70)
        
        rf_features, rf_scores = self.method1_random_forest_importance()
        mi_features, mi_scores = self.method2_mutual_information()
        f_features, f_scores = self.method3_anova_f_test()
        corr_features, corr_scores = self.method4_correlation_analysis()
        
        # Combine methods
        method_results = [
            (rf_features, rf_scores),
            (mi_features, mi_scores),
            (f_features, f_scores),
            (corr_features, corr_scores)
        ]
        
        # Find optimal number of features
        optimal_n, history = self.optimize_feature_count()
        
        # Get final feature set
        ensemble_features = self.combine_feature_rankings(method_results, n_final=optimal_n)
        
        # Final evaluation
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        
        results, final_accuracy = self.evaluate_features(ensemble_features)
        
        # Store results
        self.selected_features = ensemble_features
        
        return {
            'selected_features': ensemble_features + 1,  # Convert to 1-indexed
            'accuracy': final_accuracy,
            'cv_results': results,
            'optimization_history': history
        }

def main():
    """Main execution function."""
    
    # Define base path
    base_path = r'C:\Projects\90 Percent'
    
    # Define datasets with full paths
    datasets = [
        (f'{base_path}\\data-100000-100-4-rnd.csv', 100),
        (f'{base_path}\\data-100000-1000-4-rnd.csv', 1000),
        (f'{base_path}\\data-100000-10000-4-rnd.csv', 10000)
    ]
    
    all_results = {}
    
    for dataset_path, n_features in datasets:
        print("\n" + "#"*70)
        print(f"# PROCESSING: {dataset_path}")
        print("#"*70)
        
        try:
            selector = FeatureSelector(dataset_path, n_features)
            results = selector.run_complete_analysis()
            all_results[dataset_path] = results
            
            print("\n" + "="*70)
            print("SUMMARY FOR", dataset_path)
            print("="*70)
            print(f"Selected Features (1-indexed): {list(results['selected_features'])}")
            print(f"Final Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            
            if results['accuracy'] >= 0.90:
                print("✓ SUCCESS: Accuracy >= 90%")
            else:
                print("✗ WARNING: Accuracy < 90%")
            
        except FileNotFoundError:
            print(f"ERROR: File {dataset_path} not found!")
            print("Please ensure the CSV file is in the same directory as this script.")
        except Exception as e:
            print(f"ERROR processing {dataset_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print final summary
    print("\n" + "#"*70)
    print("# FINAL SUMMARY - ALL DATASETS")
    print("#"*70)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset}:")
        print(f"  Features: {list(results['selected_features'])}")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()