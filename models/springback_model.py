"""
MODEL 1: SPRINGBACK SURROGATE MODEL
====================================

Purpose:
    Learn the forward physics relationship: (rx, ryr) → final_bend_angle
    
Engineering Justification:
    Sheet metal bending involves elastic-plastic deformation with springback.
    The relationship between tool inputs and final angle is:
    - Highly non-linear (material yield, strain hardening)
    - Path-dependent (loading/unloading hysteresis)
    - Multi-physics (geometry + material + contact)
    
    Tree-based models (RandomForest) are ideal because:
    1. Capture non-linear relationships without explicit functional form
    2. Handle interactions between rx and ryr naturally
    3. Robust to outliers in FEM data
    4. Interpretable through feature importance
    5. No assumptions about elastic-plastic constitutive laws

Output:
    - Predicted final bend angle after springback
    - Springback Factor Ks = final_angle / (theoretical_commanded_angle)

MODEL REUSABILITY:
    ✓ Trained .pkl files CAN be shared across systems
    ✓ Requirements: Same Python version (3.10-3.12) and scikit-learn (1.3.x)
    ✓ Cross-platform compatible (Windows/Mac/Linux)
    ✓ See MODEL_REUSABILITY.md for details
"""

import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import sys


class SpringbackSurrogateModel:
    """
    Forward surrogate model for springback prediction
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize RandomForestRegressor for springback modeling
        
        Why RandomForest:
        - Handles non-linear elastic-plastic response
        - Robust to FEM noise and discretization errors
        - No hyperparameter tuning required for small datasets
        - Provides uncertainty estimates via tree variance
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True
        )
        
        self.is_trained = False
        self.feature_names = ['rx', 'ryr_in_medians']
        self.target_name = 'bend_angle'
        
    def load_data(self, csv_path):
        """Load and prepare training data"""
        df = pd.read_csv(csv_path)
        
        X = df[self.feature_names].values
        y = df[self.target_name].values
        
        return X, y, df
        
    def train(self, X, y, test_size=0.2):
        """
        Train the springback surrogate model
        
        Returns:
            dict: Training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'cv_score': cross_val_score(self.model, X, y, cv=3, 
                                       scoring='neg_mean_absolute_error').mean()
        }
        
        return metrics
    
    def predict(self, rx, ryr):
        """
        Predict final bend angle after springback
        
        Args:
            rx: Tool rotation angle (degrees, negative)
            ryr: Secondary rotation parameter
            
        Returns:
            float: Predicted final bend angle (degrees)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X = np.array([[rx, ryr]])
        prediction = self.model.predict(X)[0]
        
        return prediction
    
    def calculate_springback_factor(self, rx, ryr, target_angle):
        """
        Calculate springback correction factor
        
        Engineering Definition:
            Ks = actual_final_angle / target_angle
            
            If Ks < 1: Springback reduces angle (most common)
            If Ks > 1: Material hardens beyond expectation
            If Ks ≈ 1: Perfect forming (rare)
            
        Args:
            rx: Tool rotation
            ryr: Secondary parameter
            target_angle: Desired final angle
            
        Returns:
            tuple: (predicted_angle, Ks, required_overbend)
        """
        predicted_angle = self.predict(rx, ryr)
        
        # Springback factor
        Ks = predicted_angle / target_angle if target_angle != 0 else 1.0
        
        # Required overbend to achieve target
        required_overbend = target_angle / Ks if Ks != 0 else target_angle
        
        return predicted_angle, Ks, required_overbend
    
    def get_feature_importance(self):
        """Return feature importance for engineering interpretation"""
        if not self.is_trained:
            return None
            
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def save(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        joblib.dump(self.model, filepath)
        print(f"✓ Model saved: {filepath}")
        
    def load(self, filepath):
        """Load pre-trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"✓ Model loaded: {filepath}")


# ============================================================================
# USER-FRIENDLY PATH PROMPTING
# ============================================================================

def get_save_path_from_user(default_filename="springback_model.pkl"):
    """
    Prompt user for model save location with clear instructions
    
    This makes the training process beginner-friendly by:
    - Showing example paths
    - Allowing default location (just press ENTER)
    - Validating and creating directories
    - Confirming final save location
    """
    
    # Get script directory to determine default path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, default_filename)
    
    # Show clear instructions
    print("\n" + "="*70)
    print("MODEL SAVE LOCATION")
    print("="*70)
    print("\nWhere should the trained springback model be saved?")
    print("\n📁 Example paths:")
    print(f"   Windows: C:\\Users\\YourName\\Desktop\\project_root\\models\\{default_filename}")
    print(f"   Mac/Linux: /home/username/project_root/models/{default_filename}")
    print("\n💡 Tip: Copy the path from your file explorer/finder")
    print(f"\n⏎  Default: Press ENTER to use:")
    print(f"   {default_path}")
    print("\n" + "="*70)
    
    # Get user input
    user_input = input("\nEnter full file path (or press ENTER for default): ").strip()
    
    # Use default if empty
    if not user_input:
        save_path = default_path
        print(f"\n✓ Using default location")
    else:
        save_path = user_input
    
    # Normalize path
    save_path = os.path.abspath(save_path)
    
    # Validate and create directory
    save_dir = os.path.dirname(save_path)
    
    if not os.path.exists(save_dir):
        print(f"\n📁 Directory does not exist: {save_dir}")
        create = input("   Create this directory? (y/n): ").strip().lower()
        
        if create == 'y':
            try:
                os.makedirs(save_dir, exist_ok=True)
                print(f"✓ Directory created: {save_dir}")
            except Exception as e:
                print(f"❌ Error creating directory: {e}")
                print("   Please check the path and try again")
                sys.exit(1)
        else:
            print("   Training cancelled")
            sys.exit(0)
    
    # Confirm
    print(f"\n✓ Model will be saved to:")
    print(f"   {save_path}")
    print()
    
    return save_path


def train_and_save_model(data_path, model_save_path=None):
    """
    Complete training pipeline for springback model
    """
    print("="*70)
    print("MODEL 1: SPRINGBACK SURROGATE TRAINING")
    print("="*70)
    
    # Get save path if not provided
    if model_save_path is None:
        model_save_path = get_save_path_from_user("springback_model.pkl")
    
    # Initialize model
    model = SpringbackSurrogateModel(n_estimators=100, max_depth=10)
    
    # Load data
    print("\n📊 Loading dataset...")
    X, y, df = model.load_data(data_path)
    print(f"   Samples: {len(X)}")
    print(f"   Features: {model.feature_names}")
    print(f"   Target: {model.target_name}")
    
    # Train
    print("\n🔧 Training RandomForest surrogate...")
    metrics = model.train(X, y, test_size=0.2)
    
    print(f"\n✓ Training Complete!")
    print(f"   Train MAE: {metrics['train_mae']:.3f}°")
    print(f"   Test MAE:  {metrics['test_mae']:.3f}°")
    print(f"   Train R²:  {metrics['train_r2']:.4f}")
    print(f"   Test R²:   {metrics['test_r2']:.4f}")
    print(f"   CV Score:  {-metrics['cv_score']:.3f}°")
    
    # Feature importance
    print(f"\n📊 Feature Importance:")
    for feature, importance in model.get_feature_importance().items():
        print(f"   {feature}: {importance:.4f}")
    
    # Save model
    model.save(model_save_path)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel saved at:")
    print(f"   {model_save_path}")
    print(f"\nFile size: {os.path.getsize(model_save_path) / 1024:.1f} KB")
    print("\nYou can now:")
    print("   1. Run the Streamlit app (cd ../ui && streamlit run app.py)")
    print("   2. Share this .pkl file with others (see MODEL_REUSABILITY.md)")
    print("   3. Retrain anytime by running this script again")
    print()
    
    return model, metrics


if __name__ == "__main__":
    # Find data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'bending_dataset.csv')
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        print("\nPlease ensure bending_dataset.csv is in the data/ directory")
        sys.exit(1)
    
    # Train model
    model, metrics = train_and_save_model(data_path)
