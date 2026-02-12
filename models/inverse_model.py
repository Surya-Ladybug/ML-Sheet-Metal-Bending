"""
MODEL 2: INVERSE REGRESSION MODEL
===================================

Purpose:
    Solve the engineering inverse problem: (target_angle, rx) → (predicted_ryr, predicted_radius)
    
Engineering Justification:
    Why Analytical Inversion is Invalid:
    
    1. FORWARD MODEL IS NON-UNIQUE:
       - Multiple (ryr, rx) combinations can produce same bend_angle
       - Material path-dependency makes inversion ill-posed
       - Springback is history-dependent, not a bijective function
       
    2. NO CLOSED-FORM PHYSICS:
       - Elastic-plastic contact is solved numerically (FEM)
       - No analytical bending equation for complex loading
       - Tool-material interaction is non-linear PDE system
       
    3. ML INVERSION APPROACH:
       - Learn inverse mapping from data: (angle, rx) → (ryr, radius)
       - Captures empirical relationships from FEM results
       - Provides "most likely" solution given constraints
       - Enforces rx bounds during inference
       
    Why Two Models Required:
    - Model 1 (forward): Physics simulation surrogate
    - Model 2 (inverse): Engineering design tool
    - They solve different mathematical problems
    - Neither can replace the other

Output:
    - Predicted ryr value for target angle
    - Predicted minimal inner bend radius

MODEL REUSABILITY:
    ✓ Trained .pkl files CAN be shared across systems
    ✓ Requirements: Same Python version (3.10-3.12) and scikit-learn (1.3.x)
    ✓ Cross-platform compatible (Windows/Mac/Linux)
    ✓ See MODEL_REUSABILITY.md for details
"""

import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import sys


class InverseRegressionModel:
    """
    Inverse regression for engineering design:
    Given desired angle and rx constraint → predict ryr and radius
    """
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
        """
        Initialize GradientBoosting for inverse regression
        
        Why GradientBoosting:
        - Better for multi-target regression (ryr AND radius)
        - Learns sequential corrections to predictions
        - Handles correlated outputs naturally
        - More stable than RandomForest for inverse problems
        """
        base_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        # Multi-output wrapper for predicting both ryr and radius
        self.model = MultiOutputRegressor(base_model)
        
        self.is_trained = False
        self.feature_names = ['bend_angle', 'rx']
        self.target_names = ['ryr_in_medians', 'radius_in_mm']
        
        # RX constraints
        self.RX_UPPER_LIMIT = -10
        self.RX_LOWER_LIMIT = -140
        
    def load_data(self, csv_path):
        """Load and prepare inverse regression data"""
        df = pd.read_csv(csv_path)
        
        # Features: (target_angle, rx_constraint)
        X = df[self.feature_names].values
        
        # Targets: (ryr, radius) - both must be predicted
        y = df[self.target_names].values
        
        return X, y, df
        
    def train(self, X, y, test_size=0.2):
        """
        Train inverse regression model
        
        Returns:
            dict: Training metrics for both outputs
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
        
        # Metrics for each output
        metrics = {
            'ryr': {
                'train_mae': mean_absolute_error(y_train[:, 0], y_pred_train[:, 0]),
                'test_mae': mean_absolute_error(y_test[:, 0], y_pred_test[:, 0]),
                'train_r2': r2_score(y_train[:, 0], y_pred_train[:, 0]),
                'test_r2': r2_score(y_test[:, 0], y_pred_test[:, 0])
            },
            'radius': {
                'train_mae': mean_absolute_error(y_train[:, 1], y_pred_train[:, 1]),
                'test_mae': mean_absolute_error(y_test[:, 1], y_pred_test[:, 1]),
                'train_r2': r2_score(y_train[:, 1], y_pred_train[:, 1]),
                'test_r2': r2_score(y_test[:, 1], y_pred_test[:, 1])
            }
        }
        
        return metrics
    
    def predict(self, target_angle, rx):
        """
        Inverse prediction: What ryr and radius to expect?
        
        Args:
            target_angle: Desired final bend angle (degrees)
            rx: Tool rotation constraint (degrees, negative)
            
        Returns:
            tuple: (predicted_ryr, predicted_radius_mm)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Enforce RX bounds
        if rx > self.RX_UPPER_LIMIT or rx < self.RX_LOWER_LIMIT:
            raise ValueError(
                f"RX out of bounds! "
                f"Must be between {self.RX_LOWER_LIMIT}° and {self.RX_UPPER_LIMIT}°"
            )
        
        X = np.array([[target_angle, rx]])
        predictions = self.model.predict(X)[0]
        
        predicted_ryr = predictions[0]
        predicted_radius = predictions[1]
        
        return predicted_ryr, predicted_radius
    
    def validate_rx_bounds(self, rx):
        """Check if RX is within valid range"""
        return self.RX_LOWER_LIMIT <= rx <= self.RX_UPPER_LIMIT
    
    def get_rx_constraint_info(self):
        """Return RX constraint information for UI"""
        return {
            'upper_limit': self.RX_UPPER_LIMIT,
            'lower_limit': self.RX_LOWER_LIMIT,
            'units': 'degrees',
            'reason': 'Physical tool rotation limits based on FEM simulation range'
        }
    
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

def get_save_path_from_user(default_filename="inverse_model.pkl"):
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
    print("\nWhere should the trained inverse model be saved?")
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
    Complete training pipeline for inverse model
    """
    print("="*70)
    print("MODEL 2: INVERSE REGRESSION TRAINING")
    print("="*70)
    
    # Get save path if not provided
    if model_save_path is None:
        model_save_path = get_save_path_from_user("inverse_model.pkl")
    
    # Initialize model
    model = InverseRegressionModel(n_estimators=100, max_depth=5, learning_rate=0.1)
    
    # Load data
    print("\n📊 Loading dataset...")
    X, y, df = model.load_data(data_path)
    print(f"   Samples: {len(X)}")
    print(f"   Features: {model.feature_names}")
    print(f"   Targets: {model.target_names}")
    
    # Train
    print("\n🔧 Training GradientBoosting inverse model...")
    metrics = model.train(X, y, test_size=0.2)
    
    print(f"\n✓ Training Complete!")
    print(f"\n   RYR Prediction:")
    print(f"      Train MAE: {metrics['ryr']['train_mae']:.4f}")
    print(f"      Test MAE:  {metrics['ryr']['test_mae']:.4f}")
    print(f"      Train R²:  {metrics['ryr']['train_r2']:.4f}")
    print(f"      Test R²:   {metrics['ryr']['test_r2']:.4f}")
    
    print(f"\n   Radius Prediction:")
    print(f"      Train MAE: {metrics['radius']['train_mae']:.3f} mm")
    print(f"      Test MAE:  {metrics['radius']['test_mae']:.3f} mm")
    print(f"      Train R²:  {metrics['radius']['train_r2']:.4f}")
    print(f"      Test R²:   {metrics['radius']['test_r2']:.4f}")
    
    # RX constraints
    print(f"\n🔒 RX Constraints:")
    info = model.get_rx_constraint_info()
    print(f"   Upper limit: {info['upper_limit']}°")
    print(f"   Lower limit: {info['lower_limit']}°")
    print(f"   Reason: {info['reason']}")
    
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
