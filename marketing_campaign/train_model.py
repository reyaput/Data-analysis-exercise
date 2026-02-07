import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve
)
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class CampaignSuccessPredictor:
    """
    Main class for training and evaluating campaign success prediction models.
    """
    
    def __init__(self, data_path):
        """
        Initialize the predictor.
        
        Parameters:
        -----------
        data_path : str
            Path to the cleaned marketing campaign CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load the marketing campaign data."""
        print("=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\n✓ Data loaded successfully!")
        print(f"  - Total records: {len(self.df):,}")
        print(f"  - Total features: {len(self.df.columns)}")
        print(f"\nDataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        
    def engineer_features(self):
        """
        Create engineered features as per requirements.
        
        Features to create:
        1. CPC (Cost Per Click) = Acquisition_Cost / Clicks
        2. CTR (Click-Through Rate) = Clicks / Impressions
        3. CPM (Cost Per Thousand Impressions) = (Acquisition_Cost / Impressions) * 1000
        4. Engagement per Click = Engagement_Score / Clicks
        5. ROI Efficiency = ROI / (Acquisition_Cost / 1000)
        """
        print("\n" + "=" * 80)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 80)
        
        # Create a copy to avoid SettingWithCopyWarning
        df = self.df.copy()
        
        # 1. Cost Per Click (CPC)
        df['CPC'] = df['Acquisition_Cost'] / df['Clicks'].replace(0, 1)
        
        # 2. Click-Through Rate (CTR)
        df['CTR'] = df['Clicks'] / df['Impressions'].replace(0, 1)
        
        # 3. Cost Per Thousand Impressions (CPM)
        df['CPM'] = (df['Acquisition_Cost'] / df['Impressions'].replace(0, 1)) * 1000
        
        # 4. Engagement per Click
        df['Engagement_per_Click'] = df['Engagement_Score'] / df['Clicks'].replace(0, 1)
        
        # 5. ROI Efficiency
        df['ROI_Efficiency'] = df['ROI'] / (df['Acquisition_Cost'] / 1000)
        
        # Additional useful features
        df['Click_to_Engagement_Ratio'] = df['Clicks'] / df['Engagement_Score'].replace(0, 1)
        df['Impression_to_Click_Efficiency'] = df['Impressions'] / df['Clicks'].replace(0, 1)
        
        df['ROI_x_Conversion'] = df['ROI'] * df['Conversion_Rate']
        df['Budget_Efficiency'] = df['ROI'] / np.log1p(df['Acquisition_Cost'])
        df['Quality_Score'] = df['CTR'] * df['Engagement_Score'] / df['CPC']
        self.df = df
        
        print("\n✓ Engineered Features Created:")
        print(f"  1. CPC (Cost Per Click)")
        print(f"  2. CTR (Click-Through Rate)")
        print(f"  3. CPM (Cost Per Thousand Impressions)")
        print(f"  4. Engagement_per_Click")
        print(f"  5. ROI_Efficiency")
        print(f"  6. Click_to_Engagement_Ratio (bonus)")
        print(f"  7. Impression_to_Click_Efficiency (bonus)")
        
        # Show sample of engineered features
        print("\nSample of Engineered Features:")
        print(self.df[['CPC', 'CTR', 'CPM', 'Engagement_per_Click', 'ROI_Efficiency']].head())
        
    def create_target_variable(self):
        """
        Create target variable based on success criteria.
        
        Success = 1: ROI >= 5 AND Conversion_Rate >= 0.08
        Fail = 0: Otherwise
        """
        print("\n" + "=" * 80)
        print("STEP 3: CREATE TARGET VARIABLE")
        print("=" * 80)
        
        # Create success flag
        self.df['Campaign_Success'] = (
            (self.df['ROI'] >= 5) & 
            (self.df['Conversion_Rate'] >= 0.08)
        ).astype(int)
        
        # Distribution analysis
        success_count = self.df['Campaign_Success'].sum()
        total_count = len(self.df)
        success_rate = (success_count / total_count) * 100
        
        print(f"\n✓ Target Variable Created: Campaign_Success")
        print(f"\nSuccess Criteria:")
        print(f"  - ROI >= 5")
        print(f"  - Conversion_Rate >= 0.08")
        print(f"\nTarget Distribution:")
        print(f"  - Success (1): {success_count:,} ({success_rate:.2f}%)")
        print(f"  - Fail (0): {total_count - success_count:,} ({100 - success_rate:.2f}%)")
        
        if success_rate < 20 or success_rate > 80:
            print(f"\n⚠ Warning: Imbalanced dataset detected!")
            print(f"  Consider using class weights or resampling techniques.")
        
    def prepare_features(self):
        """
        Prepare features for modeling:
        - Select relevant features
        - Encode categorical variables
        - Handle missing values
        """
        print("\n" + "=" * 80)
        print("STEP 4: FEATURE PREPARATION")
        print("=" * 80)
        
        # Features to exclude
        exclude_cols = [
            'Campaign_ID', 
            'Date', 
            'Company',
            'Campaign_Success',  # Target variable
            'ROI',  # Used to create target
            'Conversion_Rate'  # Used to create target
        ]
        
        # Numerical features
        numerical_features = [
            'Acquisition_Cost', 'Clicks', 'Impressions', 'Engagement_Score',
            'Duration_Days', 'CPC', 'CTR', 'CPM', 'Engagement_per_Click',
            'ROI_Efficiency', 'Click_to_Engagement_Ratio', 
            'Impression_to_Click_Efficiency'
        ]
        
        # Categorical features
        categorical_features = [
            'Campaign_Type', 'Target_Audience', 'Channel_Used',
            'Location', 'Language', 'Customer_Segment', 'Gender', 'Age_Group'
        ]
        
        # Create feature dataframe
        df_features = self.df.copy()
        
        # Encode categorical variables
        print("\n✓ Encoding Categorical Variables...")
        for col in categorical_features:
            if col in df_features.columns:
                le = LabelEncoder()
                df_features[col + '_encoded'] = le.fit_transform(df_features[col].astype(str))
                self.label_encoders[col] = le
                print(f"  - {col}: {len(le.classes_)} unique values")
        
        # Select final features
        encoded_categorical = [col + '_encoded' for col in categorical_features]
        feature_columns = numerical_features + encoded_categorical
        
        # Remove any columns that don't exist
        feature_columns = [col for col in feature_columns if col in df_features.columns]
        
        X = df_features[feature_columns]
        y = df_features['Campaign_Success']
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print(f"\n⚠ Handling {X.isnull().sum().sum()} missing values...")
            X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_names = feature_columns
        
        print(f"\n✓ Feature Preparation Complete!")
        print(f"  - Total features: {len(feature_columns)}")
        print(f"  - Numerical features: {len(numerical_features)}")
        print(f"  - Categorical features (encoded): {len(categorical_features)}")
        
        return X, y
    
    def split_and_scale(self, X, y, test_size=0.2):
        """
        Split data into train/test sets and scale features.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series
            Target variable
        test_size : float
            Proportion of data for testing
        """
        print("\n" + "=" * 80)
        print("STEP 5: TRAIN-TEST SPLIT & SCALING")
        print("=" * 80)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"\n✓ Data Split Complete!")
        print(f"  - Training set: {len(self.X_train):,} samples ({(1-test_size)*100:.0f}%)")
        print(f"  - Test set: {len(self.X_test):,} samples ({test_size*100:.0f}%)")
        print(f"\n  Training set - Success rate: {self.y_train.mean()*100:.2f}%")
        print(f"  Test set - Success rate: {self.y_test.mean()*100:.2f}%")
        
    def train_models(self):
        """
        Train multiple models:
        1. Logistic Regression
        2. Random Forest
        3. XGBoost
        """
        print("\n" + "=" * 80)
        print("STEP 6: MODEL TRAINING")
        print("=" * 80)
        
        # Calculate class weights for imbalanced data
        class_counts = self.y_train.value_counts()
        class_weight = {
            0: len(self.y_train) / (2 * class_counts[0]),
            1: len(self.y_train) / (2 * class_counts[1])
        }
        
        # 1. Logistic Regression
        print("\n[1/3] Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight=class_weight,
            C=0.1
        )
        lr_model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr_model
        print("  ✓ Logistic Regression trained")
        
        # 2. Random Forest
        print("\n[2/3] Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        print("  ✓ Random Forest trained")
        
        # 3. XGBoost
        print("\n[3/3] Training XGBoost...")
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = class_counts[0] / class_counts[1]
        
        # Di train_model.py, ubah hyperparameter XGBoost:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,          # Naikkan dari 200
            max_depth=8,               # Naikkan dari 6
            learning_rate=0.03,        # Turunkan dari 0.05 (lebih conservative)
            subsample=0.85,            # Naikkan dari 0.8
            colsample_bytree=0.85,     # Naikkan dari 0.8
            min_child_weight=3,        # Tambahkan (regularization)
            gamma=0.1,                 # Tambahkan (regularization)
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE
        )
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb_model
        print("  ✓ XGBoost trained")
        
        # 4. Ensemble dengan Voting Classifier
        print("\n[4/4] Training Voting Ensemble...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('lr', lr_model), 
                ('rf', rf_model), 
                ('xgb', xgb_model)
            ],
            voting='soft',
            weights=[1, 2, 3]  # XGBoost dapat bobot lebih tinggi
        )
        ensemble_model.fit(self.X_train, self.y_train)
        self.models['Voting Ensemble'] = ensemble_model
        print("  ✓ Voting Ensemble trained")
        
        print(f"\n✓ All {len(self.models)} models trained successfully!")
        
    def evaluate_models(self):
        """
        Evaluate all models using multiple metrics.
        Primary metrics: F1-Score and ROC-AUC
        """
        print("\n" + "=" * 80)
        print("STEP 7: MODEL EVALUATION")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            print(f"\n{'=' * 80}")
            print(f"Evaluating: {model_name}")
            print(f"{'=' * 80}")
            
            # Predictions dengan custom threshold 0.6
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.6).astype(int)  # Custom threshold: 0.6 instead of 0.5
            
            # Calculate metrics
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation F1 score
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                scoring='f1'
            )
            
            # Store results
            self.results[model_name] = {
                'f1_score': f1,
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print results
            print(f"\nPrimary Metrics:")
            print(f"  F1-Score:     {f1:.4f} {'✓ PASS' if f1 > 0.80 else '✗ FAIL (target: >0.80)'}")
            print(f"  ROC-AUC:      {roc_auc:.4f} {'✓ PASS' if roc_auc > 0.85 else '✗ FAIL (target: >0.85)'}")
            
            print(f"\nAdditional Metrics:")
            print(f"  Precision:    {precision:.4f}")
            print(f"  Recall:       {recall:.4f}")
            print(f"  Accuracy:     {accuracy:.4f}")
            
            print(f"\nCross-Validation (5-fold):")
            print(f"  F1-Score:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
            print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
        
        # Select best model based on F1-Score and ROC-AUC
        self._select_best_model()
        
    def _select_best_model(self):
        """Select the best model based on F1-Score and ROC-AUC."""
        print("\n" + "=" * 80)
        print("MODEL SELECTION")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'F1-Score': [self.results[m]['f1_score'] for m in self.results.keys()],
            'ROC-AUC': [self.results[m]['roc_auc'] for m in self.results.keys()],
            'Precision': [self.results[m]['precision'] for m in self.results.keys()],
            'Recall': [self.results[m]['recall'] for m in self.results.keys()],
            'Accuracy': [self.results[m]['accuracy'] for m in self.results.keys()]
        })
        
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
        
        # Composite score: weighted average of F1 and ROC-AUC
        comparison['Composite_Score'] = (
            comparison['F1-Score'] * 0.5 + 
            comparison['ROC-AUC'] * 0.5
        )
        
        # Select best model
        best_idx = comparison['Composite_Score'].idxmax()
        self.best_model_name = comparison.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n" + "=" * 80)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"=" * 80)
        print(f"  F1-Score:        {comparison.loc[best_idx, 'F1-Score']:.4f}")
        print(f"  ROC-AUC:         {comparison.loc[best_idx, 'ROC-AUC']:.4f}")
        print(f"  Composite Score: {comparison.loc[best_idx, 'Composite_Score']:.4f}")
        
        # Check if model meets requirements
        meets_f1 = comparison.loc[best_idx, 'F1-Score'] > 0.80
        meets_roc = comparison.loc[best_idx, 'ROC-AUC'] > 0.85
        
        if meets_f1 and meets_roc:
            print(f"\n✓ Model meets all performance requirements!")
        else:
            print(f"\n⚠ Warning: Model does not meet all requirements:")
            if not meets_f1:
                print(f"  - F1-Score: {comparison.loc[best_idx, 'F1-Score']:.4f} (target: >0.80)")
            if not meets_roc:
                print(f"  - ROC-AUC: {comparison.loc[best_idx, 'ROC-AUC']:.4f} (target: >0.85)")
        
    def analyze_feature_importance(self):
        """
        Analyze and display feature importance from the best model.
        """
        print("\n" + "=" * 80)
        print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        # Get feature importance based on model type
        if self.best_model_name == 'Logistic Regression':
            # For Logistic Regression, use coefficient absolute values
            importance = np.abs(self.best_model.coef_[0])
        else:
            # For tree-based models (Random Forest, XGBoost)
            importance = self.best_model.feature_importances_
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Display top 10 features
        print("\n🔝 Top 10 Most Important Features:")
        print("=" * 80)
        for idx, row in feature_importance.head(10).iterrows():
            print(f"{row['Feature']:40s} | {row['Importance']:.6f}")
        
        # Top 5 features with insights
        print("\n" + "=" * 80)
        print("📊 TOP 5 FEATURES - DETAILED INSIGHTS")
        print("=" * 80)
        
        top_5_features = feature_importance.head(5)
        
        for idx, (i, row) in enumerate(top_5_features.iterrows(), 1):
            feature = row['Feature']
            importance = row['Importance']
            
            print(f"\n{idx}. {feature}")
            print(f"   Importance Score: {importance:.6f}")
            print(f"   Insight: {self._get_feature_insight(feature)}")
        
        # Save feature importance
        self.feature_importance = feature_importance
        
        return feature_importance
    
    def _get_feature_insight(self, feature):
        """Generate insight for each important feature."""
        insights = {
            'ROI_Efficiency': 'Measures how much ROI is generated per $1000 spent. Higher efficiency indicates better campaign performance and resource utilization.',
            'CTR': 'Click-Through Rate indicates ad relevance and engagement. Higher CTR means the campaign messaging resonates with the target audience.',
            'CPC': 'Cost Per Click reflects campaign efficiency. Lower CPC with high conversion indicates optimized targeting and ad placement.',
            'CPM': 'Cost Per Thousand Impressions shows brand awareness cost. Lower CPM with high engagement indicates efficient reach.',
            'Engagement_per_Click': 'Measures content quality. Higher engagement per click indicates compelling content that drives deeper interaction.',
            'Engagement_Score': 'Overall audience interaction metric. Higher scores correlate with brand affinity and conversion potential.',
            'Acquisition_Cost': 'Total campaign spend. Must be balanced with ROI and conversion to ensure profitability.',
            'Clicks': 'Direct measure of campaign attraction. More clicks with high conversion rate indicates effective targeting.',
            'Conversion_Rate': 'Primary success indicator. Shows percentage of clicks that convert to desired action.',
            'Channel_Used_encoded': 'Different channels perform differently. This helps identify which platforms deliver best results for your audience.',
            'Campaign_Type_encoded': 'Campaign format (Email, Social, Display, etc.) significantly impacts success. Some types work better for specific goals.',
            'Customer_Segment_encoded': 'Different segments respond differently. Understanding segment preferences is crucial for targeting.',
            'Target_Audience_encoded': 'Demographics matter. This feature helps identify which age-gender combinations are most responsive.',
            'Duration_Days': 'Campaign duration affects fatigue and reach. Optimal duration balances awareness and ad fatigue.',
            'Impressions': 'Total reach metric. High impressions with low engagement may indicate targeting issues.'
        }
        
        return insights.get(feature, 'Important predictor of campaign success based on historical data patterns.')
    
    def save_model(self, output_dir='models'):
        """
        Save the best model, scaler, encoders, and metadata.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model artifacts
        """
        import os
        
        print("\n" + "=" * 80)
        print("STEP 9: SAVING MODEL ARTIFACTS")
        print("=" * 80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save the best model
        model_path = os.path.join(output_dir, 'best_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"\n✓ Saved best model: {model_path}")
        
        # 2. Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Saved scaler: {scaler_path}")
        
        # 3. Save label encoders
        encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"✓ Saved label encoders: {encoders_path}")
        
        # 4. Save feature names
        features_path = os.path.join(output_dir, 'feature_names.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"✓ Saved feature names: {features_path}")
        
        # 5. Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performance': {
                'f1_score': self.results[self.best_model_name]['f1_score'],
                'roc_auc': self.results[self.best_model_name]['roc_auc'],
                'precision': self.results[self.best_model_name]['precision'],
                'recall': self.results[self.best_model_name]['recall'],
                'accuracy': self.results[self.best_model_name]['accuracy']
            },
            'feature_count': len(self.feature_names),
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'success_criteria': {
                'roi_threshold': 5,
                'conversion_rate_threshold': 0.08
            }
        }
        
        metadata_path = os.path.join(output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Saved metadata: {metadata_path}")
        
        # 6. Save feature importance
        feature_importance_path = os.path.join(output_dir, 'feature_importance.csv')
        self.feature_importance.to_csv(feature_importance_path, index=False)
        print(f"✓ Saved feature importance: {feature_importance_path}")
        
        # 7. Save all model results for comparison
        results_comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'F1_Score': [self.results[m]['f1_score'] for m in self.results.keys()],
            'ROC_AUC': [self.results[m]['roc_auc'] for m in self.results.keys()],
            'Precision': [self.results[m]['precision'] for m in self.results.keys()],
            'Recall': [self.results[m]['recall'] for m in self.results.keys()],
            'Accuracy': [self.results[m]['accuracy'] for m in self.results.keys()],
            'CV_F1_Mean': [self.results[m]['cv_f1_mean'] for m in self.results.keys()],
            'CV_F1_Std': [self.results[m]['cv_f1_std'] for m in self.results.keys()]
        })
        
        results_path = os.path.join(output_dir, 'model_comparison.csv')
        results_comparison.to_csv(results_path, index=False)
        print(f"✓ Saved model comparison: {results_path}")
        
        print(f"\n{'=' * 80}")
        print(f"✓ All artifacts saved successfully to '{output_dir}/' directory!")
        print(f"{'=' * 80}")
        
    def generate_visualizations(self, output_dir='models'):
        """Generate and save visualization plots."""
        import os
        
        print("\n" + "=" * 80)
        print("STEP 10: GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        top_20 = self.feature_importance.head(20)
        plt.barh(range(len(top_20)), top_20['Importance'])
        plt.yticks(range(len(top_20)), top_20['Feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top 20 Feature Importance - {self.best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved feature importance plot: {plot_path}")
        
        # 2. ROC Curves for all models
        plt.figure(figsize=(10, 8))
        
        for model_name in self.results.keys():
            y_pred_proba = self.results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = self.results[model_name]['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        roc_path = os.path.join(output_dir, 'roc_curves.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved ROC curves: {roc_path}")
        
        # 3. Confusion Matrix for best model
        y_pred = self.results[self.best_model_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved confusion matrix: {cm_path}")
        
        # 4. Model Comparison Bar Chart
        comparison = pd.DataFrame({
            'Model': list(self.results.keys()),
            'F1-Score': [self.results[m]['f1_score'] for m in self.results.keys()],
            'ROC-AUC': [self.results[m]['roc_auc'] for m in self.results.keys()]
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(comparison))
        width = 0.35
        
        ax.bar(x - width/2, comparison['F1-Score'], width, label='F1-Score', alpha=0.8)
        ax.bar(x + width/2, comparison['ROC-AUC'], width, label='ROC-AUC', alpha=0.8)
        
        ax.axhline(y=0.80, color='r', linestyle='--', alpha=0.5, label='F1 Target (0.80)')
        ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='AUC Target (0.85)')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['Model'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        comparison_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved model comparison: {comparison_path}")
        
        print(f"\n{'=' * 80}")
        print(f"✓ All visualizations saved successfully!")
        print(f"{'=' * 80}")
    
    def run_pipeline(self):
        """Execute the complete training pipeline."""
        print("\n" + "=" * 80)
        print("MARKETING CAMPAIGN SUCCESS PREDICTION - TRAINING PIPELINE")
        print("=" * 80)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute pipeline steps
        self.load_data()
        self.engineer_features()
        self.create_target_variable()
        X, y = self.prepare_features()
        self.split_and_scale(X, y)
        self.train_models()
        self.evaluate_models()
        self.analyze_feature_importance()
        self.save_model()
        self.generate_visualizations()
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nNext Steps:")
        print(f"  1. Review model performance metrics")
        print(f"  2. Analyze feature importance insights")
        print(f"  3. Build Streamlit app for deployment")
        print(f"  4. Test predictions with new campaign data")
        print("=" * 80)


def main():
    """Main execution function."""
    # Path to data
    data_path = 'marketing_campaign_cleaned.csv'
    
    # Initialize and run pipeline
    predictor = CampaignSuccessPredictor(data_path)
    predictor.run_pipeline()
    
    print("\n" + "=" * 80)
    print("📊 QUICK SUMMARY")
    print("=" * 80)
    print(f"\nBest Model: {predictor.best_model_name}")
    print(f"F1-Score: {predictor.results[predictor.best_model_name]['f1_score']:.4f}")
    print(f"ROC-AUC: {predictor.results[predictor.best_model_name]['roc_auc']:.4f}")
    print(f"\nTop 5 Important Features:")
    for idx, row in predictor.feature_importance.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']}")
    
    print("\n" + "=" * 80)
    print("Model artifacts saved in 'models/' directory:")
    print("  - best_model.pkl")
    print("  - scaler.pkl")
    print("  - label_encoders.pkl")
    print("  - feature_names.pkl")
    print("  - model_metadata.json")
    print("  - feature_importance.csv")
    print("  - model_comparison.csv")
    print("  - feature_importance.png")
    print("  - roc_curves.png")
    print("  - confusion_matrix.png")
    print("  - model_comparison.png")
    print("=" * 80)


if __name__ == "__main__":
    main()