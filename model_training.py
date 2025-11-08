import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class LoanPredictor:
    def __init__(self, data_path='cleaned_loan_data.csv'):
        self.data = pd.read_csv(data_path)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_model = None
        
    def prepare_data(self):
        """Prepare data for model training"""
        df = self.data.copy()
        
        # Separate features and target
        X = df.drop('Personal Loan', axis=1)
        y = df['Personal Loan']
        
        # Get feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        numerical_cols = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        print("✅ Data preparation completed!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features: {self.feature_names}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        # Train and evaluate models
        best_score = 0
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            print(f"CV AUC Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test AUC Score: {auc_score:.4f}")
            print(classification_report(y_test, y_pred))
            
            self.models[name] = {
                'model': model,
                'cv_score': cv_scores.mean(),
                'auc_score': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Update best model
            if auc_score > best_score:
                best_score = auc_score
                self.best_model = name
        
        print(f"\n🎯 Best Model: {self.best_model} with AUC: {best_score:.4f}")
    
    def plot_model_performance(self):
        """Plot model performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. AUC Scores comparison
        model_names = list(self.models.keys())
        auc_scores = [self.models[name]['auc_score'] for name in model_names]
        
        axes[0, 0].barh(model_names, auc_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0, 0].set_xlabel('AUC Score')
        axes[0, 0].set_title('Model Performance (AUC Score)')
        for i, v in enumerate(auc_scores):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # 2. Confusion Matrix for best model
        best_model_data = self.models[self.best_model]
        cm = confusion_matrix(self.y_test, best_model_data['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {self.best_model}')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. ROC Curves
        for name, data in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, data['probabilities'])
            auc_score = data['auc_score']
            axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Feature Importance for best model
        best_model = self.models[self.best_model]['model']
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            axes[1, 1].barh(feature_imp['feature'], feature_imp['importance'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title(f'Feature Importance - {self.best_model}')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self):
        """Get feature importance for business insights"""
        best_model = self.models[self.best_model]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\n=== FEATURE IMPORTANCE ===")
            for _, row in importance_df.iterrows():
                print(f"{row['Feature']:20} : {row['Importance']:.4f}")
            
            return importance_df
        else:
            # For linear models, use coefficients
            if hasattr(best_model, 'coef_'):
                coef_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Coefficient': best_model.coef_[0]
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                print("\n=== FEATURE COEFFICIENTS ===")
                for _, row in coef_df.iterrows():
                    print(f"{row['Feature']:20} : {row['Coefficient']:+.4f}")
                
                return coef_df
    
    def calculate_business_metrics(self):
        """Calculate business-relevant metrics"""
        best_model_data = self.models[self.best_model]
        y_pred = best_model_data['predictions']
        y_true = self.y_test
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Business metrics
        approval_rate = (tp + fp) / len(y_pred) * 100
        true_approval_rate = tp / (tp + fn) * 100
        false_approval_rate = fp / (fp + tn) * 100
        
        print("\n=== BUSINESS METRICS ===")
        print(f"Overall Approval Rate: {approval_rate:.1f}%")
        print(f"True Approval Rate (Sensitivity): {true_approval_rate:.1f}%")
        print(f"False Approval Rate: {false_approval_rate:.1f}%")
        print(f"Correct Approvals (TP): {tp}")
        print(f"Missed Opportunities (FN): {fn}")
        print(f"Risk Approvals (FP): {fp}")
        print(f"Correct Rejections (TN): {tn}")
        
        return {
            'approval_rate': approval_rate,
            'true_approval_rate': true_approval_rate,
            'false_approval_rate': false_approval_rate,
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn
        }
    
    def save_models(self):
        """Save trained models and scaler"""
        # Save the best model
        with open('best_loan_model.pkl', 'wb') as f:
            pickle.dump(self.models[self.best_model]['model'], f)
        
        # Save scaler
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print("✅ Models and artifacts saved successfully!")
    
    def train_complete_pipeline(self):
        """Run complete training pipeline"""
        self.train_models()
        self.plot_model_performance()
        self.get_feature_importance()
        self.calculate_business_metrics()
        self.save_models()

# Run the training
if __name__ == "__main__":
    predictor = LoanPredictor()
    predictor.train_complete_pipeline()