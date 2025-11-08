import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.clean_data = None
        
    def explore_data(self):
        print("=== DATASET EXPLORATION ===")
        print(f"Dataset Shape: {self.data.shape}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\n=== DATA TYPES ===")
        print(self.data.dtypes)
        
        print("\n=== MISSING VALUES ===")
        print(self.data.isnull().sum())
        
        print("\n=== BASIC STATISTICS ===")
        print(self.data.describe())
        
        # Check for negative experience values
        negative_exp = self.data[self.data['Experience'] < 0]
        print(f"\nRows with negative experience: {len(negative_exp)}")
        
    def clean_dataset(self):
        """Clean and preprocess the dataset"""
        df = self.data.copy()
        
        # Handle negative experience - convert to absolute values
        df['Experience'] = abs(df['Experience'])
        
        # Drop ID and ZIP Code as they are identifiers
        df = df.drop(['ID', 'ZIP Code'], axis=1)
        
        # Convert categorical variables to proper data types
        categorical_cols = ['Family', 'Education', 'Securities Account', 
                          'CD Account', 'Online', 'CreditCard', 'Personal Loan']
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        
        self.clean_data = df
        print("✅ Data cleaning completed!")
        print(f"Final dataset shape: {self.clean_data.shape}")
        
        return df
    
    def analyze_target_variable(self):
        """Analyze the distribution of the target variable"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        loan_dist = self.clean_data['Personal Loan'].value_counts()
        plt.pie(loan_dist.values, labels=['Not Accepted', 'Accepted'], 
                autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
        plt.title('Personal Loan Distribution')
        
        plt.subplot(1, 2, 2)
        # Income vs Loan acceptance
        sns.boxplot(x='Personal Loan', y='Income', data=self.clean_data)
        plt.title('Income Distribution by Loan Acceptance')
        
        plt.tight_layout()
        plt.savefig('target_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n=== TARGET VARIABLE ANALYSIS ===")
        print(f"Loan Accepted: {loan_dist[1]} ({loan_dist[1]/len(self.clean_data)*100:.2f}%)")
        print(f"Loan Not Accepted: {loan_dist[0]} ({loan_dist[0]/len(self.clean_data)*100:.2f}%)")
    
    def feature_correlation(self):
        """Analyze correlation between features"""
        # Convert categorical to numeric for correlation
        corr_data = self.clean_data.copy()
        for col in corr_data.select_dtypes(['category']).columns:
            corr_data[col] = corr_data[col].cat.codes
            
        plt.figure(figsize=(12, 10))
        correlation_matrix = corr_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation with target
        target_corr = correlation_matrix['Personal Loan'].sort_values(ascending=False)
        print("\n=== CORRELATION WITH TARGET (Personal Loan) ===")
        for feature, corr in target_corr.items():
            if feature != 'Personal Loan':
                print(f"{feature:20} : {corr:+.3f}")
    
    def save_clean_data(self, output_path='cleaned_loan_data.csv'):
        """Save cleaned dataset"""
        if self.clean_data is not None:
            # Convert categorical back to int for saving (better for modeling)
            save_data = self.clean_data.copy()
            for col in save_data.select_dtypes(['category']).columns:
                save_data[col] = save_data[col].cat.codes
                
            save_data.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to {output_path}")
            return True
        else:
            print("❌ No clean data to save!")
            return False
    
    def full_analysis(self):
        """Run complete analysis"""
        self.explore_data()
        self.clean_dataset()
        self.analyze_target_variable()
        self.feature_correlation()
        success = self.save_clean_data()
        
        if success:
            print("\n🎉 Data analysis completed successfully! Ready for model training.")
        else:
            print("\n❌ Data analysis failed!")

# Run the analysis
if __name__ == "__main__":
    analyzer = DataAnalyzer('Bank_Personal_Loan_Modelling.csv')
    analyzer.full_analysis()