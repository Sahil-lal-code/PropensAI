import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

class TerminalLoanPredictor:
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        """Load trained models and artifacts"""
        try:
            with open('best_loan_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            print("✅ Models loaded successfully!")
        except FileNotFoundError:
            print("❌ Model files not found. Please run model_training.py first.")
            exit()
    
    def get_user_input(self):
        """Get input from user via terminal"""
        print("\n" + "="*50)
        print("🏦 BANK LOAN PREDICTION TERMINAL")
        print("="*50)
        
        print("\n📋 IMPORTANT: This predicts if customers will ACCEPT loan offers")
        print("   It does NOT assess credit risk or loan affordability")
        print("="*50)
        
        print("\nPlease enter applicant details:")
        
        age = int(input("Age (23-67): "))
        experience = int(input("Experience in years (0-43): "))
        income = int(input("Annual Income ($000) (8-224): "))
        family = int(input("Family Size (1-4): "))
        cc_avg = float(input("Monthly Credit Card Avg ($000) (0.0-10.0): "))
        
        print("\nEducation Level:")
        print("1 - Undergraduate")
        print("2 - Graduate") 
        print("3 - Advanced/Professional")
        education = int(input("Choose (1-3): "))
        
        mortgage = int(input("Mortgage Value ($000) (0-635): "))
        
        securities = int(input("Securities Account? (0=No, 1=Yes): "))
        cd_account = int(input("CD Account? (0=No, 1=Yes): "))
        online = int(input("Uses Online Banking? (0=No, 1=Yes): "))
        credit_card = int(input("Uses Bank Credit Card? (0=No, 1=Yes): "))
        
        return {
            'Age': age,
            'Experience': experience,
            'Income': income,
            'Family': family,
            'CCAvg': cc_avg,
            'Education': education,
            'Mortgage': mortgage,
            'Securities Account': securities,
            'CD Account': cd_account,
            'Online': online,
            'CreditCard': credit_card
        }
    
    def predict_loan(self, input_data):
        """Make prediction for input data"""
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[self.feature_names]
        
        # Scale numerical features
        numerical_cols = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']
        input_df[numerical_cols] = self.scaler.transform(input_df[numerical_cols])
        
        # Make prediction
        probability = self.model.predict_proba(input_df)[0][1]
        prediction = self.model.predict(input_df)[0]
        
        return prediction, probability
    
    def explain_prediction(self, input_data, probability):
        """Explain the prediction with business context"""
        print("\n" + "="*50)
        print("📊 PREDICTION ANALYSIS")
        print("="*50)
        
        print(f"\n🎯 Loan Acceptance Probability: {probability:.1%}")
        
        if probability > 0.8:
            recommendation = "✅ STRONG RECOMMENDATION: OFFER LOAN"
            confidence = "Very High"
            emoji = "🎯"
        elif probability > 0.6:
            recommendation = "🟢 RECOMMENDATION: OFFER LOAN"
            confidence = "High"
            emoji = "✅"
        elif probability > 0.4:
            recommendation = "🟡 MODERATE RECOMMENDATION: CONSIDER OFFERING"
            confidence = "Medium"
            emoji = "💭"
        elif probability > 0.2:
            recommendation = "🟠 WEAK RECOMMENDATION: LOW PRIORITY"
            confidence = "Low"
            emoji = "⚠️"
        else:
            recommendation = "❌ RECOMMENDATION: DO NOT OFFER"
            confidence = "Very Low"
            emoji = "🚫"
            
        print(f"{emoji} Recommendation: {recommendation}")
        print(f"🎲 Confidence Level: {confidence}")
        
        # Key factors analysis
        print("\n🔍 KEY DECISION FACTORS:")
        
        positive_factors = []
        negative_factors = []
        
        # Analyze each feature
        if input_data['Income'] > 100:
            positive_factors.append(f"High Income (${input_data['Income']}K)")
        elif input_data['Income'] < 50:
            negative_factors.append(f"Low Income (${input_data['Income']}K)")
            
        if input_data['Education'] == 3:
            positive_factors.append("Advanced Education (Strong positive)")
        elif input_data['Education'] == 2:
            positive_factors.append("Graduate Education (Positive)")
        elif input_data['Education'] == 1:
            negative_factors.append("Undergraduate Only (Weaker signal)")
            
        if input_data['CCAvg'] > 3.0:
            positive_factors.append(f"High Spender (CC Avg: ${input_data['CCAvg']}K)")
        elif input_data['CCAvg'] > 1.5:
            positive_factors.append(f"Active Spender (CC Avg: ${input_data['CCAvg']}K)")
        elif input_data['CCAvg'] < 0.5:
            negative_factors.append(f"Low Credit Card Usage (CC Avg: ${input_data['CCAvg']}K)")
            
        if input_data['CD Account'] == 1:
            positive_factors.append("CD Account Holder (Strong positive - shows savings)")
        else:
            negative_factors.append("No CD Account (Missing strong positive signal)")
            
        if input_data['Securities Account'] == 1:
            positive_factors.append("Securities Account (Multiple banking relationships)")
            
        if input_data['Family'] in [3, 4]:
            positive_factors.append(f"Family Size {input_data['Family']} (Potential need for funds)")
        elif input_data['Family'] == 1:
            negative_factors.append("Single (Potentially lower need)")
        
        # Display factors
        if positive_factors:
            print("\n✅ POSITIVE INDICATORS:")
            for factor in positive_factors:
                print(f"   • {factor}")
                
        if negative_factors:
            print("\n⚠️  NEGATIVE INDICATORS:")
            for factor in negative_factors:
                print(f"   • {factor}")
                
        if not positive_factors and not negative_factors:
            print("\n📊 Neutral profile - no strong indicators either way")
    
    def calculate_business_value(self, probability, income, cc_avg, has_cd_account, has_securities):
        """Calculate realistic business impact"""
        print("\n💼 BUSINESS IMPACT ANALYSIS:")
        print("   Based on: Customer likely to ACCEPT our loan offer")
        
        # REALISTIC assumptions for personal loans
        if income > 150:
            avg_loan_amount = 50000  # $50,000 for high-income customers
            interest_rate = 0.10  # 10% for premium customers
        elif income > 100:
            avg_loan_amount = 35000  # $35,000 for upper-middle income
            interest_rate = 0.12  # 12% standard
        elif income > 60:
            avg_loan_amount = 20000  # $20,000 for middle income
            interest_rate = 0.14  # 14% 
        else:
            avg_loan_amount = 10000  # $10,000 for lower income
            interest_rate = 0.16  # 16% higher risk
        
        loan_term = 3  # 3 years
        marketing_cost = 75  # Cost to market and process the loan
        
        # Calculate revenue
        annual_interest = avg_loan_amount * interest_rate
        total_interest = annual_interest * loan_term
        
        # Risk adjustment - higher probability = lower default risk
        # Customers who are more likely to accept are often better risks
        default_risk_adjustment = (1 - probability) * 0.15
        
        # Net expected value
        risk_adjusted_revenue = total_interest * (1 - default_risk_adjustment)
        net_expected_value = risk_adjusted_revenue - marketing_cost
        
        # Customer lifetime value bonus
        lifetime_bonus = 0
        if has_cd_account:
            lifetime_bonus += 500  # CD holders are valuable long-term
        if has_securities:
            lifetime_bonus += 300  # Multiple products = loyal customer
        if cc_avg > 2:
            lifetime_bonus += 200  # Active credit card users
            
        total_expected_value = net_expected_value + lifetime_bonus
        
        print(f"   • Realistic Loan Amount: ${avg_loan_amount:,}")
        print(f"   • Interest Rate: {interest_rate*100:.1f}% for {loan_term} years")
        print(f"   • Total Interest Revenue: ${total_interest:,.0f}")
        print(f"   • Risk-Adjusted Revenue: ${risk_adjusted_revenue:,.0f}")
        print(f"   • Marketing & Processing Cost: ${marketing_cost}")
        
        if lifetime_bonus > 0:
            print(f"   • Lifetime Value Bonus: ${lifetime_bonus:,.0f}")
        
        print(f"   • Total Expected Value: ${total_expected_value:,.0f}")
        
        if total_expected_value > 0:
            roi = (total_expected_value / marketing_cost) * 100
            print(f"   • 💰 EXPECTED PROFIT: ${total_expected_value:,.0f}")
            print(f"   • 📈 Return on Investment: {roi:.0f}%")
            
            if probability > 0.8:
                print(f"   • 🎯 PRIORITY: HIGH - Excellent candidate")
            else:
                print(f"   • 🎯 PRIORITY: MEDIUM - Good candidate")
        else:
            print(f"   • 📉 EXPECTED LOSS: ${abs(total_expected_value):,.0f}")
            print(f"   • 🎯 PRIORITY: LOW - High acquisition cost")
        
        # Strategic insight
        print(f"\n💡 STRATEGIC INSIGHT:")
        if probability > 0.7:
            print(f"   This customer is very likely to accept - low marketing cost")
            if has_cd_account and has_securities:
                print(f"   Existing relationship - consider premium offer")
        elif probability > 0.4:
            print(f"   Moderate chance of acceptance - standard marketing approach")
        else:
            print(f"   Low chance of acceptance - high marketing cost per conversion")
    
    def run_prediction(self):
        """Run complete prediction workflow"""
        while True:
            try:
                # Get user input
                input_data = self.get_user_input()
                
                # Make prediction
                prediction, probability = self.predict_loan(input_data)
                
                # Explain prediction
                self.explain_prediction(input_data, probability)
                
                # Business impact
                self.calculate_business_value(
                    probability, 
                    input_data['Income'], 
                    input_data['CCAvg'], 
                    input_data['CD Account'],
                    input_data['Securities Account']
                )
                
                # Ask to continue
                print("\n" + "="*50)
                continue_pred = input("\nPredict another applicant? (y/n): ").lower()
                if continue_pred != 'y':
                    print("\nThank you for using Bank Loan Predictor! 👋")
                    print("Remember: This predicts ACCEPTANCE, not CREDIT RISK")
                    break
                    
            except ValueError:
                print("❌ Invalid input. Please enter numbers only.")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again with valid inputs.")

if __name__ == "__main__":
    predictor = TerminalLoanPredictor()
    predictor.run_prediction()