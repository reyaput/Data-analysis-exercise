import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .churn-yes {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .churn-no {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Load model dan preprocessing objects
@st.cache_resource
def load_models():
    """Load trained model, scaler, and encoders"""
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

try:
    model, scaler, label_encoders = load_models()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Title and description
st.title("🏦 Bank Customer Churn Prediction System")
st.markdown("""
    <div style='background-color: #0000; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3>Welcome to Customer Churn Predictor!</h3>
        <p>This application predicts whether a bank customer is likely to churn (leave the bank) 
        based on their profile and behavior. Use this tool to:</p>
        <ul>
            <li>🎯 Predict individual customer churn risk</li>
            <li>📊 Analyze churn probability</li>
            <li>📁 Batch predict multiple customers via CSV upload</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎯 Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Single Prediction", "Batch Prediction", "Model Info"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
    **About this App:**
    - Model: Random Forest Classifier
    - Features: 18 customer attributes
    
    **Created by:** Reynanda Arya Putra
    **GitHub:** [https://github.com/reyaput]
""")

def preprocess_input(data_dict):
    """
    Preprocess user input untuk prediksi
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing user input
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([data_dict])
    
    # Encode categorical variables
    categorical_cols = {
        'Geography': label_encoders['Geography'],
        'Gender': label_encoders['Gender'],
        'Card Type': label_encoders['Card Type'],
        'AgeGroup': label_encoders['AgeGroup'],
        'Tenure_Category': label_encoders['Tenure_Category'],
        'Balance_Category': label_encoders['Balance_Category'],
        'CreditScore_Category': label_encoders['CreditScore_Category']
    }
    
    for col, encoder in categorical_cols.items():
        df[col + '_Encoded'] = encoder.transform(df[col])
    
    # Define feature order (sama seperti saat training)
    feature_list = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
        'EstimatedSalary', 'Complain', 'Satisfaction Score', 
        'Point Earned',
        'Geography_Encoded', 'Gender_Encoded', 'Card Type_Encoded',
        'AgeGroup_Encoded', 'Tenure_Category_Encoded', 
        'Balance_Category_Encoded', 'CreditScore_Category_Encoded'
    ]
    
    return df[feature_list]


def get_age_group(age):
    """Categorize age into groups"""
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Adult'
    else:
        return 'Senior'


def get_tenure_category(tenure):
    """Categorize tenure into groups"""
    if tenure < 2:
        return 'New'
    elif tenure < 7:
        return 'Established'
    else:
        return 'Loyal'


def get_balance_category(balance):
    """Categorize balance into groups"""
    if balance == 0:
        return 'Zero Balance'
    elif balance <= 100000:
        return 'Low Balance'
    else:
        return 'High Balance'
        return 'High '

def get_credit_score_category(score):
    """Categorize credit score into groups"""
    if score < 350:
        return 'Low'
    elif score < 584:
        return 'Medium'
    elif score < 652:
        return 'High'
    else:
        return 'Very High'
    
if app_mode == "Single Prediction":
    st.header("🎯 Single Customer Prediction")
    st.markdown("Enter customer information below to predict churn risk.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Basic Information")
        
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            help="Customer's credit score (300-850)"
        )
        
        geography = st.selectbox(
            "Geography",
            options=['France', 'Germany', 'Spain'],
            help="Customer's country"
        )
        
        gender = st.selectbox(
            "Gender",
            options=['Male', 'Female'],
            help="Customer's gender"
        )
        
        age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            help="Customer's age"
        )
        
        tenure = st.slider(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=5,
            help="How many years the customer has been with the bank"
        )
        
        balance = st.number_input(
            "Account Balance ($)",
            min_value=0.0,
            max_value=300000.0,
            value=50000.0,
            step=1000.0,
            help="Current account balance"
        )
    
    with col2:
        st.subheader("💳 Account Details")
        
        num_of_products = st.selectbox(
            "Number of Products",
            options=[1, 2, 3, 4],
            help="How many bank products the customer has"
        )
        
        has_cr_card = st.selectbox(
            "Has Credit Card?",
            options=["Yes", "No"],
            help="Does the customer have a credit card?"
        )
        has_cr_card = 1 if has_cr_card == "Yes" else 0
        
        is_active_member = st.selectbox(
            "Is Active Member?",
            options=["Yes", "No"],
            help="Is the customer an active member?"
        )
        is_active_member = 1 if is_active_member == "Yes" else 0
        
        estimated_salary = st.number_input(
            "Estimated Salary ($)",
            min_value=0.0,
            max_value=200000.0,
            value=50000.0,
            step=1000.0,
            help="Customer's estimated annual salary"
        )
        
        complain = st.selectbox(
            "Has Complaint?",
            options=["Yes", "No"],
            help="Has the customer filed a complaint?"
        )
        complain = 1 if complain == "Yes" else 0
        
        satisfaction_score = st.slider(
            "Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3,
            help="Customer satisfaction score (1-5)"
        )
        
        card_type = st.selectbox(
            "Card Type",
            options=['DIAMOND', 'GOLD', 'PLATINUM', 'SILVER'],
            help="Type of card the customer holds"
        )
        
        point_earned = st.number_input(
            "Points Earned",
            min_value=0,
            max_value=1000,
            value=500,
            help="Loyalty points earned"
        )
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Churn Risk", type="primary"):
        
        # Auto-generate engineered features
        age_group = get_age_group(age)
        tenure_category = get_tenure_category(tenure)
        balance_category = get_balance_category(balance)
        credit_score_category = get_credit_score_category(credit_score)
        
        # Prepare input data
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary,
            'Complain': complain,
            'Satisfaction Score': satisfaction_score,
            'Card Type': card_type,
            'Point Earned': point_earned,
            'AgeGroup': age_group,
            'Tenure_Category': tenure_category,
            'Balance_Category': balance_category,
            'CreditScore_Category': credit_score_category
        }
        
        # Preprocess and predict
        with st.spinner("Analyzing customer data..."):
            processed_data = preprocess_input(input_data)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Create three columns for results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            churn_status = "Will Churn 😟" if prediction == 1 else "Will Stay 😊"
            churn_color = "#f44336" if prediction == 1 else "#4caf50"
            st.markdown(f"""
                <div style='background-color: {churn_color}20; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: {churn_color}; margin: 0;'>{churn_status}</h3>
                </div>
            """, unsafe_allow_html=True)
        
        with res_col2:
            st.metric(
                "Churn Probability",
                f"{probability[1]:.1%}",
                delta=f"{probability[1] - 0.5:.1%} vs baseline"
            )
        
        with res_col3:
            risk_level = "🔴 High Risk" if probability[1] > 0.7 else "🟡 Medium Risk" if probability[1] > 0.5 else "🟢 Low Risk"
            st.markdown(f"""
                <div style='background-color: #00000; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='margin: 0;'>{risk_level}</h3>
                </div>
            """, unsafe_allow_html=True)
        
        # Probability breakdown
        st.markdown("### 📈 Probability Breakdown")
        prob_df = pd.DataFrame({
            'Outcome': ['Will Stay', 'Will Churn'],
            'Probability': [probability[0], probability[1]]
        })
        
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#4caf50', '#f44336']
        bars = ax.barh(prob_df['Outcome'], prob_df['Probability'], color=colors)
        ax.set_xlabel('Probability', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, prob_df['Probability'])):
            ax.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if prediction == 1:
            st.warning("""
                **⚠️ High churn risk detected!** Consider these actions:
                - 📞 Immediate customer outreach
                - 🎁 Offer retention incentives (discounts, upgrades)
                - 🔍 Investigate and resolve any complaints
                - ⭐ Improve customer satisfaction through personalized service
            """)
        else:
            st.success("""
                **✅ Low churn risk!** Maintain customer satisfaction:
                - 🎯 Continue current engagement strategies
                - 📊 Monitor customer behavior regularly
                - 💎 Consider loyalty rewards program
                - 📧 Keep customer informed about new products
            """)

elif app_mode == "Batch Prediction":
    st.header("📁 Batch Prediction")
    st.markdown("Upload a CSV file with multiple customer records to predict churn for all of them at once.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with customer data. Must contain all required columns."
    )
    
    # Show sample format
    with st.expander("📋 View Required CSV Format"):
        st.markdown("""
            Your CSV file should contain these columns:
            - CreditScore, Geography, Gender, Age, Tenure, Balance
            - NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
            - Complain, Satisfaction Score, Card Type, Point Earned
            - AgeGroup, Tenure_Category, Balance_Category, CreditScore_Category
        """)
        
        sample_data = pd.DataFrame({
            'CreditScore': [650, 700],
            'Geography': ['France', 'Germany'],
            'Gender': ['Male', 'Female'],
            'Age': [35, 42],
            'Tenure': [5, 8],
            'Balance': [50000, 120000],
            'NumOfProducts': [2, 1],
            'HasCrCard': [1, 1],
            'IsActiveMember': [1, 0],
            'EstimatedSalary': [60000, 80000],
            'Complain': [0, 1],
            'Satisfaction Score': [3, 2],
            'Card Type': ['GOLD', 'PLATINUM'],
            'Point Earned': [500, 750],
            'AgeGroup': ['Middle-Aged', 'Middle-Aged'],
            'Tenure_Category': ['Regular', 'Loyal'],
            'Balance_Category': ['Medium', 'High'],
            'CreditScore_Category': ['Good', 'Excellent']
        })
        st.dataframe(sample_data)
    
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(batch_df)} customers.")
            
            # Show preview
            with st.expander("👀 Preview Data (First 5 rows)"):
                st.dataframe(batch_df.head())
            
            # Predict button
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing predictions..."):
                    
                    # Store predictions
                    predictions = []
                    probabilities = []
                    
                    for idx, row in batch_df.iterrows():
                        input_data = row.to_dict()
                        processed_data = preprocess_input(input_data)
                        pred = model.predict(processed_data)[0]
                        prob = model.predict_proba(processed_data)[0][1]
                        
                        predictions.append(pred)
                        probabilities.append(prob)
                    
                    # Add predictions to dataframe
                    batch_df['Predicted_Churn'] = predictions
                    batch_df['Churn_Probability'] = probabilities
                    batch_df['Risk_Level'] = batch_df['Churn_Probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.5 else 'Low'
                    )
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Batch Prediction Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Customers", len(batch_df))
                
                with col2:
                    churn_count = batch_df['Predicted_Churn'].sum()
                    st.metric("Predicted Churners", churn_count)
                
                with col3:
                    churn_rate = (churn_count / len(batch_df)) * 100
                    st.metric("Churn Rate", f"{churn_rate:.1f}%")
                
                with col4:
                    high_risk = (batch_df['Risk_Level'] == 'High').sum()
                    st.metric("High Risk Customers", high_risk)
                
                # Visualizations
                st.markdown("### 📈 Visual Analysis")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Churn distribution
                    fig1, ax1 = plt.subplots(figsize=(8, 5))
                    churn_counts = batch_df['Predicted_Churn'].value_counts()
                    colors = ['#4caf50', '#f44336']
                    ax1.pie(churn_counts, labels=['Will Stay', 'Will Churn'], 
                           autopct='%1.1f%%', colors=colors, startangle=90)
                    ax1.set_title('Churn Distribution', fontweight='bold')
                    st.pyplot(fig1)
                
                with viz_col2:
                    # Risk level distribution
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    risk_counts = batch_df['Risk_Level'].value_counts()
                    risk_colors = {'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'}
                    bars = ax2.bar(risk_counts.index, risk_counts.values, 
                                  color=[risk_colors[x] for x in risk_counts.index])
                    ax2.set_title('Risk Level Distribution', fontweight='bold')
                    ax2.set_ylabel('Count')
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig2)
                
                # Show full results table
                st.markdown("### 📋 Detailed Results")
                
                # Select columns to display
                display_cols = ['CreditScore', 'Geography', 'Age', 'Balance', 
                               'Predicted_Churn', 'Churn_Probability', 'Risk_Level']
                st.dataframe(
                    batch_df[display_cols].style.background_gradient(
                        subset=['Churn_Probability'], 
                        cmap='RdYlGn_r'
                    ),
                    use_container_width=True
                )
                
                # Download results
                st.markdown("### 💾 Download Results")
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Please make sure your CSV file has all required columns in the correct format.")

elif app_mode == "Model Info":
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    # Model details
    with col1:
        st.markdown("""
            <div style='background-color: #00000; padding: 20px; border-radius: 10px;'>
                <h3>🤖 Model Details</h3>
                <ul>
                    <li><strong>Algorithm:</strong> Random Forest Classifier</li>
                    <li><strong>Number of Trees:</strong> 100</li>
                    <li><strong>Training Data:</strong> 10,000 customer records</li>
                    <li><strong>Train-Test Split:</strong> 80-20</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: #00000; padding: 20px; border-radius: 10px;'>
                <h3>📈 Model Performance</h3>
                <ul>
                    <li><strong>Accuracy:</strong> 99.9%</li>
                    <li><strong>Precision:</strong> 99.75%</li>
                    <li><strong>Recall:</strong> 99.75%</li>
                    <li><strong>F1-Score:</strong> 99.75%</li>
                    <li><strong>ROC-AUC:</strong> 99.9%</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature importance visualization
    st.subheader("📊 Feature Importance Visualization")
    
    feature_names = [
        'Age', 'NumOfProducts', 'IsActiveMember', 'Balance', 'Geography_Encoded',
        'Complain', 'CreditScore', 'EstimatedSalary', 'Satisfaction Score',
        'Tenure', 'Gender_Encoded', 'Point Earned', 'HasCrCard',
        'Card Type_Encoded', 'AgeGroup_Encoded', 'Tenure_Category_Encoded',
        'Balance_Category_Encoded', 'CreditScore_Category_Encoded'
    ]
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, y='Feature', x='Importance', palette='viridis', ax=ax)
    ax.set_title('Top 10 Most Important Features', fontweight='bold', fontsize=14)
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    # About section
    st.markdown("---")
    st.html("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; color: #333;'>
        <h3>ℹ️ About This Project</h3>
        <p>This Customer Churn Prediction system is built using machine learning to help banks 
        identify customers who are at risk of leaving.</p>
        <h4>📚 Technologies Used:</h4>
        <ul>
            <li>Python 3.11</li>
            <li>Scikit-learn</li>
            <li>Streamlit</li>
            <li>Pandas & NumPy</li>
            <li>Matplotlib & Seaborn</li>
        </ul>
    </div>
    """)

