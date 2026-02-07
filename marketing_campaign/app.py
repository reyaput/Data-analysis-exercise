"""
Marketing Campaign Success Prediction - Streamlit Application
==============================================================

This app predicts whether a marketing campaign will be successful based on
various campaign parameters and provides actionable recommendations.

Success Criteria:
    - ROI >= 5 AND Conversion_Rate >= 0.08
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Page Configuration
st.set_page_config(
    page_title="Campaign Success Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }
    .fail-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .recommendation-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts(model_dir='models'):
    """Load all model artifacts (cached for performance)."""
    try:
        artifacts = {}
        
        # Load model
        with open(f'{model_dir}/best_model.pkl', 'rb') as f:
            artifacts['model'] = pickle.load(f)
        
        # Load scaler
        with open(f'{model_dir}/scaler.pkl', 'rb') as f:
            artifacts['scaler'] = pickle.load(f)
        
        # Load label encoders
        with open(f'{model_dir}/label_encoders.pkl', 'rb') as f:
            artifacts['label_encoders'] = pickle.load(f)
        
        # Load feature names
        with open(f'{model_dir}/feature_names.pkl', 'rb') as f:
            artifacts['feature_names'] = pickle.load(f)
        
        # Load metadata
        try:
            with open(f'{model_dir}/model_metadata.json', 'r') as f:
                artifacts['metadata'] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            artifacts['metadata'] = {
                'model_name': 'XGBoost',
                'training_date': 'N/A',
                'performance': {
                    'f1_score': 0.0,
                    'roc_auc': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'accuracy': 0.0
                }
            }
        
        # Load feature importance
        try:
            artifacts['feature_importance'] = pd.read_csv(f'{model_dir}/feature_importance.csv')
        except:
            artifacts['feature_importance'] = pd.DataFrame({
                'Feature': ['Unknown'],
                'Importance': [1.0]
            })
        
        return artifacts, True
    except Exception as e:
        return None, str(e)


class CampaignPredictor:
    """Class to handle model loading and predictions."""
    
    def __init__(self, model_dir='models'):
        """Initialize predictor by loading saved artifacts."""
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.metadata = {}
        self.feature_importance = None
        
    def load_artifacts(self):
        """Load all model artifacts."""
        artifacts, success = load_model_artifacts(self.model_dir)
        
        if not success:
            st.error(f"Error loading model artifacts: {artifacts}")
            return False
        
        # Assign artifacts to instance variables
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.label_encoders = artifacts['label_encoders']
        self.feature_names = artifacts['feature_names']
        self.metadata = artifacts['metadata']
        self.feature_importance = artifacts['feature_importance']
        
        return True
    
    def engineer_features(self, input_data):
        """Engineer features from input data."""
        df = pd.DataFrame([input_data])
        
        # Calculate engineered features
        df['CPC'] = df['Acquisition_Cost'] / df['Clicks'].replace(0, 1)
        df['CTR'] = df['Clicks'] / df['Impressions'].replace(0, 1)
        df['CPM'] = (df['Acquisition_Cost'] / df['Impressions'].replace(0, 1)) * 1000
        df['Engagement_per_Click'] = df['Engagement_Score'] / df['Clicks'].replace(0, 1)
        df['ROI_Efficiency'] = df['ROI'] / (df['Acquisition_Cost'] / 1000)
        df['Click_to_Engagement_Ratio'] = df['Clicks'] / df['Engagement_Score'].replace(0, 1)
        df['Impression_to_Click_Efficiency'] = df['Impressions'] / df['Clicks'].replace(0, 1)
        
        return df
    
    def prepare_input(self, input_data):
        """Prepare input data for prediction."""
        # Engineer features
        df = self.engineer_features(input_data)
        
        # Encode categorical variables
        categorical_features = [
            'Campaign_Type', 'Target_Audience', 'Channel_Used',
            'Location', 'Language', 'Customer_Segment', 'Gender', 'Age_Group'
        ]
        
        for col in categorical_features:
            if col in self.label_encoders and col in df.columns:
                try:
                    df[col + '_encoded'] = self.label_encoders[col].transform([df[col].iloc[0]])[0]
                except:
                    # If new category, use most frequent class
                    df[col + '_encoded'] = 0
        
        # Select features in correct order
        X = df[self.feature_names]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names
        )
        
        return X_scaled, df
    
    def predict(self, input_data):
        """Make prediction and return results."""
        # Prepare input
        X_scaled, df_features = self.prepare_input(input_data)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        # Get feature values for explanation
        feature_values = X_scaled.iloc[0].to_dict()
        
        results = {
            'prediction': prediction,
            'probability_fail': probability[0],
            'probability_success': probability[1],
            'confidence': max(probability),
            'feature_values': feature_values,
            'engineered_features': df_features.iloc[0].to_dict()
        }
        
        return results
    
    def get_recommendations(self, input_data, results):
        """Generate actionable recommendations based on prediction."""
        recommendations = []
        
        prob_success = results['probability_success']
        features = results['engineered_features']
        
        # ROI Efficiency Analysis
        roi_efficiency = features['ROI_Efficiency']
        if roi_efficiency < 0.5:
            recommendations.append({
                'category': '💰 ROI Optimization',
                'priority': 'HIGH',
                'message': f"ROI Efficiency sangat rendah ({roi_efficiency:.2f}). Target minimal 0.7.",
                'action': "Reduce acquisition cost atau tingkatkan ROI melalui better targeting dan messaging."
            })
        
        # CPC Analysis
        cpc = features['CPC']
        if cpc > 50:
            recommendations.append({
                'category': '💸 Cost Per Click',
                'priority': 'HIGH',
                'message': f"CPC terlalu tinggi (${cpc:.2f}). Industry benchmark: $20-40.",
                'action': "Optimize ad quality score, improve targeting, atau test different ad formats."
            })
        
        # CTR Analysis
        ctr = features['CTR']
        if ctr < 0.02:
            recommendations.append({
                'category': '📊 Click-Through Rate',
                'priority': 'MEDIUM',
                'message': f"CTR rendah ({ctr*100:.2f}%). Target minimal: 2-3%.",
                'action': "Improve ad copy, use compelling CTAs, dan test different creatives."
            })
        elif ctr > 0.05:
            recommendations.append({
                'category': '✅ Click-Through Rate',
                'priority': 'LOW',
                'message': f"CTR excellent ({ctr*100:.2f}%)! Ad messaging resonates dengan audience.",
                'action': "Maintain current creative strategy dan consider scaling budget."
            })
        
        # Engagement Analysis
        engagement_per_click = features['Engagement_per_Click']
        if engagement_per_click < 0.01:
            recommendations.append({
                'category': '👥 Engagement Quality',
                'priority': 'MEDIUM',
                'message': f"Engagement per click rendah ({engagement_per_click:.4f}).",
                'action': "Improve landing page experience, content quality, dan user journey."
            })
        
        # Budget Analysis
        acquisition_cost = input_data['Acquisition_Cost']
        if acquisition_cost > 15000:
            recommendations.append({
                'category': '💵 Budget Allocation',
                'priority': 'MEDIUM',
                'message': f"High acquisition cost (${acquisition_cost:,.0f}).",
                'action': "Ensure expected returns justify investment. Consider A/B testing dengan smaller budget first."
            })
        
        # Channel-specific recommendations
        channel = input_data['Channel_Used']
        if channel == 'Google Ads':
            recommendations.append({
                'category': '🎯 Channel Strategy',
                'priority': 'LOW',
                'message': "Google Ads dipilih - bagus untuk intent-based marketing.",
                'action': "Focus on high-intent keywords, optimize Quality Score, dan use negative keywords."
            })
        elif channel == 'Facebook':
            recommendations.append({
                'category': '🎯 Channel Strategy',
                'priority': 'LOW',
                'message': "Facebook dipilih - excellent untuk audience targeting.",
                'action': "Leverage detailed targeting, create lookalike audiences, dan test different ad formats."
            })
        
        # Overall recommendation based on probability
        if prob_success >= 0.7:
            recommendations.insert(0, {
                'category': '🎉 Overall Assessment',
                'priority': 'SUCCESS',
                'message': f"Campaign punya probability sukses tinggi ({prob_success*100:.1f}%)!",
                'action': "Proceed dengan confidence! Monitor performance dan scale jika results positif."
            })
        elif prob_success >= 0.5:
            recommendations.insert(0, {
                'category': '⚠️ Overall Assessment',
                'priority': 'MEDIUM',
                'message': f"Campaign punya probability sukses moderate ({prob_success*100:.1f}%).",
                'action': "Consider testing dengan smaller budget first atau optimize parameters di atas."
            })
        else:
            recommendations.insert(0, {
                'category': '❌ Overall Assessment',
                'priority': 'HIGH',
                'message': f"Campaign punya probability sukses rendah ({prob_success*100:.1f}%).",
                'action': "Strongly recommend revising campaign strategy sebelum execution."
            })
        
        return recommendations
    
    def calculate_potential_savings(self, input_data, results):
        """Calculate potential savings from using this model."""
        acquisition_cost = input_data['Acquisition_Cost']
        prob_fail = results['probability_fail']
        
        # If model predicts high probability of failure, potential savings = avoided cost
        if prob_fail > 0.7:
            potential_savings = acquisition_cost * 0.9  # 90% of cost saved
            confidence = "High"
        elif prob_fail > 0.5:
            potential_savings = acquisition_cost * 0.5  # 50% of cost saved
            confidence = "Medium"
        else:
            potential_savings = 0
            confidence = "Low"
        
        return potential_savings, confidence


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🎯 Marketing Campaign Success Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict campaign success probability and get actionable recommendations</div>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = CampaignPredictor()
    
    # Load model artifacts
    if not predictor.load_artifacts():
        st.error("❌ Failed to load model artifacts. Please ensure the 'models' folder exists with all required files.")
        return
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"""
        **Model:** {predictor.metadata.get('model_name', 'XGBoost')}
        
        **Performance Metrics:**
        - F1-Score: {predictor.metadata['performance']['f1_score']:.4f}
        - ROC-AUC: {predictor.metadata['performance']['roc_auc']:.4f}
        - Precision: {predictor.metadata['performance']['precision']:.4f}
        - Recall: {predictor.metadata['performance']['recall']:.4f}
        
        **Training Date:** {predictor.metadata.get('training_date', 'N/A')}
        
        **Success Criteria:**
        - ROI ≥ 5
        - Conversion Rate ≥ 0.08
        """)
        
        st.header("🔝 Top Features")
        top_5_features = predictor.feature_importance.head(5)
        for idx, row in top_5_features.iterrows():
            st.metric(
                label=row['Feature'],
                value=f"{row['Importance']:.4f}"
            )
    
    # Main Content
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Batch Analysis", "📈 Feature Importance"])
    
    # TAB 1: Single Prediction
    with tab1:
        st.header("Campaign Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Campaign Settings")
            campaign_type = st.selectbox(
                "Campaign Type",
                options=['Email', 'Social Media', 'Display', 'Search', 'Influencer'],
                help="Type of marketing campaign"
            )
            
            channel_used = st.selectbox(
                "Channel",
                options=['Google Ads', 'Facebook', 'Instagram', 'YouTube', 'Email', 'Website'],
                help="Marketing channel to be used"
            )
            
            customer_segment = st.selectbox(
                "Customer Segment",
                options=['Tech Enthusiasts', 'Fashionistas', 'Foodies', 'Health & Wellness', 'Outdoor Adventurers'],
                help="Target customer segment"
            )
            
            duration_days = st.slider(
                "Campaign Duration (days)",
                min_value=7,
                max_value=90,
                value=30,
                help="How long the campaign will run"
            )
        
        with col2:
            st.subheader("Target Audience")
            gender = st.selectbox(
                "Gender",
                options=['Men', 'Women', 'All'],
                help="Target gender"
            )
            
            age_group = st.selectbox(
                "Age Group",
                options=['18-24', '25-34', '35-44', 'All Ages'],
                help="Target age group"
            )
            
            location = st.selectbox(
                "Location",
                options=['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'],
                help="Target location"
            )
            
            language = st.selectbox(
                "Language",
                options=['English', 'Spanish', 'Mandarin', 'French', 'German'],
                help="Campaign language"
            )
        
        with col3:
            st.subheader("Budget & Metrics")
            acquisition_cost = st.number_input(
                "Acquisition Cost ($)",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                help="Total campaign budget"
            )
            
            impressions = st.number_input(
                "Expected Impressions",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="Expected number of impressions"
            )
            
            clicks = st.number_input(
                "Expected Clicks",
                min_value=100,
                max_value=10000,
                value=500,
                step=100,
                help="Expected number of clicks"
            )
            
            engagement_score = st.slider(
                "Engagement Score",
                min_value=1,
                max_value=10,
                value=5,
                help="Expected engagement level (1-10)"
            )
        
        # Additional fields needed for calculation
        st.subheader("Additional Parameters")
        col4, col5 = st.columns(2)
        
        with col4:
            roi = st.number_input(
                "Expected ROI",
                min_value=0.0,
                max_value=20.0,
                value=5.0,
                step=0.1,
                help="Expected Return on Investment"
            )
        
        with col5:
            conversion_rate = st.number_input(
                "Expected Conversion Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.08,
                step=0.01,
                format="%.3f",
                help="Expected conversion rate (0-1)"
            )
        
        # Combine target audience
        if gender == 'All':
            target_audience = f"{gender} {age_group}"
        else:
            target_audience = f"{gender} {age_group}"
        
        # Predict button
        st.markdown("---")
        predict_btn = st.button("🚀 Predict Campaign Success", type="primary", use_container_width=True)
        
        if predict_btn:
            # Prepare input data
            input_data = {
                'Campaign_Type': campaign_type,
                'Target_Audience': target_audience,
                'Channel_Used': channel_used,
                'Location': location,
                'Language': language,
                'Customer_Segment': customer_segment,
                'Gender': gender,
                'Age_Group': age_group,
                'Acquisition_Cost': acquisition_cost,
                'Clicks': clicks,
                'Impressions': impressions,
                'Engagement_Score': engagement_score,
                'Duration_Days': duration_days,
                'ROI': roi,
                'Conversion_Rate': conversion_rate
            }
            
            # Make prediction
            with st.spinner("🔮 Analyzing campaign parameters..."):
                results = predictor.predict(input_data)
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            # Main prediction box
            if results['prediction'] == 1:
                st.markdown(f"""
                <div class="success-box">
                    <h2 style="color: #28a745; margin:0;">✅ CAMPAIGN LIKELY TO SUCCEED</h2>
                    <h3 style="margin-top:1rem;">Success Probability: {results['probability_success']*100:.1f}%</h3>
                    <p style="font-size:1.1rem; margin-top:0.5rem;">
                        This campaign has a <strong>{results['probability_success']*100:.1f}%</strong> chance of achieving:
                        <br>• ROI ≥ 5
                        <br>• Conversion Rate ≥ 8%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="fail-box">
                    <h2 style="color: #dc3545; margin:0;">❌ CAMPAIGN LIKELY TO FAIL</h2>
                    <h3 style="margin-top:1rem;">Failure Probability: {results['probability_fail']*100:.1f}%</h3>
                    <p style="font-size:1.1rem; margin-top:0.5rem;">
                        This campaign has a <strong>{results['probability_fail']*100:.1f}%</strong> chance of NOT achieving success criteria.
                        <br><strong>Consider revising your strategy!</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability gauge
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=results['probability_success'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Success Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ffcdd2"},
                            {'range': [30, 70], 'color': "#fff9c4"},
                            {'range': [70, 100], 'color': "#c8e6c9"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            st.subheader("📊 Campaign Metrics")
            metric_cols = st.columns(5)
            
            engineered = results['engineered_features']
            
            with metric_cols[0]:
                st.metric(
                    "ROI Efficiency",
                    f"{engineered['ROI_Efficiency']:.2f}",
                    help="ROI per $1000 spent"
                )
            
            with metric_cols[1]:
                st.metric(
                    "CPC",
                    f"${engineered['CPC']:.2f}",
                    help="Cost Per Click"
                )
            
            with metric_cols[2]:
                st.metric(
                    "CTR",
                    f"{engineered['CTR']*100:.2f}%",
                    help="Click-Through Rate"
                )
            
            with metric_cols[3]:
                st.metric(
                    "CPM",
                    f"${engineered['CPM']:.2f}",
                    help="Cost Per Thousand Impressions"
                )
            
            with metric_cols[4]:
                st.metric(
                    "Engagement/Click",
                    f"{engineered['Engagement_per_Click']:.4f}",
                    help="Engagement per Click"
                )
            
            # Recommendations
            st.markdown("---")
            st.header("💡 Recommendations")
            
            recommendations = predictor.get_recommendations(input_data, results)
            
            for rec in recommendations:
                priority_color = {
                    'SUCCESS': '🟢',
                    'HIGH': '🔴',
                    'MEDIUM': '🟡',
                    'LOW': '🔵'
                }
                
                with st.expander(f"{priority_color.get(rec['priority'], '⚪')} {rec['category']}", expanded=(rec['priority'] in ['SUCCESS', 'HIGH'])):
                    st.markdown(f"**{rec['message']}**")
                    st.markdown(f"**Action:** {rec['action']}")
            
            # Potential Savings
            st.markdown("---")
            st.header("💰 Estimated Savings")
            
            potential_savings, confidence = predictor.calculate_potential_savings(input_data, results)
            
            if potential_savings > 0:
                st.success(f"""
                **Potential Savings by Screening:** ${potential_savings:,.2f}
                
                **Confidence Level:** {confidence}
                
                By using this model to screen campaigns before execution, you could potentially save 
                ${potential_savings:,.2f} by avoiding this likely-to-fail campaign.
                """)
            else:
                st.info("""
                **Low Risk Campaign**
                
                This campaign shows good potential for success. Proceed with execution and monitor performance closely.
                """)
    
    # TAB 2: Batch Analysis
    with tab2:
        st.header("📊 Batch Campaign Analysis")
        st.info("Upload a CSV file with multiple campaigns to analyze them all at once.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should contain columns: Campaign_Type, Channel_Used, Customer_Segment, etc."
        )
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_batch)} campaigns")
                
                st.subheader("Preview Data")
                st.dataframe(df_batch.head(10))
                
                if st.button("🚀 Analyze All Campaigns", type="primary"):
                    with st.spinner("Analyzing campaigns..."):
                        predictions = []
                        probabilities = []
                        
                        for idx, row in df_batch.iterrows():
                            input_dict = row.to_dict()
                            result = predictor.predict(input_dict)
                            predictions.append(result['prediction'])
                            probabilities.append(result['probability_success'])
                        
                        df_batch['Predicted_Success'] = predictions
                        df_batch['Success_Probability'] = probabilities
                        
                        # Summary statistics
                        st.subheader("📈 Analysis Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Campaigns",
                                len(df_batch)
                            )
                        
                        with col2:
                            st.metric(
                                "Predicted Success",
                                f"{sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)",
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "Predicted Failure",
                                f"{len(predictions)-sum(predictions)} ({(len(predictions)-sum(predictions))/len(predictions)*100:.1f}%)",
                                delta=None
                            )
                        
                        with col4:
                            avg_prob = np.mean(probabilities)
                            st.metric(
                                "Avg Success Prob",
                                f"{avg_prob*100:.1f}%"
                            )
                        
                        # Results table
                        st.subheader("📋 Detailed Results")
                        st.dataframe(
                            df_batch[['Campaign_Type', 'Channel_Used', 'Customer_Segment', 
                                     'Acquisition_Cost', 'Predicted_Success', 'Success_Probability']],
                            use_container_width=True
                        )
                        
                        # Download results
                        csv = df_batch.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name=f"campaign_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visualization
                        st.subheader("📊 Probability Distribution")
                        fig = px.histogram(
                            df_batch,
                            x='Success_Probability',
                            nbins=20,
                            title="Distribution of Success Probabilities",
                            labels={'Success_Probability': 'Success Probability'},
                            color_discrete_sequence=['#1f77b4']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # TAB 3: Feature Importance
    with tab3:
        st.header("📈 Feature Importance Analysis")
        
        st.markdown("""
        Understanding which features most strongly influence campaign success helps you 
        optimize future campaigns more effectively.
        """)
        
        # Check if feature importance data is available
        if predictor.feature_importance is None or len(predictor.feature_importance) == 0 or predictor.feature_importance.iloc[0]['Feature'] == 'Unknown':
            st.warning("""
            ⚠️ Feature importance data not available yet.
            
            Please run the training script first:
            ```bash
            python train_model.py
            ```
            
            This will generate the feature importance rankings.
            """)
        else:
            # Top 10 features chart
            top_10 = predictor.feature_importance.head(10)
            
            fig = px.bar(
                top_10,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'},
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("📋 All Features")
            st.dataframe(
                predictor.feature_importance.style.background_gradient(subset=['Importance'], cmap='Blues'),
                use_container_width=True
            )
            
            # Feature insights
            st.subheader("💡 Key Insights")
            
            insights = {
                'ROI_Efficiency': "**Most critical factor!** Measures ROI generated per $1000 spent. Focus on maximizing this metric through efficient targeting and cost optimization.",
                'Acquisition_Cost': "Budget allocation matters significantly. Higher costs don't guarantee success - efficiency is key.",
                'CPM': "Cost Per Thousand Impressions indicates reach efficiency. Lower CPM with high engagement = winning combination.",
                'CPC': "Cost Per Click shows ad quality and targeting effectiveness. Optimize to reduce waste.",
                'CTR': "Click-Through Rate reflects ad relevance. Higher CTR = better audience resonance."
            }
            
            for idx, row in top_10.head(5).iterrows():
                feature = row['Feature']
                if feature in insights:
                    st.info(f"**{feature}:** {insights[feature]}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎯 Marketing Campaign Success Predictor | Built with Streamlit & Machine Learning</p>
        <p>Model trained on 200,000+ historical campaigns</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()