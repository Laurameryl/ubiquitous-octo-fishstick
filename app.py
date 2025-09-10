# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="ML Classifier App",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Machine Learning Classifier")
st.markdown("""
This application demonstrates a machine learning classifier. 
Please input the feature values and click the 'Predict' button to see the prediction.
""")

# Create a sample dataset for demonstration
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample features
    age = np.random.randint(18, 70, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    experience = np.random.randint(0, 40, n_samples)
    
    # Create a target variable based on some rules
    target = ((income > 55000) & (education.isin(['Master', 'PhD'])) | 
             ((experience > 15) & (income > 45000))).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'experience': experience,
        'target': target
    })
    
    return data

# Train a model for demonstration
@st.cache_resource
def train_model():
    data = create_sample_data()
    
    # Encode categorical features
    le = LabelEncoder()
    data['education_encoded'] = le.fit_transform(data['education'])
    
    # Prepare features and target
    X = data[['age', 'income', 'education_encoded', 'experience']]
    y = data['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save encoders and model for later use
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    return model, le

# Load the model and encoder
try:
    model, le = train_model()
except:
    st.error("Error loading the model. Please check the model files.")
    st.stop()

# Create input form
with st.form("input_form"):
    st.header("Feature Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=70, value=30)
        income = st.number_input("Income", min_value=10000, max_value=150000, value=50000, step=1000)
    
    with col2:
        education = st.selectbox("Education Level", 
                                options=['High School', 'Bachelor', 'Master', 'PhD'])
        experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5)
    
    # Submit button
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            'education': [education],
            'experience': [experience]
        })
        
        # Encode education
        input_data['education_encoded'] = le.transform(input_data['education'])
        
        # Make prediction
        features = input_data[['age', 'income', 'education_encoded', 'experience']]
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Display results
        st.success("Prediction completed!")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.subheader("Prediction")
            if prediction[0] == 1:
                st.markdown("**Class: Positive** ✅")
            else:
                st.markdown("**Class: Negative** ❌")
        
        with result_col2:
            st.subheader("Confidence")
            st.write(f"Positive class probability: {probability[0][1]:.2%}")
            st.progress(probability[0][1])
        
        # Show input values
        with st.expander("See input values"):
            st.write(input_data[['age', 'income', 'education', 'experience']])

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info("""
This is a demonstration of a machine learning classifier deployed using Streamlit.

**Features used:**
- Age
- Income
- Education Level
- Years of Experience

The model is a Random Forest classifier trained on synthetic data.
""")

# Add deployment instructions
st.sidebar.header("Deployment")
st.sidebar.info("""
This app can be deployed on Streamlit Share by:

1. Creating a GitHub repository
2. Pushing this code to the repository
3. Connecting the repository to Streamlit Share
4. Deploying the application
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
