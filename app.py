import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import time # For simulating AI response delay

# --- Configuration and Environment Setup ---
load_dotenv() # Load environment variables from .env file

# Mock IBM Watson API credentials - replace with your actual keys for live integration
# Ensure these are set in your .env file or Streamlit Cloud secrets
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "your_mock_watsonx_api_key")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "your_mock_watsonx_project_id")

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="HealthAI: Intelligent Healthcare Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced UI ---
st.markdown("""
    <style>
        .main {
            background-color: #F0F2F6;
            color: #333;
            font-family: 'Inter', sans-serif;
            border-radius: 10px;
        }
        .sidebar .sidebar-content {
            background-color: #FFFFFF;
            border-right: 1px solid #E0E0E0;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 8px;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1A237E; /* Deep Indigo */
        }
        .chat-message-user {
            background-color: #DCF8C6; /* Light green for user messages */
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            text-align: right;
            margin-left: 20%;
        }
        .chat-message-ai {
            background-color: #E0E0E0; /* Light gray for AI messages */
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            text-align: left;
            margin-right: 20%;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #eee;
            border-radius: 8px;
            background-color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State for Patient Data and Chat History ---
if 'patient_profile' not in st.session_state:
    st.session_state.patient_profile = {
        "name": "",
        "age": 0,
        "gender": "Male",
        "medical_history": "",
        "current_medications": "",
        "allergies": ""
    }
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'health_metrics' not in st.session_state:
    st.session_state.health_metrics = pd.DataFrame()
if 'generated_treatment_plan' not in st.session_state:
    st.session_state.generated_treatment_plan = ""
if 'predicted_conditions' not in st.session_state:
    st.session_state.predicted_conditions = []

# --- Mock IBM Granite Model Integration ---
class MockGraniteModel:
    """
    A mock class to simulate IBM Granite-13b-instruct-v2 model's generate_text method.
    In a real application, this would be replaced with actual IBM Watson ML SDK calls.
    """
    def generate_text(self, prompt):
        # Simulate a delay for AI processing
        time.sleep(2)

        # Basic keyword-based responses for demonstration
        prompt_lower = prompt.lower()

        if "patient question:" in prompt_lower:
            query = prompt_lower.split("patient question:")[1].strip()
            if "fever" in query and "cough" in query:
                return """The symptoms you're describing (fever, cough, runny nose, headache, joint pain) are common with the flu or common cold.
                For most people, these conditions resolve with rest and fluids.
                However, if your symptoms are severe, worsen, or persist for more than 7-10 days,
                it's important to consult a healthcare professional. They can provide an accurate diagnosis and recommend appropriate treatment.
                This information is for general guidance and not a substitute for professional medical advice."""
            elif "headache" in query and "fatigue" in query and "fever" in query:
                return """Persistent headache, fatigue, and mild fever can be symptoms of various conditions, from viral infections to more serious issues.
                It's crucial to consult a doctor for a proper diagnosis. They might recommend further tests to determine the underlying cause and the best course of action.
                Remember, this AI provides general information and cannot diagnose."""
            elif "stomach pain" in query:
                return """Stomach pain can have many causes, from indigestion to more serious conditions like appendicitis or gallstones.
                If the pain is severe, persistent, accompanied by fever, vomiting, or blood in stool, seek immediate medical attention.
                For mild, occasional pain, over-the-counter antacids or dietary changes might help. Always consult a doctor for persistent or severe symptoms."""
            return """I can provide general health information, but I'm not a substitute for a medical professional.
            For specific medical advice, diagnosis, or treatment, please consult a qualified doctor or healthcare provider."""

        elif "predict potential health conditions" in prompt_lower:
            if "dry cough" in prompt_lower and "shortness of breath" in prompt_lower:
                return """1. COVID-19\nLikelihood: High\nBrief explanation: Symptoms are highly consistent with viral respiratory infection.\nRecommended next steps: Get tested, self-isolate, consult a doctor.\n\n2. Bronchitis\nLikelihood: Medium\nBrief explanation: Inflammation of bronchial tubes, often follows a cold.\nRecommended next steps: Rest, fluids, consider cough suppressants if severe.\n\n3. Pneumonia\nLikelihood: Medium\nBrief explanation: Lung infection that inflames air sacs.\nRecommended next steps: Seek medical attention for diagnosis and treatment."""
            elif "headache" in prompt_lower and "fatigue" in prompt_lower:
                return """1. Tension Headache\nLikelihood: High\nBrief explanation: Common type of headache often associated with stress.\nRecommended next steps: Rest, hydration, over-the-counter pain relievers.\n\n2. Migraine\nLikelihood: Medium\nBrief explanation: Severe headache often accompanied by nausea and sensitivity to light/sound.\nRecommended next steps: Avoid triggers, pain relief medication, consult doctor for prescription options.\n\n3. Viral Infection (e.g., common cold or flu)\nLikelihood: Medium\nBrief explanation: General body aches and fatigue are common with viral illnesses.\nRecommended next steps: Rest, fluids, monitor symptoms."""
            return """Based on the provided symptoms and patient data, I can't give a definitive prediction. Please consult a healthcare professional for diagnosis."""

        elif "generate a personalized treatment plan" in prompt_lower:
            if "mouth ulcer" in prompt_lower:
                return """
                **Personalized Treatment Plan for Mouth Ulcer:**

                1.  **Recommended Medications:**
                    * **Topical Gels/Pastes:** Over-the-counter products containing benzocaine (e.g., Orajel), triamcinolone acetonide (prescription), or amlexanox may reduce pain and inflammation. Apply as directed, typically 3-4 times daily after meals.
                    * **Antiseptic Mouthwashes:** Chlorhexidine gluconate or diluted salt water rinses (1/2 teaspoon salt in 1 cup warm water) can help keep the area clean and prevent secondary infection. Use 2-3 times daily.
                    * **Pain Relievers:** Over-the-counter pain relievers like ibuprofen or acetaminophen can help manage pain if discomfort is significant.

                2.  **Lifestyle Modifications:**
                    * **Avoid Irritants:** Steer clear of spicy, acidic, salty, or very hot foods/drinks that can irritate the ulcer.
                    * **Soft Diet:** Opt for soft, bland foods that are easy to chew and swallow.
                    * **Good Oral Hygiene:** Gently brush your teeth with a soft-bristled toothbrush. Avoid abrasive toothpaste.
                    * **Stress Reduction:** Stress can sometimes trigger or worsen mouth ulcers. Practice relaxation techniques like meditation or deep breathing.

                3.  **Follow-up Testing and Monitoring:**
                    * Monitor the ulcer for signs of healing. Most simple mouth ulcers heal within 1-2 weeks.
                    * If the ulcer does not heal within 3 weeks, becomes larger, more painful, or you develop new symptoms (like fever or swollen lymph nodes), consult your dentist or doctor for further evaluation to rule out other conditions.
                    * Recurrent ulcers may require further investigation to identify underlying causes (e.g., nutritional deficiencies, autoimmune conditions).

                4.  **Dietary Recommendations:**
                    * Ensure adequate intake of B vitamins (especially B12, folate) and iron, as deficiencies can contribute to ulcers. Consider supplements if dietary intake is insufficient, but consult a doctor first.
                    * Stay well-hydrated.

                5.  **Physical Activity Guidelines:**
                    * Maintain regular, moderate physical activity to support overall health and stress reduction. No specific restrictions due to mouth ulcers unless discomfort is severe.

                6.  **Mental Health Considerations:**
                    * Recognize that stress can impact physical health, including oral health. Manage stress through adequate sleep, hobbies, and if necessary, professional support.

                *Disclaimer: This plan is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.*
                """
            elif "hypertension" in prompt_lower:
                return """
                **Personalized Treatment Plan for Hypertension (High Blood Pressure):**

                1.  **Recommended Medications:**
                    * Your doctor may prescribe medications such as ACE inhibitors (e.g., lisinopril), ARBs (e.g., valsartan), calcium channel blockers (e.g., amlodipine), or diuretics (e.g., hydrochlorothiazide).
                    * Dosage and specific medication will be determined by your doctor based on your individual health profile and response. It's crucial to take medications exactly as prescribed and not to stop without consulting your doctor.

                2.  **Lifestyle Modifications:**
                    * **DASH Diet:** Adopt the Dietary Approaches to Stop Hypertension (DASH) eating plan, which emphasizes fruits, vegetables, whole grains, lean protein, and low-fat dairy, while limiting saturated and trans fats, cholesterol, and sodium.
                    * **Sodium Reduction:** Aim for less than 2,300 mg of sodium per day, ideally less than 1,500 mg for most adults. Read food labels carefully.
                    * **Weight Management:** If overweight or obese, losing even a small amount of weight can significantly lower blood pressure.
                    * **Regular Physical Activity:** Aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week.
                    * **Limit Alcohol:** If you drink alcohol, do so in moderation (up to one drink per day for women, up to two for men).
                    * **Quit Smoking:** Smoking significantly increases the risk of heart disease and stroke.
                    * **Stress Management:** Practice stress-reducing techniques such as meditation, yoga, or deep breathing.

                3.  **Follow-up Testing and Monitoring:**
                    * Regularly monitor your blood pressure at home and keep a record to share with your doctor.
                    * Schedule regular follow-up appointments with your healthcare provider to monitor your blood pressure, review your medication effectiveness, and adjust your treatment plan as needed.
                    * Regular blood tests (e.g., kidney function, electrolytes) may be performed to monitor medication side effects.

                4.  **Dietary Recommendations:**
                    * Increase intake of potassium-rich foods (e.g., bananas, spinach, potatoes), but consult your doctor if you have kidney issues or are on certain medications.
                    * Consume foods rich in magnesium and calcium.

                5.  **Physical Activity Guidelines:**
                    * Incorporate a mix of aerobic activities (walking, jogging, swimming) and strength training (at least twice a week).
                    * Consult your doctor before starting any new exercise regimen.

                6.  **Mental Health Considerations:**
                    * Manage stress effectively as chronic stress can contribute to high blood pressure. Seek support if you experience anxiety or depression.

                *Disclaimer: This plan is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment of hypertension.*
                """
            return "A personalized treatment plan cannot be generated with the provided information. Please ensure the condition and patient details are complete."

        return "I am unable to generate a response for this request at the moment. Please try rephrasing your query."

def init_granite_model():
    """
    Initializes the mock IBM Granite model.
    In a real scenario, this would involve authenticating with IBM Watson ML and
    loading the Granite-13b-instruct-v2 model.
    """
    # Placeholder for actual IBM Watson ML client initialization
    # from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
    # from ibm_watson_machine_learning.foundation_models import Model
    # model = Model(
    #     model_id=ModelTypes.GRANITE_13B_INSTRUCT_V2,
    #     params={...}, # Add necessary model parameters
    #     credentials={
    #         "url": "https://us-south.ml.cloud.ibm.com", # Or your region's endpoint
    #         "apikey": WATSONX_API_KEY
    #     },
    #     project_id=WATSONX_PROJECT_ID
    # )
    return MockGraniteModel() # Return the mock model instance

# Initialize the model once
if 'granite_model' not in st.session_state:
    st.session_state.granite_model = init_granite_model()
    st.write(f"Connecting to IBM Watson ML (Mock)... API Key: {'*' * (len(WATSONX_API_KEY) - 4)}{WATSONX_API_KEY[-4:]}, Project ID: {WATSONX_PROJECT_ID}")


# --- Core Functionalities ---

def predict_disease(symptoms, patient_profile):
    """
    Mocks disease prediction using the Granite model.
    """
    model = st.session_state.granite_model
    prompt = f"""
    As a medical AI assistant, predict potential health conditions based on the following patient data:

    Current Symptoms: {symptoms}
    Age: {patient_profile['age']}
    Gender: {patient_profile['gender']}
    Medical History: {patient_profile['medical_history'] if patient_profile['medical_history'] else 'None'}
    Recent Health Metrics:
    - (Mock data: Average Heart Rate: 70 bpm)
    - (Mock data: Average Blood Pressure: 120/80 mmHg)
    - (Mock data: Average Blood Glucose: 90 mg/dL)
    - Recently Reported Symptoms: {symptoms}

    Format your response as:
    1. Potential condition name
    2. Likelihood (High/Medium/Low)
    3. Brief explanation
    4. Recommended next steps

    Provide the top 3 most likely conditions based on the data provided.
    """
    with st.spinner("Analyzing symptoms and predicting potential conditions..."):
        prediction = model.generate_text(prompt)
    return prediction

def generate_treatment_plan(condition, patient_profile):
    """
    Mocks treatment plan generation using the Granite model.
    """
    model = st.session_state.granite_model
    prompt = f"""
    As a medical AI assistant, generate a personalized treatment plan for the following scenario:

    Patient Profile:
    - Condition: {condition}
    - Age: {patient_profile['age']}
    - Gender: {patient_profile['gender']}
    - Medical History: {patient_profile['medical_history'] if patient_profile['medical_history'] else 'None'}

    Create a comprehensive, evidence-based treatment plan that includes:
    1. Recommended medications (include dosage guidelines if appropriate)
    2. Lifestyle modifications
    3. Follow-up testing and monitoring
    4. Dietary recommendations
    5. Physical activity guidelines
    6. Mental health considerations

    Format this as a clear, structured treatment plan that follows current medical guidelines while being personalized to this patient's specific needs.
    """
    with st.spinner(f"Generating personalized treatment plan for {condition}..."):
        treatment_plan = model.generate_text(prompt)
    return treatment_plan

def answer_patient_query(query):
    """
    Mocks answering patient health questions using the Granite model.
    """
    model = st.session_state.granite_model
    prompt = f"""
    As a healthcare AI assistant, provide a helpful, accurate, and evidence-based response to the following patient question:

    PATIENT QUESTION: {query}

    Provide a clear, empathetic response that:
    - Directly addresses the question
    - Includes relevant medical facts
    - Acknowledges limitations (when appropriate)
    - Suggests when to seek professional medical advice
    - Avoids making definitive diagnoses
    - Uses accessible, non-technical language

    RESPONSE:
    """
    with st.spinner("Thinking..."):
        answer = model.generate_text(prompt)
    return answer

def generate_sample_health_metrics(num_days=30):
    """Generates realistic-looking sample health metrics over a period."""
    if not st.session_state.health_metrics.empty:
        return st.session_state.health_metrics

    dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=num_days, freq='D'))

    # Simulate variations for heart rate, BP, and glucose
    heart_rate = np.random.normal(70, 5, num_days).astype(int)
    systolic_bp = np.random.normal(120, 8, num_days).astype(int)
    diastolic_bp = np.random.normal(80, 5, num_days).astype(int)
    blood_glucose = np.random.normal(95, 10, num_days).astype(int)

    # Add some anomalies for demonstration
    if num_days > 5:
        heart_rate[-3] += 15 # Spike
        systolic_bp[-2] += 20 # Spike
        blood_glucose[-4] += 30 # Spike

    df = pd.DataFrame({
        'Date': dates,
        'Heart Rate (bpm)': heart_rate,
        'Systolic BP (mmHg)': systolic_bp,
        'Diastolic BP (mmHg)': diastolic_bp,
        'Blood Glucose (mg/dL)': blood_glucose
    })
    df['BP'] = df['Systolic BP (mmHg)'].astype(str) + '/' + df['Diastolic BP (mmHg)'].astype(str)

    st.session_state.health_metrics = df
    return df

# --- UI Components ---

st.title("🩺 HealthAI - Intelligent Healthcare Assistant")

# Sidebar for Patient Profile
st.sidebar.header("Patient Profile")
with st.sidebar.form("patient_profile_form"):
    st.session_state.patient_profile["name"] = st.text_input(
        "Name", value=st.session_state.patient_profile["name"]
    )
    st.session_state.patient_profile["age"] = st.number_input(
        "Age", min_value=0, max_value=120, value=st.session_state.patient_profile["age"]
    )
    st.session_state.patient_profile["gender"] = st.selectbox(
        "Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.patient_profile["gender"])
    )
    st.session_state.patient_profile["medical_history"] = st.text_area(
        "Medical History (e.g., Diabetes, Asthma)", value=st.session_state.patient_profile["medical_history"]
    )
    st.session_state.patient_profile["current_medications"] = st.text_area(
        "Current Medications", value=st.session_state.patient_profile["current_medications"]
    )
    st.session_state.patient_profile["allergies"] = st.text_area(
        "Allergies (e.g., Penicillin)", value=st.session_state.patient_profile["allergies"]
    )
    if st.form_submit_button("Update Profile"):
        st.sidebar.success("Patient Profile Updated!")

# Main content area with tabs
tab_names = ["Patient Chat", "Disease Prediction", "Treatment Plans", "Health Analytics"]
tabs = st.tabs(tab_names)

with tabs[0]: # Patient Chat
    st.header("24/7 Patient Support")
    st.write("Ask any health-related question for immediate assistance.")

    # Display chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="chat-message-user">🙋‍♂️ You: {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">🤖 HealthAI: {message}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    user_query = st.text_input("Ask your health question...", key="patient_chat_input")
    if st.button("Send Query"):
        if user_query:
            st.session_state.chat_history.append(("user", user_query))
            ai_response = answer_patient_query(user_query)
            st.session_state.chat_history.append(("ai", ai_response))
            st.experimental_rerun() # Rerun to clear input and update chat history

with tabs[1]: # Disease Prediction
    st.header("Disease Prediction System")
    st.write("Enter symptoms and patient data to receive potential condition predictions.")

    symptoms_input = st.text_area(
        "Current Symptoms",
        value="Describe symptoms in detail (e.g., persistent headache for 3 days, fatigue, mild fever of 99.5°F)",
        height=150,
        key="symptoms_input"
    )

    if st.button("Generate Prediction"):
        if symptoms_input:
            predicted_output = predict_disease(symptoms_input, st.session_state.patient_profile)
            st.session_state.predicted_conditions = predicted_output.split('\n\n') # Split into individual conditions
        else:
            st.warning("Please enter symptoms to generate a prediction.")

    if st.session_state.predicted_conditions:
        st.subheader("Potential Conditions")
        for condition_info in st.session_state.predicted_conditions:
            st.markdown(f"**{condition_info}**")


with tabs[2]: # Treatment Plans
    st.header("Personalized Treatment Plan Generator")
    st.write("Generate customized treatment recommendations based on specific conditions.")

    medical_condition = st.text_input(
        "Medical Condition",
        value="Mouth Ulcer", # Example pre-fill
        key="medical_condition_input"
    )

    if st.button("Generate Treatment Plan"):
        if medical_condition:
            st.session_state.generated_treatment_plan = generate_treatment_plan(medical_condition, st.session_state.patient_profile)
        else:
            st.warning("Please enter a medical condition to generate a treatment plan.")

    if st.session_state.generated_treatment_plan:
        st.subheader("Personalized Treatment Plan")
        st.markdown(st.session_state.generated_treatment_plan)

with tabs[3]: # Health Analytics
    st.header("Health Analytics Dashboard")
    st.write("Visualize your vital signs over time and receive AI-generated insights.")

    # Generate or load health metrics
    health_metrics_df = generate_sample_health_metrics()

    if not health_metrics_df.empty:
        st.subheader("Health Metrics Trends")

        # Heart Rate Trend Line Chart
        fig_hr = px.line(health_metrics_df, x='Date', y='Heart Rate (bpm)', title='Heart Rate Trend',
                         labels={'Heart Rate (bpm)': 'Heart Rate', 'Date': 'Date'},
                         line_shape='spline')
        fig_hr.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Average Healthy HR")
        st.plotly_chart(fig_hr, use_container_width=True)

        # Blood Pressure Dual-Line Chart
        fig_bp = go.Figure()
        fig_bp.add_trace(go.Scatter(x=health_metrics_df['Date'], y=health_metrics_df['Systolic BP (mmHg)'],
                                    mode='lines+markers', name='Systolic BP'))
        fig_bp.add_trace(go.Scatter(x=health_metrics_df['Date'], y=health_metrics_df['Diastolic BP (mmHg)'],
                                    mode='lines+markers', name='Diastolic BP'))
        fig_bp.update_layout(title='Blood Pressure Trend',
                              yaxis_title='BP (mmHg)', xaxis_title='Date')
        fig_bp.add_hrect(y0=120, y1=129, line_width=0, fillcolor="yellow", opacity=0.2, annotation_text="Elevated Systolic")
        fig_bp.add_hrect(y0=80, y1=80, line_width=0, fillcolor="red", opacity=0.2, annotation_text="Elevated Diastolic")
        st.plotly_chart(fig_bp, use_container_width=True)

        # Blood Glucose Trend Line Chart
        fig_glucose = px.line(health_metrics_df, x='Date', y='Blood Glucose (mg/dL)', title='Blood Glucose Trend',
                              labels={'Blood Glucose (mg/dL)': 'Blood Glucose', 'Date': 'Date'},
                              line_shape='spline')
        fig_glucose.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Pre-diabetic Threshold")
        st.plotly_chart(fig_glucose, use_container_width=True)

        st.subheader("Health Metrics Summary")
        col1, col2, col3 = st.columns(3)

        # Calculate current values and basic trends
        current_hr = health_metrics_df['Heart Rate (bpm)'].iloc[-1]
        avg_hr_prev_week = health_metrics_df['Heart Rate (bpm)'].iloc[-8:-1].mean()
        hr_delta = current_hr - avg_hr_prev_week
        hr_status = "Normal" if 60 <= current_hr <= 100 else "Abnormal"

        current_systolic = health_metrics_df['Systolic BP (mmHg)'].iloc[-1]
        current_diastolic = health_metrics_df['Diastolic BP (mmHg)'].iloc[-1]

        current_glucose = health_metrics_df['Blood Glucose (mg/dL)'].iloc[-1]
        avg_glucose_prev_week = health_metrics_df['Blood Glucose (mg/dL)'].iloc[-8:-1].mean()
        glucose_delta = current_glucose - avg_glucose_prev_week
        glucose_status = "Normal" if 70 <= current_glucose <= 100 else "Abnormal"


        with col1:
            st.metric(label="Current Heart Rate", value=f"{current_hr} bpm", delta=f"{hr_delta:.1f} from last week")
            st.write(f"Status: **{hr_status}**")
        with col2:
            st.metric(label="Current Blood Pressure", value=f"{current_systolic}/{current_diastolic} mmHg")
            bp_status = "Normal"
            if current_systolic >= 130 or current_diastolic >= 80:
                bp_status = "Elevated/High"
            st.write(f"Status: **{bp_status}**")
        with col3:
            st.metric(label="Current Blood Glucose", value=f"{current_glucose} mg/dL", delta=f"{glucose_delta:.1f} from last week")
            st.write(f"Status: **{glucose_status}**")

        st.subheader("AI-Generated Insights (Mock)")
        st.info("""
        Based on your recent health metrics, your heart rate is generally stable, but we observed a slight increase in the last few days.
        Your blood pressure is currently within a healthy range. Blood glucose levels show a recent minor spike; ensuring consistent diet is recommended.

        **Recommendations:**
        * Continue to monitor heart rate, especially if you notice palpitations or shortness of breath.
        * Maintain your current lifestyle to keep blood pressure healthy.
        * Focus on consistent meal timings and balanced nutrition to stabilize blood glucose. If spikes persist, consult your doctor.
        """)
    else:
        st.write("No health metrics data available. Generate sample data or upload yours.")

# --- Footer ---
st.markdown("---")
st.markdown("HealthAI is powered by intelligent AI and aims to provide helpful health information. Always consult a healthcare professional for diagnosis and treatment.")