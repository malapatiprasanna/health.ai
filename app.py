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
            color: #1A237E;
            /* Deep Indigo */
        }
        .chat-message-user {
            background-color: #DCF8C6;
            /* Light green for user messages */
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            text-align: right;
            margin-left: 20%;
        }
        .chat-message-ai {
            background-color: #E0E0E0;
            /* Light gray for AI messages */
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
                it's important to consult a healthcare professional.
                They can provide an accurate diagnosis and recommend appropriate treatment.
                This information is for general guidance and not a substitute for professional medical advice."""
            elif "headache" in query and "fatigue" in query and "fever" in query:
                return """Persistent headache, fatigue, and mild fever can be symptoms of various conditions, from viral infections to more serious issues.
                It's crucial to consult a doctor for a proper diagnosis.
                They might recommend further tests to determine the underlying cause and the best course of action.
                Remember, this AI provides general information and cannot diagnose."""
            elif "stomach pain" in query and "severe stomach pain":
                return """Stomach pain can have many causes, from indigestion to more serious conditions like appendicitis or gallstones.
                If the pain is severe, persistent, accompanied by fever, vomiting, or blood in stool, seek immediate medical attention.
                For mild, occasional pain, over-the-counter antacids or dietary changes might help.
                Always consult a doctor for persistent or severe symptoms."""
            elif "persistent cough" in query and "low-grade fever" in query:
                return """A persistent cough and low-grade fever can be symptoms of various conditions, including a common cold, bronchitis, or even early stages of a viral infection like the flu or COVID-19.
                It's important to monitor your symptoms. If they worsen, you develop shortness of breath, severe chest pain, or the fever increases, please consult a healthcare professional for a proper diagnosis and advice.
                Rest and staying hydrated are generally recommended."""
            elif "fever" in query and "runny nose" in query and "headache" in query and "joint pain" in query:
                return """"The symptoms you're describing (fever, cough, runny nose, headache, joint pain) are common with the flu or common cold. 
                For most people, these conditions resolve with rest and fluids. However, if your symptoms are severe, worsen, or persist for more than 7-10 days, it's important to consult a healthcare professional. They can provide an accurate diagnosis and recommend appropriate treatment. 
                This information is for general guidance and not a substitute for professional medical advice."""
            elif "headache" in query and "fatigue" in query and "mild fever" in query:
                return """Persistent headache, fatigue, and mild fever can be symptoms of various conditions, from viral infections to more serious issues. 
                It's crucial to consult a doctor for a proper diagnosis. They might recommend further tests to determine the underlying cause and the best course of action. 
                Remember, this AI provides general information and cannot diagnose."""
            elif "fever" in query:
                return """Fever is a temporary rise in body temperature, often due to an illness. It’s a symptom — not a disease — and is usually a sign that your body is fighting an infection.

                🌡️ Normal vs. Fever
                Normal body temp: ~98.6°F (37°C)

                Fever: Usually defined as:

                Low-grade: 99.5°F to 100.9°F (37.5–38.3°C)

                Moderate: 101°F to 103°F (38.3–39.4°C)

                High: 104°F or more (≥ 40°C)

                🩺 Common Causes of Fever
                Viral infections (cold, flu, COVID-19)

                Bacterial infections (UTI, strep throat)

                Heat exhaustion

                Inflammatory conditions (e.g. rheumatoid arthritis)

                Vaccinations (as a side effect)

                🏠 Home Care Tips
                Rest: Let your body recover.

                Hydrate: Drink plenty of fluids (water, electrolyte drinks, soups).

                Dress lightly: Avoid heavy clothing or blankets.

                Cool compress: Use a damp washcloth on forehead/neck.

                Medications:

                Paracetamol (acetaminophen) or Ibuprofen can reduce fever.

                🚨 When to See a Doctor
                Fever > 103°F (39.4°C)

                Lasts more than 3 days

                Severe headache, rash, stiff neck

                Difficulty breathing or chest pain

                Fever with confusion, seizures, or persistent vomiting

                For infants under 3 months, a fever is always an emergency"""
            elif "cough" in query:
                return """🩺 Common Causes
                Viral infections (cold, flu, COVID-19)

                Bacterial infections (bronchitis, pneumonia)

                Allergies or asthma

                Acid reflux (GERD)

                Postnasal drip

                Environmental irritants (smoke, pollution)

                🏠 Home Remedies
                Warm fluids: Ginger tea, warm water with honey and lemon

                Steam inhalation: Helps loosen mucus

                Honey (for >1 yr old): Soothes throat and suppresses cough

                Saltwater gargle: Relieves throat irritation

                Avoid irritants: Smoke, perfumes, and dust

                💊 Medications
                Dry Cough:

                Dextromethorphan (cough suppressants)

                Antihistamines (if allergy-related)

                Wet Cough:

                Expectorants like guaifenesin to thin mucus

                Antibiotics if it's bacterial (only with a prescription)

                """
            elif "cold" in query:
                return """🤧 Common Symptoms
                Runny or stuffy nose

                Sneezing

                Sore throat

                Cough

                Mild fever (sometimes)

                Fatigue

                Headache

                Watery eyes
                🏠 Home Remedies
                Rest: Let your immune system work

                Hydration: Warm water, soup, herbal teas

                Steam inhalation: Helps with congestion

                Saltwater gargle: Soothes sore throat

                Honey + lemon: Eases throat and cough

                Tulsi, ginger, turmeric milk: Natural anti-inflammatory options

                💊 Over-the-Counter Medicines
                Symptom	Medicine Example
                Runny nose	Antihistamines (cetirizine, loratadine)
                Nasal congestion	Decongestants (phenylephrine, xylometazoline)
                Cough	Dextromethorphan, guaifenesin
                Fever/body ache	Paracetamol, ibuprofen

                ⚠️ Avoid antibiotics — they do not work against viruses."""
            elif "cold" in query and "fever" in query:
                return """🏠 Home Remedies
                ✅ Works for both cold & cough relief:

                Ginger tea with honey

                Tulsi + turmeric + pepper concoction

                Steam inhalation (add Vicks or eucalyptus oil)

                Honey + lemon in warm water (especially for dry cough)

                Saltwater gargle (for throat relief)

                Stay hydrated: warm water, soups, ORS

                Rest: allow your body to heal

                💊 Medicines (OTC)
                Symptom	Medicine Type	Examples
                Fever, body ache	Pain relievers	Paracetamol, Ibuprofen
                Runny nose	Antihistamines	Cetirizine, Loratadine
                Blocked nose	Decongestants	Xylometazoline nasal spray
                Dry cough	Cough suppressants	Dextromethorphan
                Wet cough	Expectorants	Guaifenesin

                🔹 Use syrups like Benadryl, Ascoril, or Grilinctus (as per need).
                🔹 Always consult a doctor if symptoms last >7 days or worsen.

                🍲 Diet Suggestions
                Warm liquids: soup, tea, turmeric milk

                Soft foods: khichdi, dal, porridge

                Avoid cold drinks, ice cream, fried and spicy foods"""
            elif "feeling headache" in query or "headache" in query: # Added specific response for headache
                return """
                **Quick Relief Tips for Headaches:**
                1.  **Hydration:** Drink a glass or two of water. Dehydration is a common cause of headaches.
                2.  **Rest:** Lie down in a quiet, dark room and close your eyes. Try to sleep or just relax without screen time or loud noises.
                3.  **Cold or Warm Compress:** Apply a cold compress on your forehead or temples (often helpful for migraines). Use a warm compress on the back of the neck (can help with tension headaches).
                4.  **Caffeine:** A small amount of caffeine (like in tea or coffee) can help relieve some headaches, but avoid overuse as it can also cause withdrawal headaches.
                5.  **Over-the-counter Medication:** Consider paracetamol (acetaminophen), ibuprofen, or aspirin—if you don’t have any medical conditions that prevent their use and if they are appropriate for your age.

                **🩺 When to Seek Medical Attention:**
                * Headache is severe and sudden (thunderclap headache).
                * It’s your worst headache ever.
                * You have headache accompanied by fever, stiff neck, rash, confusion, seizures, double vision, weakness, numbness, or difficulty speaking.
                * Headache follows a head injury.
                * You get headaches frequently or they are worsening over time.
                * New headaches if you are over 50.

                This information is for general guidance and not a substitute for professional medical advice. If you have concerns, please consult a qualified doctor or healthcare provider.
                """
            return """I can provide general health information, but I'm not a substitute for a medical professional.
            For specific medical advice, diagnosis, or treatment, please consult a qualified doctor or healthcare provider."""

        elif "predict potential health conditions" in prompt_lower:
            if "dry cough" in prompt_lower and "shortness of breath" in prompt_lower:
                return """1. COVID-19\nLikelihood: High\nBrief explanation: Symptoms are highly consistent with viral respiratory infection.\nRecommended next steps: Get tested, self-isolate, consult a doctor.\n\n2. Bronchitis\nLikelihood: Medium\nBrief explanation: Inflammation of bronchial tubes, often follows a cold.\nRecommended next steps: Rest, fluids, consider cough suppressants if severe.\n\n3. Pneumonia\nLikelihood: Medium\nBrief explanation: Lung infection that inflames air sacs.\nRecommended next steps: Seek medical attention for diagnosis and treatment."""
            elif "headache" in prompt_lower and "fatigue" in prompt_lower and "fever" in prompt_lower:
                return """1. Tension Headache\nLikelihood: High\nBrief explanation: Common type of headache often associated with stress.\nRecommended next steps: Rest, hydration, over-the-counter pain relievers.\n\n2. Migraine\nLikelihood: Medium\nBrief explanation: Severe headache often accompanied by nausea and sensitivity to light/sound.\nRecommended next steps: Avoid triggers, pain relief medication, consult doctor for prescription options.\n\n3. Viral Infection (e.g., common cold or flu)\nLikelihood: Medium\nBrief explanation: General body aches and fatigue are common with viral illnesses.\nRecommended next steps: Rest, fluids, monitor symptoms."""
            elif "fever" in prompt_lower and "body aches" in prompt_lower and "cough" in prompt_lower:
                return """1. Influenza (Flu)\nLikelihood: High\nBrief explanation: Acute respiratory illness caused by influenza viruses.\nRecommended next steps: Rest, fluids, antiviral medication if prescribed, avoid contact with others, consult doctor if severe.\n\n2. Common Cold\nLikelihood: Medium\nBrief explanation: Milder viral infection of the nose and throat.\nRecommended next steps: Rest, fluids, symptom relief.\n\n3. COVID-19\nLikelihood: Medium\nBrief explanation: Viral respiratory illness with similar symptoms to flu.\nRecommended next steps: Get tested, self-isolate, consult a doctor."""
            elif "unexplained weight loss" in prompt_lower and "fatigue" in prompt_lower and "night sweats" in prompt_lower:
                return """1. HIV (Human Immunodeficiency Virus)\nLikelihood: High\nBrief explanation: A virus that attacks the body's immune system. Early symptoms can be flu-like.\nRecommended next steps: Get tested immediately, seek medical consultation for antiretroviral therapy (ART).\n\n2. Tuberculosis (TB)\nLikelihood: Medium\nBrief explanation: A bacterial infection that usually attacks the lungs. Symptoms include persistent cough, fever, night sweats, weight loss.\nRecommended next steps: Seek medical evaluation and testing (e.g., TB skin test, chest X-ray).\n\n3. Cancer\nLikelihood: Medium\nBrief explanation: Unexplained weight loss, fatigue, and night sweats can be general symptoms of various cancers.\nRecommended next steps: Consult a doctor for comprehensive diagnostic workup."""
            elif "persistent cough" in prompt_lower and "chest pain" in prompt_lower and "shortness of breath" in prompt_lower:
                return """1. Pneumonia\nLikelihood: High\nBrief explanation: Infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.\nRecommended next steps: Seek immediate medical attention for diagnosis and treatment (antibiotics/antivirals).\n\n2. Tuberculosis (TB)\nLikelihood: High\nBrief explanation: A bacterial infection primarily affecting the lungs, leading to chronic cough, chest pain, and other systemic symptoms.\nRecommended next steps: Consult a doctor for TB testing and treatment.\n\n3. Bronchitis\nLikelihood: Medium\nBrief explanation: Inflammation of the lining of your bronchial tubes, which carry air to and from your lungs.\nRecommended next steps: Rest, fluids, cough suppressants, consult doctor if symptoms persist."""
            elif "fever" in prompt_lower and "chills" in prompt_lower and "sweating" in prompt_lower and "muscle pain" in prompt_lower and "travel history to malaria-prone area" in prompt_lower:
                return """1. Malaria\nLikelihood: High\nBrief explanation: A serious mosquito-borne disease caused by a parasite. Characterized by fever, chills, sweating, and flu-like illness, especially after travel to endemic areas.\nRecommended next steps: Seek urgent medical attention, inform doctor about travel history, immediate blood test for malaria parasites.\n\n2. Dengue Fever\nLikelihood: Medium\nBrief explanation: A mosquito-borne viral infection causing flu-like illness, severe muscle and joint pain, rash, and fever. More common in tropical and subtropical regions.\nRecommended next steps: Consult a doctor, symptomatic treatment, monitoring for warning signs.\n\n3. Influenza (Flu)\nLikelihood: Low\nBrief explanation: While symptoms can overlap, the travel history makes malaria or dengue more likely.\nRecommended next steps: Standard flu treatment if diagnosed, but prioritize ruling out tropical diseases."""
            elif "unexplained weight loss" in prompt_lower and "fatigue" in prompt_lower and "changes in bowel habits" in prompt_lower:
                return """1. Colon Cancer\nLikelihood: High\nBrief explanation: Cancer of the large intestine. Symptoms can include changes in bowel habits, blood in stool, fatigue, and unexplained weight loss.\nRecommended next steps: Consult a gastroenterologist for screening and diagnostic tests (e.g., colonoscopy).\n\n2. Pancreatic Cancer\nLikelihood: Medium\nBrief explanation: Often presents with non-specific symptoms like weight loss, fatigue, and abdominal pain. Can also affect digestion leading to bowel changes.\nRecommended next steps: Seek medical evaluation, potentially imaging and specific blood tests.\n\n3. Irritable Bowel Syndrome (IBS)\nLikelihood: Low\nBrief explanation: While IBS causes changes in bowel habits, unexplained weight loss and significant fatigue are less typical primary symptoms, but can occur due to chronic discomfort.\nRecommended next steps: Consult a doctor for differential diagnosis and management."""
            elif "frequent urination" in prompt_lower and "increased thirst" in prompt_lower and "blurred vision" in prompt_lower:
                return """1. Diabetes (Type 2)\nLikelihood: High\nBrief explanation: A chronic condition that affects the way your body processes blood sugar (glucose). Classic symptoms include increased thirst, frequent urination, and blurred vision.\nRecommended next steps: Get blood sugar tested (fasting glucose, HbA1c), consult an endocrinologist or primary care doctor for management.\n\n2. Diabetes Insipidus\nLikelihood: Low\nBrief explanation: A rare condition where your body can't balance fluids, leading to extreme thirst and frequent urination. Not related to blood sugar.\nRecommended next steps: Medical evaluation to differentiate from diabetes mellitus.\n\n3. Urinary Tract Infection (UTI)\nLikelihood: Low\nBrief explanation: Can cause frequent urination and discomfort, but typically not increased thirst or blurred vision.\nRecommended next steps: Urinalysis if UTI is suspected."""
            elif "high fever" in prompt_lower and "severe headache" in prompt_lower and "joint and muscle pain" in prompt_lower and "skin rash" in prompt_lower:
                return """1. Dengue Fever\nLikelihood: High\nBrief explanation: A mosquito-borne viral infection prevalent in tropical and subtropical regions. Characterized by high fever, severe headache, joint/muscle pain ("breakbone fever"), and a rash.\nRecommended next steps: Consult a doctor immediately, manage symptoms with pain relievers (avoid NSAIDs), monitor for warning signs (e.g., severe abdominal pain, bleeding).\n\n2. Chikungunya\nLikelihood: Medium\nBrief explanation: Another mosquito-borne viral infection with similar symptoms, but joint pain is often more prominent and debilitating.\nRecommended next steps: Medical evaluation for diagnosis and symptomatic treatment.\n\n3. Measles\nLikelihood: Low\nBrief explanation: While it causes fever and rash, measles typically presents with a specific sequence of symptoms including cough, coryza, conjunctivitis before the rash, and less severe joint pain.\nRecommended next steps: Consult doctor if measles is suspected."""
            return """Based on the provided symptoms and patient data, I can't give a definitive prediction.
            Please consult a healthcare professional for diagnosis."""

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
            elif "asthma" in prompt_lower:
                return """
                **Personalized Treatment Plan for Asthma:**

                1.  **Recommended Medications:**
                    * **Reliever Inhalers (Short-Acting Beta-Agonists - SABAs):** Such as albuterol, used for quick relief of symptoms.
                    * **Controller Inhalers (Inhaled Corticosteroids - ICS):** Such as fluticasone or budesonide, used daily to reduce inflammation and prevent attacks.
                    * **Combination Inhalers (ICS + Long-Acting Beta-Agonists - LABAs):** For more severe asthma.
                    * **Oral Corticosteroids:** For severe exacerbations.
                    * **Leukotriene Modifiers:** Such as montelukast, can help reduce inflammation and improve lung function.
                    * **Biologics:** For severe allergic or eosinophilic asthma.
                    * *Dosage and specific medication will be determined by your doctor based on your asthma severity and triggers. Adherence to the prescribed regimen is crucial.*
                2.  **Lifestyle Modifications:**
                    * **Identify and Avoid Triggers:** Common triggers include allergens (pollen, dust mites, pet dander), irritants (smoke, pollution, strong odors), exercise, cold air, and stress.
                    * **Maintain a Clean Environment:** Regularly clean your home to reduce dust mites and mold.
                    * **Avoid Smoking and Secondhand Smoke.**
                    * **Get Vaccinated:** Annual flu shots and pneumonia vaccines are recommended.
                3.  **Follow-up Testing and Monitoring:**
                    * **Regular Doctor Visits:** For ongoing assessment of asthma control and adjustment of medication.
                    * **Peak Flow Monitoring:** Use a peak flow meter daily to monitor lung function and detect early signs of worsening asthma.
                    * **Asthma Action Plan:** Develop a written plan with your doctor that outlines daily management, how to handle worsening symptoms, and when to seek emergency care.
                4.  **Dietary Recommendations:**
                    * No specific diet cures asthma, but a healthy, balanced diet rich in fruits, vegetables, and whole grains can support overall health.
                    * Some studies suggest vitamin D and omega-3 fatty acids might have a protective effect, but consult your doctor before taking supplements.
                5.  **Physical Activity Guidelines:**
                    * Regular physical activity is encouraged, as it strengthens the lungs and improves overall fitness.
                    * Warm-up exercises before activity and using a reliever inhaler (if prescribed) can help prevent exercise-induced asthma.
                    * Choose activities suitable for your condition; swimming is often well-tolerated.
                6.  **Mental Health Considerations:**
                    * Stress and anxiety can trigger asthma symptoms. Practice stress-reduction techniques like deep breathing, meditation, or yoga.
                    * Seek support if you experience anxiety or depression related to your asthma.
                *Disclaimer: This plan is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment of asthma.*
                """
            elif "diabetes" in prompt_lower:
                return """
                **Personalized Treatment Plan for Diabetes (Type 2):**

                1.  **Recommended Medications:**
                    * **Metformin:** Often the first-line medication, helps reduce glucose production by the liver and improve insulin sensitivity.
                    * **Sulfonylureas (e.g., glipizide):** Stimulate the pancreas to produce more insulin.
                    * **DPP-4 Inhibitors (e.g., sitagliptin):** Help the body make more insulin and reduce glucose.
                    * **SGLT2 Inhibitors (e.g., empagliflozin):** Help the kidneys remove glucose from the body.
                    * **GLP-1 Receptor Agonists (e.g., liraglutide):** Slow digestion and help the body produce more insulin.
                    * **Insulin Therapy:** May be required if other medications are not sufficient to control blood sugar.
                    * *Medication choice and dosage depend on individual factors, blood sugar levels, and other health conditions. Strict adherence and regular monitoring are essential.*
                2.  **Lifestyle Modifications:**
                    * **Healthy Eating:** Focus on a balanced diet rich in non-starchy vegetables, lean proteins, and whole grains. Limit refined carbohydrates, sugary drinks, and unhealthy fats.
                    * **Portion Control:** Manage portion sizes to control calorie and carbohydrate intake.
                    * **Regular Physical Activity:** Aim for at least 150 minutes of moderate-intensity aerobic activity per week, plus muscle-strengthening activities at least two days a week.
                    * **Weight Management:** If overweight or obese, even modest weight loss can significantly improve blood sugar control.
                    * **Quit Smoking:** Smoking worsens diabetes complications.
                    * **Limit Alcohol:** Consume alcohol in moderation, if at all.
                3.  **Follow-up Testing and Monitoring:**
                    * **Regular Blood Glucose Monitoring:** Home blood glucose monitoring (HBGM) as advised by your doctor.
                    * **HbA1c Testing:** Every 3-6 months to assess long-term blood sugar control.
                    * **Regular Doctor Visits:** For medication adjustments, screening for complications (eyes, kidneys, nerves, feet), and overall health assessment.
                    * **Blood Pressure and Cholesterol Monitoring:** Manage these to reduce cardiovascular risk.
                4.  **Dietary Recommendations:**
                    * **Carbohydrate Counting:** Learn to count carbohydrates to manage blood sugar levels, especially if on insulin.
                    * **Glycemic Index (GI):** Understand how different foods affect blood sugar and choose lower GI options.
                    * **Fiber Intake:** Increase fiber-rich foods (vegetables, fruits, whole grains) to help regulate blood sugar.
                5.  **Physical Activity Guidelines:**
                    * Incorporate both aerobic (walking, cycling, swimming) and resistance training.
                    * Monitor blood sugar before, during, and after exercise, especially if on insulin or certain medications, to prevent hypoglycemia.
                6.  **Mental Health Considerations:**
                    * Managing a chronic condition like diabetes can be stressful. Seek support for anxiety or depression.
                    * Diabetes education and support groups can provide valuable resources and coping strategies.
                *Disclaimer: This plan is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment of diabetes.*
                """
            elif "cancer" in prompt_lower:
                return """
                **Personalized Treatment Plan for Cancer:**

                1.  **Recommended Treatments (Varies Greatly by Cancer Type and Stage):**
                    * **Surgery:** To remove the tumor and surrounding tissue.
                    * **Chemotherapy:** Medications that kill cancer cells or slow their growth. Administered orally or intravenously.
                    * **Radiation Therapy:** Uses high-energy rays to kill cancer cells. Can be external beam radiation or brachytherapy (internal).
                    * **Targeted Therapy:** Drugs that specifically target cancer cells' unique vulnerabilities, often with fewer side effects than chemotherapy.
                    * **Immunotherapy:** Boosts the body's own immune system to fight cancer.
                    * **Hormone Therapy:** Used for hormone-sensitive cancers (e.g., breast, prostate) to block hormone production or action.
                    * **Stem Cell Transplant:** Used for certain blood cancers to replace diseased bone marrow.
                    * *The specific treatment regimen will be determined by a multidisciplinary team of oncologists, surgeons, radiation therapists, etc., based on the cancer type, stage, patient's overall health, and genetic markers.*
                2.  **Lifestyle Modifications:**
                    * **Nutritional Support:** Maintain a balanced diet, often with the help of a dietitian, to manage treatment side effects and maintain strength.
                    * **Gentle Physical Activity:** As tolerated and approved by your medical team, light exercise can help with fatigue and mood.
                    * **Avoid Smoking and Alcohol:** Crucial for improving treatment outcomes and preventing recurrence or new cancers.
                    * **Rest:** Adequate rest is vital during treatment.
                3.  **Follow-up Testing and Monitoring:**
                    * **Regular Imaging (CT, MRI, PET scans):** To monitor treatment response and detect recurrence.
                    * **Blood Tests (Tumor Markers, CBC):** To monitor cancer activity, treatment side effects, and overall health.
                    * **Frequent Oncologist Visits:** For treatment adjustments, symptom management, and long-term surveillance.
                    * **Genetic Testing:** May be recommended to guide targeted therapies.
                4.  **Dietary Recommendations:**
                    * Focus on nutrient-dense foods, small frequent meals to manage nausea and appetite changes.
                    * Stay hydrated.
                    * Avoid raw or undercooked foods if your immune system is compromised.
                5.  **Physical Activity Guidelines:**
                    * Tailored exercise plans are important. Start with short walks and gradually increase activity as tolerated.
                    * Consult with your care team before starting any new exercise program.
                6.  **Mental Health Considerations:**
                    * Cancer diagnosis and treatment are emotionally challenging. Seek psychological support, counseling, or support groups.
                    * Manage stress through relaxation techniques, mindfulness, or professional therapy.
                    * Discuss concerns about pain, fatigue, and other side effects with your medical team.
                *Disclaimer: This plan is for informational purposes only and is not a substitute for professional medical advice. Cancer treatment is highly individualized and requires comprehensive care from a specialized medical team.*
                """
            elif "migraine" in prompt_lower:
                return """
                **Personalized Treatment Plan for Migraine:**

                1.  **Recommended Medications:**
                    * **Acute Treatment (for attacks):**
                        * **Over-the-counter pain relievers:** Ibuprofen, naproxen, acetaminophen (for mild migraines).
                        * **Triptans (e.g., sumatriptan, zolmitriptan):** Specific migraine medications that relieve pain, nausea, and light/sound sensitivity.
                        * **CGRP Receptor Antagonists (e.g., ubrogepant, rimegepant):** Newer drugs for acute treatment.
                        * **Ergotamines:** Used for severe, prolonged attacks.
                        * **Anti-nausea medications:** If nausea and vomiting are significant.
                    * **Preventive Treatment (taken regularly to reduce frequency/severity):**
                        * **Beta-blockers (e.g., propranolol):** Originally for heart conditions, effective for migraine prevention.
                        * **Antidepressants (e.g., amitriptyline):** Tricyclic antidepressants can help.
                        * **Anti-seizure drugs (e.g., topiramate, valproate):** Also effective for migraine prevention.
                        * **CGRP Monoclonal Antibodies (e.g., erenumab, fremanezumab):** Newer injectable preventive treatments.
                        * **Botox Injections:** For chronic migraine (15 or more headache days per month).
                    * *Medication selection depends on frequency, severity, and other health conditions. Work with your doctor to find the most effective combination.*
                2.  **Lifestyle Modifications:**
                    * **Identify and Avoid Triggers:** Common triggers include certain foods (aged cheese, processed meats, caffeine withdrawal), stress, sleep deprivation, hormonal changes, bright lights, loud noises, and strong smells. Keep a migraine diary.
                    * **Regular Sleep Schedule:** Go to bed and wake up at consistent times.
                    * **Regular Meals:** Don't skip meals.
                    * **Hydration:** Drink plenty of water throughout the day.
                    * **Stress Management:** Practice relaxation techniques like yoga, meditation, deep breathing, or biofeedback.
                3.  **Follow-up Testing and Monitoring:**
                    * **Migraine Diary:** Crucial for tracking triggers, symptoms, and medication effectiveness.
                    * **Regular Doctor Visits:** To review medication effectiveness, adjust treatment, and discuss new strategies.
                    * **Neurological Evaluation:** May be performed to rule out other conditions.
                4.  **Dietary Recommendations:**
                    * While not a universal solution, identifying and avoiding food triggers can be helpful for some.
                    * Maintain a balanced diet.
                5.  **Physical Activity Guidelines:**
                    * Regular, moderate exercise can help reduce migraine frequency and severity.
                    * Avoid intense exercise during an impending or active migraine attack.
                    * Warm up and cool down properly.
                6.  **Mental Health Considerations:**
                    * Migraines can significantly impact quality of life and lead to anxiety or depression.
                    * Seek counseling or therapy to cope with chronic pain and stress.
                    * Support groups can provide a sense of community and shared experience.
                *Disclaimer: This plan is for informational purposes only and is not a substitute for professional medical advice. Always consult a qualified healthcare provider for diagnosis and treatment of migraine.*
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
        answer = model.generate_text(prompt=query)
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
            st.rerun() # Rerun to clear input and update chat history

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
