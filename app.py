import streamlit as st
import joblib
import time

# ==========================================
# 1. LOAD THE BRAIN
# ==========================================
# We use @st.cache_resource so it only loads once (makes the app faster)
@st.cache_resource
def load_model():
    try:
        return joblib.load('career_model.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# ==========================================
# 2. SETUP THE INTERFACE
# ==========================================
st.set_page_config(page_title="CareerPath AI", page_icon="🎓")

st.title("🎓 CareerPath AI")
st.write("Tell me about your background, and I'll recommend a career path.")
st.markdown("---")

# ==========================================
# 3. CREATE THE CHAT INPUTS
# ==========================================
# Since we are an MVP, we use explicit fields instead of one complex text bar.
# This ensures the AI gets exactly the data it needs.

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("What is your name?", placeholder="Student Name")
    degree = st.text_input("What is your Degree/Major?", placeholder="e.g. BS Mathematics")

with col2:
    skills = st.text_input("What are your Top Skills?", placeholder="e.g. Python, Communication, Excel")
    interests = st.text_input("What are your Interests?", placeholder="e.g. Data analysis, Helping people")

# ==========================================
# 4. THE "GENERATE" BUTTON
# ==========================================
if st.button("Analyze My Profile"):
    
    # Check if the model is loaded
    if model is None:
        st.error("🚨 Error: 'career_model.pkl' not found. Did you run career_brain.py first?")
    
    # Check if fields are filled
    elif not degree or not skills:
        st.warning("⚠️ Please enter at least your Degree and Skills.")
        
    else:
        # Simulate "Thinking" time for a realistic chat feel
        with st.spinner('Analyzing your profile against market trends...'):
            time.sleep(1.5) 
        
        # Prepare the input for the AI
        user_profile = f"{degree} {skills} {interests}"
        
        # Predict
        prediction = model.predict([user_profile])[0]
        probs = model.predict_proba([user_profile])
        confidence = max(probs[0]) * 100

        # Display Result
        st.success("I have found a match!")
        
        st.subheader(f"🎯 Recommended Career: **{prediction}**")
        st.write(f"Confidence Score: **{confidence:.1f}%**")
        
        st.info(f"Based on your skills in *{skills}* and background in *{degree}*, this role aligns well with your profile.")