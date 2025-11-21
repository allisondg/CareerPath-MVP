import streamlit as st
import joblib
import time
import pandas as pd

# ==========================================
# 1. LOAD THE BRAIN
# ==========================================
@st.cache_resource
def load_model():
    try:
        return joblib.load('career_model.pkl')
    except FileNotFoundError:
        st.error("File not found: career_model.pkl")
        return None

model = load_model()

# ==========================================
# 2. SETUP THE INTERFACE
# ==========================================
st.set_page_config(page_title="CareerPath AI", page_icon="🎓", layout="centered")

st.title("🎓 CareerPath AI")
st.markdown("### Intelligent Career Guidance System")
st.write("Enter your details below to get AI-powered career recommendations.")
st.markdown("---")

# ==========================================
# 3. CREATE THE CHAT INPUTS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name", placeholder="e.g., John Doe")
    degree = st.text_input("Degree / Major", placeholder="e.g., BS Computer Science")

with col2:
    skills = st.text_input("Top Skills", placeholder="e.g., Python, Java, Communication")
    interests = st.text_input("Interests", placeholder="e.g., AI, Web Development")

# ==========================================
# 4. THE "ANALYZE" BUTTON
# ==========================================
if st.button("Analyze Profile", type="primary"):
    
    if not degree or not skills:
        st.warning("⚠️ Please fill in at least your Degree and Skills to get a result.")
    
    else:
        # 1. Show a loading spinner
        with st.spinner('Analyzing career paths...'):
            time.sleep(1)  # Small delay for effect
            
            # 2. Prepare Input
            user_profile = f"{degree} {skills} {interests}"
            
            # 3. Get Probabilities (The Math)
            # This returns an array of probabilities for every job the AI knows
            probs = model.predict_proba([user_profile])[0]
            classes = model.classes_
            
            # 4. Create a Dataframe of Results
            # We combine the Job Name (classes) with the Score (probs)
            df_results = pd.DataFrame({
                'Career Option': classes,
                'Confidence': probs
            })
            
            # 5. Sort from Highest to Lowest
            df_results = df_results.sort_values(by='Confidence', ascending=False)
            
            # 6. Get the Top Result
            top_match = df_results.iloc[0]
            
            # --- DISPLAY SECTION ---
            
            st.success("Analysis Complete!")
            
            # A. Show the top recommendation
            st.markdown(f"## 🎯 Top Recommendation: **{top_match['Career Option']}**")
            st.progress(float(top_match['Confidence'])) # Show a progress bar
            st.caption(f"Confidence Score: {top_match['Confidence']*100:.2f}%")
            
            st.markdown("---")
            
            # B. Show other potential matches
            st.subheader("🔍 Other Potential Matches")
            st.write("Based on your profile, you might also be suited for:")
            
            # We take the next 4 results (Rows 1 to 5)
            top_5 = df_results.iloc[1:6].copy()
            
            # Format the confidence as a percentage string for display
            top_5['Confidence'] = top_5['Confidence'].apply(lambda x: f"{x*100:.1f}%")
            
            # Reset index so it looks like a list
            top_5 = top_5.reset_index(drop=True)
            top_5.index += 1 
            
            # Display as a clean table
            st.table(top_5)
            
            # C. Visualization
            with st.expander("📊 View Visualization"):
                # Take top 10 for the chart
                chart_data = df_results.head(10).set_index('Career Option')
                st.bar_chart(chart_data)