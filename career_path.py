import pandas as pd
import joblib
import re # Regex for text cleaning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# ==========================================
# STEP 1: LOAD & CLEAN REAL DATA
# ==========================================
print("1. Loading data...")

try:
    df = pd.read_csv('career_data.csv')

    # --- COLUMNS SETUP ---
    col_degree = df.columns[2]    
    col_major = df.columns[3]     
    col_interests = df.columns[4] 
    col_skills = df.columns[5]    
    col_target = df.columns[10]   

    # --- DATA CLEANING ---
    
    # 1. Standardize Target
    df[col_target] = df[col_target].astype(str).str.strip().str.title()

    # 2. CONSOLIDATE JOBS
    job_mapping = {
        'Software Developer': 'Software Engineer',
        'Java Developer': 'Software Engineer',
        'Web Developer': 'Software Engineer',
        'Android Developer': 'Software Engineer',
        'Python Developer': 'Software Engineer',
        'Programmer': 'Software Engineer',
        'Computer Software Engineer': 'Software Engineer',
        'Information Technology': 'IT Specialist',
        'System Administrator': 'IT Specialist',
        'Network Engineer': 'IT Specialist',
        'Data Analyst': 'Data Scientist',
        'Business Analyst': 'Data Scientist',
        'Graphic Designer': 'Designer',
        'Ui/Ux Designer': 'Designer',
        'Content Writer': 'Writer',
        'Manager': 'Business Manager',
        'Human Resources': 'HR Specialist',
        'Primary School Teacher': 'Teacher',
        'Secondary School Teacher': 'Teacher',
        'Tutor': 'Teacher',
        'Lecturer': 'Teacher',
        'Professor': 'Teacher',
        'Educator': 'Teacher',
        'Instructor': 'Teacher'
    }
    df[col_target] = df[col_target].replace(job_mapping)

    # --- BALANCING THE DATASET ---
    # 1. Check which jobs have way too many rows
    job_counts = df[col_target].value_counts()
    
    # 2. Define a limit. No job should have more than 150 samples.
    MAX_SAMPLES_PER_JOB = 150
    
    balanced_df_list = []
    
    for job_title in df[col_target].unique():
        job_rows = df[df[col_target] == job_title]
        
        # If a job has too many rows, sample only 150 of them
        if len(job_rows) > MAX_SAMPLES_PER_JOB:
            job_rows = job_rows.sample(n=MAX_SAMPLES_PER_JOB, random_state=42)
            
        balanced_df_list.append(job_rows)
        
    # Rebuild the dataframe with the balanced data
    df = pd.concat(balanced_df_list)
    
    print(f"   ⚖️  Dataset Balanced. 'Bully' classes capped at {MAX_SAMPLES_PER_JOB} rows.")

    # 3. FILTER JUNK
    bad_patterns = 'student|unemployed|not working|na|0|nan'
    df = df[~df[col_target].str.lower().str.contains(bad_patterns, na=False)]
    
    # 4. FREQUENCY FILTER (Minimum 5 occurrences to be learned)
    job_counts = df[col_target].value_counts()
    valid_jobs = job_counts[job_counts >= 5].index
    df = df[df[col_target].isin(valid_jobs)]

    # 5. CLEAN DEGREES
    # Remove "B.Tech", "B.E", "BS" so some degrees match better
    def clean_degree(text):
        text = str(text).lower()
        text = re.sub(r'b\.?tech|b\.?e|b\.?s\.?|bachelor of|honors', '', text)
        return text.strip()

    df[col_degree] = df[col_degree].apply(clean_degree)

    # 6. FILL & COMBINE
    df = df.fillna('')
    df['combined_text'] = (
        df[col_degree] + " " + 
        df[col_major].astype(str) + " " + 
        df[col_skills].astype(str) + " " + 
        df[col_interests].astype(str)
    )
    
    df['target_career'] = df[col_target]

    print(f"   Data Cleaned. Training on {len(df)} profiles.")

except Exception as e:
    print(f"ERROR: {e}")
    exit()

# ==========================================
# STEP 2: BUILD & TRAIN (OPTIMIZED)
# ==========================================
print("2. Training the OPTIMIZED model...")

model_pipeline = make_pipeline(
    # IMPROVEMENT 1: N-Grams
    # Read words in groups of 1, 2, and 3 (e.g. "Computer Science" is now one token)
    TfidfVectorizer(
        stop_words='english', 
        ngram_range=(1, 3), 
        max_features=5000
    ),
    
    # IMPROVEMENT 2: Class Weighting
    # 'balanced' tells the AI to pay more attention to rare jobs
    RandomForestClassifier(
        n_estimators=200,      # More trees = smoother voting
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
)

model_pipeline.fit(df['combined_text'], df['target_career'])
print("   Model Trained Successfully!")

# ==========================================
# STEP 3: SAVE THE BRAIN
# ==========================================
joblib.dump(model_pipeline, 'career_model.pkl', compress=3)
print("3. Brain saved.")

# ==========================================
# STEP 4: TEST
# ==========================================
print("-" * 30)
print("🧠 SYSTEM READY. PLEASE TYPE YOUR INFO.")
print("-" * 30)

# Reverting back to input() functions
test_degree = input("Enter a Degree (e.g., BS Computer Science): ")
test_skills = input("Enter Skills (e.g., Python, SQL): ")
test_interests = input("Enter Interests (e.g., AI, Data): ")

# Combine inputs to match the training format
test_profile = f"{test_degree} {test_skills} {test_interests}"

print("\n... Analyzing ...\n")

# Predict
prediction = model_pipeline.predict([test_profile])
probs = model_pipeline.predict_proba([test_profile])
confidence = max(probs[0]) * 100

print(f"✅ Recommendation: {prediction[0]}")
print(f"📊 Confidence: {confidence:.2f}%")