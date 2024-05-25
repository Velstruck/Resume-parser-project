from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import chardet

app = Flask(__name__)

# Load the labeled dataset
resume_data = pd.read_csv('resumes.csv')

# Extract features using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resume_data['text'])

# Train logistic regression models for each parameter
model_skills = LogisticRegression()
model_experience = LogisticRegression()
model_projects = LogisticRegression()
model_academics = LogisticRegression()

model_skills.fit(X, resume_data['skills'])
model_experience.fit(X, resume_data['experience'])
model_projects.fit(X, resume_data['projects'])
model_academics.fit(X, resume_data['academics'])

# Define route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define route for parsing resumes
@app.route('/parse_resume', methods=['POST'])
def parse_resume():
    if 'resume' not in request.files:
        return "No file part"
    
    resume = request.files['resume']
    if resume.filename == '':
        return "No selected file"
    
    # Detect file encoding using chardet
    raw_data = resume.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    
    # Try different encodings if the first one fails
    encodings_to_try = [encoding, 'utf-8', 'ISO-8859-1', 'latin1']

    text = None
    for enc in encodings_to_try:
        try:
            text = raw_data.decode(enc)
            break  # Exit the loop if decoding is successful
        except (UnicodeDecodeError, TypeError):
            continue  # Try the next encoding

    if text is None:
        return "Error decoding file. Please upload a valid text file."
    
    # Extract features from the text
    X_new = vectorizer.transform([text])
    
    # Predict the likelihood of each parameter
    skills_prob = model_skills.predict_proba(X_new)[0][1] * 100
    experience_prob = model_experience.predict_proba(X_new)[0][1] * 100
    projects_prob = model_projects.predict_proba(X_new)[0][1] * 100
    academics_prob = model_academics.predict_proba(X_new)[0][1] * 100
    
    # Format the probabilities with a percentage sign
    skills_prob = f"{skills_prob:.2f}%"
    experience_prob = f"{experience_prob:.2f}%"
    projects_prob = f"{projects_prob:.2f}%"
    academics_prob = f"{academics_prob:.2f}%"
    
    # Pass the probabilities to the result template
    return render_template('result.html', skills_prob=skills_prob, 
                           experience_prob=experience_prob, 
                           projects_prob=projects_prob, 
                           academics_prob=academics_prob)

if __name__ == '__main__':
    app.run(debug=True)
