"""
Fake News Detection Model Training Script
Generates synthetic fake news dataset and trains a machine learning model
using TF-IDF vectorizer and Logistic Regression classifier.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def generate_synthetic_data():
    """
    Generate synthetic fake and real news articles for training.
    Returns a DataFrame with 'text' (article) and 'label' (0=fake, 1=real) columns.
    """
    
    # Sample real news articles (reliable sources, factual)
    real_news = [
        "Scientists discover new species of deep-sea fish in the Mariana Trench. Researchers have identified over 50 previously unknown species during the expedition.",
        "Government announces new infrastructure investment plan worth $2 billion. The project aims to improve roads, bridges, and public transportation across the country.",
        "World Health Organization releases updated guidelines on preventive health measures. The guidelines emphasize the importance of regular exercise and healthy diet.",
        "New study shows increased use of renewable energy reduces carbon emissions by 30%. Researchers analyzed data from 50 countries over the past decade.",
        "Tech company releases latest smartphone with improved camera technology. The device features a 108-megapixel camera and advanced AI processing.",
        "Stock market reaches record high amid positive economic indicators. Analysts attribute growth to strong corporate earnings and consumer spending.",
        "University researchers develop new cancer treatment showing 85% success rate. Clinical trials involved over 1000 patients across multiple institutions.",
        "International climate summit produces agreement on emission reduction targets. Countries commit to net-zero emissions by 2050.",
        "New railway line opens connecting major cities, reducing travel time by 40%. The project took 5 years and cost $5 billion to complete.",
        "Education department launches literacy program in rural areas. The initiative aims to improve reading skills among 100,000 children.",
        "NASA discovers water ice on the Moon. Scientists collected samples during the latest lunar mission.",
        "Central bank announces interest rate increase to combat inflation. Economic experts debate the impact on borrowing costs.",
        "Archaeological team uncovers ancient Roman artifacts in Italy. The discovery provides insights into daily life 2000 years ago.",
        "Medical breakthrough: new vaccine shows 95% effectiveness against disease. The vaccine has been approved for emergency use.",
        "Amazon rainforest reforestation project plants 10 million trees. Environmental organizations praise the conservation effort.",
    ]
    
    # Sample fake news articles (misleading, unverified claims)
    fake_news = [
        "SHOCKING: Local doctor reveals secret that big pharma doesn't want you to know! This one trick cures all diseases instantly!",
        "Celebrity secretly admits they are not human and have been an alien the whole time. World leaders allegedly knew about it.",
        "Breaking: Government creates mind control device hidden in water supply. Scientists confirm but are being silenced by authorities.",
        "Miracle cure found in remote jungle destroys $1 trillion pharmaceutical industry. Big Pharma desperately trying to suppress the truth.",
        "Famous actor reveals he never actually died and faked his death 20 years ago. He's been living in secret location this whole time.",
        "New study proves that eating only chocolate is healthier than everything else. Nutritionists are lying to you about this.",
        "Top secret government documents leaked: UFOs land in Washington every Tuesday night. The military is hiding aliens in underground bases.",
        "Shocking truth: The moon is actually made of cheese and NASA has been lying to us. Astronauts confirmed this in secret testimony.",
        "Billionaire tech CEO admits that smartphones are poisoning your brain. He secretly uses a flip phone from 1995.",
        "Breaking news: Celebrities don't age because they use secret reptilian blood treatment. Conspiracy experts have all the proof.",
        "Scientists discover that vaccines cause instant superpowers but governments want to keep it secret from the public.",
        "Local woman claims she can communicate with animals and solve world peace. Experts shocked by her incredible abilities.",
        "Ancient pyramid discovered that proves dinosaurs built human civilization. Mainstream scientists refuse to acknowledge this.",
        "Millionaire reveals his secret formula that lets you make $10,000 per day with zero effort. Banks hate him for this one trick.",
        "Breaking: Time travelers from future warn us about upcoming disaster. They've given world leaders secret blueprint to save humanity.",
    ]
    
    # Create labels: 0 for fake, 1 for real
    data = []
    for article in fake_news:
        data.append({'text': article, 'label': 0})
    for article in real_news:
        data.append({'text': article, 'label': 1})
    
    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def train_model():
    """
    Train the fake news detection model and save it along with the vectorizer.
    """
    
    print("=" * 60)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Generate synthetic training data
    print("\n[1/4] Generating synthetic training data...")
    df = generate_synthetic_data()
    print(f"  ✓ Generated {len(df)} articles ({sum(df['label']==0)} fake, {sum(df['label']==1)} real)")
    
    # Step 2: Create TF-IDF vectorizer and transform text
    print("\n[2/4] Creating TF-IDF vectorizer and transforming text...")
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Use top 5000 features
        stop_words='english',  # Remove common English words
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=1,  # Minimum document frequency
        max_df=0.8  # Maximum document frequency (remove very common terms)
    )
    
    # Transform text to numerical features
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    print(f"  ✓ Vectorizer created with {X.shape[1]} features")
    
    # Step 3: Split data and train the model
    print("\n[3/4] Training Logistic Regression classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the model
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    print(f"  ✓ Model training complete")
    
    # Step 4: Evaluate the model
    print("\n[4/4] Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  ✓ Accuracy:  {accuracy:.2%}")
    print(f"  ✓ Precision: {precision:.2%}")
    print(f"  ✓ Recall:    {recall:.2%}")
    print(f"  ✓ F1-Score:  {f1:.2%}")
    
    # Save model and vectorizer
    print("\n[5/5] Saving model and vectorizer...")
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("  ✓ Model saved to model.pkl")
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("  ✓ Vectorizer saved to vectorizer.pkl")
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print("\nYou can now run the Flask app with: python app.py")


if __name__ == "__main__":
    train_model()
