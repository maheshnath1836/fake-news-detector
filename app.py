"""
Fake News Detection Flask Web Application
Provides a user-friendly interface to detect fake news articles.
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Global variables to store model and vectorizer
model = None
vectorizer = None


def load_model_and_vectorizer():
    """
    Load the trained model and vectorizer from pickle files.
    If files don't exist, train the model first.
    Returns True if successful, False otherwise.
    """
    global model, vectorizer
    
    # Check if model and vectorizer files exist
    if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
        print("\n⚠️  Model files not found. Automatically training the model...")
        print("This may take a few moments...\n")
        
        # Import and run the training script
        from train_model import train_model
        train_model()
    
    try:
        # Load the vectorizer
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✓ Vectorizer loaded successfully")
        
        # Load the model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def predict_fake_news(news_text):
    """
    Predict whether the given news text is fake or real.
    
    Args:
        news_text (str): The news article text to analyze
        
    Returns:
        dict: Contains prediction, confidence, and label
    """
    if not news_text or len(news_text.strip()) == 0:
        return {
            'error': 'Please enter news text to check',
            'prediction': None,
            'confidence': None,
            'label': None
        }
    
    try:
        # Vectorize the input text
        X = vectorizer.transform([news_text])
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X)[0]
        
        # Get prediction (0=fake, 1=real)
        prediction = model.predict(X)[0]
        
        # Calculate confidence as percentage
        confidence = max(probabilities) * 100
        
        # Create result dictionary
        result = {
            'error': None,
            'prediction': int(prediction),
            'confidence': round(confidence, 2),
            'label': 'REAL' if prediction == 1 else 'FAKE'
        }
        
        return result
    
    except Exception as e:
        return {
            'error': f'Error processing text: {str(e)}',
            'prediction': None,
            'confidence': None,
            'label': None
        }


@app.route('/')
def index():
    """Render the home page with the news checking interface."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict if news is fake or real.
    Expects JSON POST request with 'news_text' field.
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'news_text' not in data:
            return jsonify({
                'error': 'No news text provided'
            }), 400
        
        news_text = data['news_text']
        
        # Make prediction
        result = predict_fake_news(news_text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint to verify the app is running."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTION WEB APP - LOADING")
    print("=" * 60)
    
    # Load model and vectorizer
    if load_model_and_vectorizer():
        print("\n" + "=" * 60)
        print("✓ APPLICATION READY!")
        print("=" * 60)
        print("\nStarting Flask server...")
        print("Open your browser and navigate to: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Run Flask app
        app.run(debug=True, host='localhost', port=5000)
    else:
        print("\n❌ Failed to load model. Please check the error messages above.")
