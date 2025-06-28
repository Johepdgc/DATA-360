import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Load environment variables
load_dotenv()

# Initialize NLP components
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}'
}

# Output directories
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Configure better logging
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'topic_analysis.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('topic_analysis')

# Helper functions
def preprocess_text(text):
    """Clean and normalize text for better topic modeling"""
    if not isinstance(text, str) or not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keeping Spanish accents
    text = re.sub(r'[^\w\s\á\é\í\ó\ú\ü\ñ]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('spanish'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

# Enhanced error handling for Supabase connection
def fetch_data_from_supabase():
    """Fetch complaint data from Supabase with improved error handling"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        error_msg = "Supabase credentials missing. Check your .env file."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        logger.info(f"Fetching data from Supabase: {SUPABASE_URL}")
        url = f"{SUPABASE_URL}/rest/v1/cx_quejas?select=*"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Successfully fetched {len(data)} records from Supabase")
            return data
        else:
            error_msg = f"Supabase API error: {response.status_code}, {response.text}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    except Exception as e:
        error_msg = f"Failed to fetch data from Supabase: {str(e)}"
        logger.error(error_msg)
        raise

def analyze_topics():
    """Analyze complaint topics using BERTopic"""
    # Fetch data
    logger.info("Fetching data from Supabase...")
    data = fetch_data_from_supabase()
    
    if not data:
        logger.warning("No data available for analysis")
        return {'status': 'error', 'message': 'No data available'}
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Clean data
    logger.info("Preprocessing complaint text...")
    df['processed_text'] = df['Comentarios'].apply(preprocess_text)
    df = df[df['processed_text'] != ""]  # Remove empty comments
    
    if df.empty:
        logger.warning("No valid data after preprocessing")
        return {'status': 'error', 'message': 'No valid data after preprocessing'}
    
    # Create custom vectorizer with Spanish stopwords
    spanish_stopwords = stopwords.words('spanish')
    vectorizer = CountVectorizer(stop_words=spanish_stopwords)
    
    # Use Spanish multilingual model for embeddings
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    
    # Create BERTopic model
    logger.info("Creating BERTopic model...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        language="spanish",
        min_topic_size=5,  # Minimum number of documents per topic
        verbose=True
    )
    
    # Fit model to data
    logger.info("Fitting BERTopic model to complaint data...")
    docs = df['processed_text'].tolist()
    topics, probs = topic_model.fit_transform(docs)
    
    # Add topic information to dataframe
    df['topic_id'] = topics
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    
    # Get representative docs per topic
    logger.info("Extracting representative complaints for each topic...")
    topic_docs = {}
    topic_sentiments = {}
    
    for topic in topic_info['Topic'].values:
        if topic != -1:  # Skip outlier topic
            # Get top documents for this topic
            documents = topic_model.get_representative_docs(topic, top_n=5)
            topic_docs[topic] = documents
            
            # Calculate sentiment distribution for this topic
            topic_df = df[df['topic_id'] == topic]
            sentiments = analyze_sentiments(topic_df['Comentarios'].tolist())
            sentiment_counts = {
                'positive': sum(1 for s in sentiments if s == 'positive'),
                'neutral': sum(1 for s in sentiments if s == 'neutral'),
                'negative': sum(1 for s in sentiments if s == 'negative')
            }
            total = len(sentiments)
            sentiment_pcts = {k: round(v / total * 100) for k, v in sentiment_counts.items()}
            topic_sentiments[topic] = sentiment_pcts
    
    # Plot topic visualization
    logger.info("Generating visualization...")
    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(PLOTS_DIR, 'topic_visualization.html'))
    
    # Plot topic hierarchy
    try:
        hierarchy_fig = topic_model.visualize_hierarchy()
        hierarchy_fig.write_html(os.path.join(PLOTS_DIR, 'topic_hierarchy.html'))
    except Exception as e:
        logger.warning(f"Could not generate hierarchy visualization: {e}")
    
    # Plot topic distributions
    try:
        bars_fig = topic_model.visualize_barchart(top_n_topics=10)
        bars_fig.write_html(os.path.join(PLOTS_DIR, 'topic_barchart.html'))
    except Exception as e:
        logger.warning(f"Could not generate barchart visualization: {e}")
    
    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get topics with keywords
    topics_keywords = {}
    for topic_id in topic_info['Topic']:
        if topic_id != -1:  # Skip outlier topic
            keywords = topic_model.get_topic(topic_id)
            topics_keywords[str(topic_id)] = [keyword for keyword, _ in keywords]
    
    # Create results dictionary
    results = {
        'timestamp': timestamp,
        'total_complaints': len(df),
        'topics_count': len(topics_keywords),
        'topics': {
            str(topic_id): {
                'keywords': topics_keywords.get(str(topic_id), []),
                'representative_docs': topic_docs.get(topic_id, []),
                'sentiment': topic_sentiments.get(topic_id, {})
            } for topic_id in topics_keywords.keys()
        }
    }
    
    # Save topic analysis results
    with open(os.path.join(OUTPUT_DIR, f'topic_analysis_{timestamp}.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save full data with topic assignments
    df_output = df[['Motivo de su solicitud', 'Comentarios', 'Fecha de interacción', 'topic_id']]
    df_output.to_json(os.path.join(OUTPUT_DIR, f'complaints_with_topics_{timestamp}.json'), orient='records')
    
    logger.info(f"Topic analysis complete. Results saved to {OUTPUT_DIR}")
    return {
        'status': 'success',
        'topics_count': len(topics_keywords),
        'records': len(df),
        'timestamp': timestamp,
        'output_files': [
            f'topic_analysis_{timestamp}.json',
            f'complaints_with_topics_{timestamp}.json'
        ]
    }

def analyze_sentiments(texts):
    """Perform simple sentiment analysis on texts"""
    sentiments = []
    
    # Define sentiment lexicons
    positive_words = {'gracias', 'bueno', 'excelente', 'genial', 'satisfecho', 'bien', 'perfecto', 'éxito', 'solucionar'}
    negative_words = {'problema', 'error', 'falla', 'queja', 'mal', 'pésimo', 'terrible', 'molesto', 'insatisfecho', 'demora'}
    
    for text in texts:
        if not isinstance(text, str) or not text:
            sentiments.append('neutral')
            continue
            
        text = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            sentiments.append('positive')
        elif neg_count > pos_count:
            sentiments.append('negative')
        else:
            sentiments.append('neutral')
    
    return sentiments

if __name__ == "__main__":
    analyze_topics()