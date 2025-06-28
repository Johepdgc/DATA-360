import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import json
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
import logging
import sys

# Load environment variables
load_dotenv()

# Initialize NLP components
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
try:
    nlp = spacy.load('es_core_news_sm')
except:
    os.system('python -m spacy download es_core_news_sm')
    nlp = spacy.load('es_core_news_sm')

# Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
HEADERS = {
    'apikey': SUPABASE_KEY,
    'Authorization': f'Bearer {SUPABASE_KEY}'
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'sentiment_analysis.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('sentiment_analysis')

# Helper functions
def preprocess_text(text):
    """Clean and tokenize text for analysis"""
    if not isinstance(text, str) or not text:
        return ""
    
    # Lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
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

def analyze_sentiments():
    """Analyze complaint sentiments and categorize them"""
    # Fetch data
    logger.info("Fetching data from Supabase...")
    data = fetch_data_from_supabase()
    
    if not data:
        logger.warning("No data available for analysis")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Preprocess comments
    logger.info("Preprocessing complaint text...")
    df['processed_text'] = df['Comentarios'].apply(preprocess_text)
    
    # Create TF-IDF vectors
    logger.info("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df['processed_text'])
    
    # Determine optimal number of clusters
    logger.info("Finding optimal number of clusters...")
    inertias = []
    range_n_clusters = range(5, 20)
    for n_clusters in range_n_clusters:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        inertias.append(model.inertia_)
    
    # Find elbow point
    deltas = np.diff(inertias)
    elbow_point = list(range_n_clusters)[np.argmin(deltas) + 1]
    logger.info(f"Optimal number of clusters: {elbow_point}")
    
    # Apply K-means clustering
    logger.info(f"Clustering data into {elbow_point} categories...")
    kmeans = KMeans(n_clusters=elbow_point, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Get top terms for each cluster
    logger.info("Extracting top terms for each cluster...")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for i in range(kmeans.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        cluster_keywords[i] = top_terms
    
    # Determine cluster names based on keywords and complaint motives
    logger.info("Naming clusters based on content...")
    cluster_names = {}
    for cluster_id, keywords in cluster_keywords.items():
        # Get the most common "Motivo de su solicitud" for this cluster
        cluster_complaints = df[df['cluster'] == cluster_id]
        top_motives = cluster_complaints['Motivo de su solicitud'].value_counts().head(3)
        
        # Create a cluster name from the most common motive and top keywords
        if not top_motives.empty:
            main_motive = top_motives.index[0]
            cluster_names[cluster_id] = f"{main_motive} ({', '.join(keywords[:3])})"
        else:
            cluster_names[cluster_id] = f"Cluster {cluster_id} ({', '.join(keywords[:5])})"
    
    # Apply cluster names
    df['category'] = df['cluster'].map(cluster_names)
    
    # Simple sentiment analysis
    logger.info("Performing sentiment analysis...")
    def analyze_sentiment(text):
        if not isinstance(text, str) or not text:
            return 'neutral'
            
        doc = nlp(text)
        
        # Simple rule-based approach
        positive_words = {'gracias', 'bueno', 'excelente', 'genial', 'satisfecho'}
        negative_words = {'problema', 'error', 'falla', 'queja', 'mal', 'pésimo', 'terrible'}
        
        pos_count = sum(1 for token in doc if token.text.lower() in positive_words)
        neg_count = sum(1 for token in doc if token.text.lower() in negative_words)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
    
    df['sentiment'] = df['Comentarios'].apply(analyze_sentiment)
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    category_stats = df.groupby('category').agg({
        'sentiment': lambda x: pd.Series(x).value_counts().to_dict(),
        'Comentarios': 'count'
    }).reset_index()
    
    category_stats['total'] = category_stats['Comentarios']
    category_stats = category_stats.drop(columns=['Comentarios'])
    
    # Save results to JSON
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full clustered data
    df_output = df[['Motivo de su solicitud', 'Comentarios', 'Fecha de interacción', 'category', 'sentiment']]
    df_output.to_json(os.path.join(output_dir, f'clustered_complaints_{timestamp}.json'), orient='records')
    
    # Summary statistics
    with open(os.path.join(output_dir, f'category_stats_{timestamp}.json'), 'w') as f:
        json.dump(category_stats.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    
    # Cluster descriptions
    cluster_info = {
        'clusters': {str(k): {
            'name': v,
            'keywords': cluster_keywords[k]
        } for k, v in cluster_names.items()},
        'timestamp': timestamp,
        'total_records': len(df)
    }
    
    with open(os.path.join(output_dir, f'cluster_info_{timestamp}.json'), 'w') as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return {
        'status': 'success',
        'clusters': len(cluster_names),
        'records': len(df),
        'output_files': [
            f'clustered_complaints_{timestamp}.json',
            f'category_stats_{timestamp}.json',
            f'cluster_info_{timestamp}.json'
        ]
    }

if __name__ == "__main__":
    analyze_sentiments()