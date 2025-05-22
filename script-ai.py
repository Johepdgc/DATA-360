# Install necessary libraries if you haven't already:
# pip install pandas nltk sentence-transformers scikit-learn

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --- Configuration ---
INPUT_CSV_PATH = 'Copy of Tracking de solicitudes (Responses) - Form Responses 1.csv' # Replace with your CSV file path
OUTPUT_CSV_PATH = 'categorized_complaints.csv'
TEXT_COLUMN = 'Comentarios' # Column containing the natural language comments
MOTIVE_COLUMN = 'Motivo de su solicitud' # Existing categorization column
N_CLUSTERS = 20  # Adjust based on your data and desired granularity. Experimentation is key!
# Consider using the elbow method or silhouette analysis to find an optimal k.

# --- Download NLTK resources (if not already downloaded) ---
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')

# --- Helper Functions ---

def preprocess_text(text, language='spanish'):
    """
    Cleans and preprocesses a single text string.
    - Converts to lowercase
    - Removes punctuation and numbers
    - Removes stop words
    - Tokenizes
    """
    if pd.isna(text) or not isinstance(text, str):
        return "" # Return empty string for NaN or non-string inputs
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text)     # Remove numbers
    
    tokens = word_tokenize(text, language=language)
    
    stop_words_list = stopwords.words(language)
    # Add any custom stop words relevant to your domain
    # custom_stop_words = ['n1co', 'app', 'cliente', 'favor'] 
    # stop_words_list.extend(custom_stop_words)
    
    processed_tokens = [word for word in tokens if word not in stop_words_list and len(word) > 1]
    
    return " ".join(processed_tokens)

def generate_embeddings(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Generates sentence embeddings for a list of texts.
    Uses a multilingual model suitable for Spanish text.
    """
    print(f"Loading sentence transformer model: {model_name}...")
    # This model is good for multilingual text, including Spanish.
    model = SentenceTransformer(model_name)
    print("Model loaded. Generating embeddings (this may take some time)...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print("Embeddings generated.")
    return embeddings

def perform_clustering(embeddings, n_clusters):
    """
    Performs K-Means clustering on the embeddings.
    """
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)
    print("Clustering complete.")
    return cluster_labels

def get_top_keywords_per_cluster(texts, cluster_labels, n_clusters, n_keywords=5, language='spanish'):
    """
    Extracts top TF-IDF keywords for each cluster.
    """
    print("Extracting top keywords for each cluster...")
    df_clusters = pd.DataFrame({'text': texts, 'cluster': cluster_labels})
    cluster_keywords = {}
    
    stop_words_list = stopwords.words(language)

    for i in range(n_clusters):
        cluster_texts = df_clusters[df_clusters['cluster'] == i]['text'].tolist()
        if not cluster_texts: # Handle empty clusters
            cluster_keywords[i] = ["N/A - Empty Cluster"]
            continue
        
        try:
            vectorizer = TfidfVectorizer(max_features=n_keywords * 2, # Get more to choose from
                                         stop_words=stop_words_list,
                                         ngram_range=(1,2)) # Consider unigrams and bigrams
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores for each term across the cluster
            sum_tfidf = tfidf_matrix.sum(axis=0)
            # Get indices of top N keywords
            top_n_indices = np.argsort(sum_tfidf)[0, ::-1][0, :n_keywords].A1 # Squeeze to 1D array
            
            top_keywords = [feature_names[idx] for idx in top_n_indices]
            cluster_keywords[i] = top_keywords if top_keywords else ["N/A - No distinct keywords"]

        except ValueError as e:
            # This can happen if all documents in a cluster are empty after preprocessing
            # or if they only contain stop words.
            print(f"Warning: Could not extract keywords for cluster {i}: {e}")
            cluster_keywords[i] = ["N/A - Keyword extraction error"]
            
    print("Keyword extraction complete.")
    return cluster_keywords

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting complaint categorization process...")

    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully loaded {INPUT_CSV_PATH}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

    # Ensure the text column exists
    if TEXT_COLUMN not in df.columns:
        print(f"Error: Text column '{TEXT_COLUMN}' not found in the CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        exit()

    # Handle missing values in the text column before preprocessing
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('') 

    # 2. Preprocess Text
    print(f"Preprocessing text in column: '{TEXT_COLUMN}'...")
    # Create a new column for processed text to keep the original
    df['processed_comentarios'] = df[TEXT_COLUMN].apply(lambda x: preprocess_text(x, language='spanish'))
    print("Text preprocessing complete.")
    print(f"Sample of processed comments:\n{df[['processed_comentarios', TEXT_COLUMN]].head()}")

    # Filter out rows where processed_comentarios is empty, as they cannot be embedded meaningfully
    # Or, decide on a strategy for them (e.g., assign to a special cluster)
    # For now, we'll proceed with non-empty ones for embedding.
    # If many are empty, it might indicate an issue with preprocessing or original data.
    non_empty_texts_df = df[df['processed_comentarios'].str.strip() != '']
    if non_empty_texts_df.empty:
        print("Error: All comments are empty after preprocessing. Cannot proceed with embedding.")
        exit()
    
    original_indices = non_empty_texts_df.index # Keep track of original indices

    # 3. Generate Embeddings
    # Only generate embeddings for non-empty processed texts
    texts_for_embedding = non_empty_texts_df['processed_comentarios'].tolist()
    embeddings = generate_embeddings(texts_for_embedding)

    # 4. Perform Clustering
    # Ensure N_CLUSTERS is not greater than the number of samples
    actual_n_clusters = min(N_CLUSTERS, len(texts_for_embedding))
    if actual_n_clusters < N_CLUSTERS:
        print(f"Warning: N_CLUSTERS ({N_CLUSTERS}) was greater than the number of non-empty samples ({len(texts_for_embedding)}). Adjusting to {actual_n_clusters}.")
    if actual_n_clusters < 2: # K-Means needs at least 2 clusters (or 1 if only 1 sample)
        print("Error: Not enough unique samples to perform clustering. Need at least 2.")
        # Potentially save the preprocessed data and exit
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"Preprocessed data (without clustering) saved to {OUTPUT_CSV_PATH}")
        exit()
        
    cluster_labels = perform_clustering(embeddings, actual_n_clusters)
    
    # Add cluster labels back to the non_empty_texts_df
    non_empty_texts_df['nlp_cluster'] = cluster_labels
    
    # Merge cluster labels back into the original DataFrame
    df['nlp_cluster'] = np.nan # Initialize with NaN
    df.loc[original_indices, 'nlp_cluster'] = non_empty_texts_df['nlp_cluster']
    # Convert to integer type if no NaNs, otherwise keep as float
    if df['nlp_cluster'].notna().all():
        df['nlp_cluster'] = df['nlp_cluster'].astype(int)


    # 5. Get Top Keywords
    # Use texts_for_embedding and cluster_labels which correspond to each other
    cluster_keywords_map = get_top_keywords_per_cluster(texts_for_embedding, cluster_labels, actual_n_clusters, language='spanish')
    
    # Map keywords to the original DataFrame
    df['nlp_cluster_keywords'] = df['nlp_cluster'].map(cluster_keywords_map).fillna("N/A - Original comment was empty or not clustered")
    # Ensure keywords are stored as strings (e.g., comma-separated)
    df['nlp_cluster_keywords'] = df['nlp_cluster_keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)


    # 6. Save Results
    try:
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig') # utf-8-sig for better Excel compatibility with special chars
        print(f"Categorization complete. Results saved to {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    print("\n--- Process Summary ---")
    print(f"Total rows processed: {len(df)}")
    print(f"Rows with non-empty comments used for clustering: {len(non_empty_texts_df)}")
    print(f"Number of clusters generated: {actual_n_clusters}")
    print(f"Output file: {OUTPUT_CSV_PATH}")
    print("\n--- Next Steps ---")
    print(f"1. Review '{OUTPUT_CSV_PATH}'.")
    print("2. Analyze the 'nlp_cluster' and 'nlp_cluster_keywords' columns.")
    print("3. Manually inspect comments within each cluster to assign meaningful, human-readable category names.")
    print("4. Experiment with 'N_CLUSTERS' and preprocessing steps to refine categories.")
    print("5. Consider using techniques like the elbow method or silhouette score to help determine an optimal number of clusters.")
