# Install necessary libraries if you haven't already:
# pip install pandas bertopic sentence-transformers "scikit-learn<1.4.0" umap-learn hdbscan matplotlib openpyxl
# NOTE: BERTopic can have specific dependencies. scikit-learn version < 1.4.0 is often needed.

import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import os
import re

# --- Configuration ---
ORIGINAL_CSV_PATH = 'Tracking de solicitudes (Responses) - Form Responses 1.csv'

OUTPUT_REPORT_PATH = 'manual_exploration_report.txt'
OUTPUT_ENHANCED_CSV_PATH = 'unsupervised_topic_analysis.csv'
OUTPUT_PLOTS_DIR = 'analysis_plots_unsupervised/'

# Ensure output directory exists
if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)

# Columns from your CSV
TEXT_COLUMN = 'Comentarios'
DATE_COLUMN = 'Marca temporal'

# Time Filtering: Last 6 full months from today's date
# Let's assume today is June 6, 2025 for consistent results.
# The 6 full months prior are Dec 2024, Jan 2025, Feb 2025, Mar 2025, Apr 2025, May 2025.
TODAY = datetime(2025, 6, 6)
END_DATE = (TODAY.replace(day=1) - timedelta(days=1)) # End of last month (May 31, 2025)
START_DATE = (END_DATE.replace(day=1) - relativedelta(months=5)) # Start of the 6-month window (Dec 1, 2024)

# BERTopic Configuration
# Using a multilingual model is safe for Spanish text
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' 
TOP_N_TOPICS_TO_EXPLORE = 5
NUM_EXAMPLE_COMMENTS = 5 # Number of example comments to show per topic

# --- Helper Functions ---
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    # BERTopic handles a lot of preprocessing, so we keep it simple here.
    return text

def plot_topic_trends(df_trends, output_dir):
    """Plots the monthly trends for the topics provided in the dataframe."""
    plt.figure(figsize=(14, 8))
    
    for topic_label in df_trends['Topic_Label'].unique():
        topic_data = df_trends[df_trends['Topic_Label'] == topic_label]
        plt.plot(topic_data['YearMonth'], topic_data['Count'], marker='o', linestyle='-', label=topic_label)

    plt.title('Monthly Trends of Top 5 Complaint Topics')
    plt.xlabel('Month')
    plt.ylabel('Number of Complaints')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    
    plot_filename = os.path.join(output_dir, 'top_5_topic_trends.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Top 5 topic trends plot saved to {plot_filename}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Unsupervised Complaint Analysis for Manual Exploration...")

    # 1. Load and Filter Data
    try:
        df_orig = pd.read_csv(ORIGINAL_CSV_PATH)
        print(f"Successfully loaded {ORIGINAL_CSV_PATH}. Shape: {df_orig.shape}")
    except FileNotFoundError:
        print(f"Error: Original CSV file not found at {ORIGINAL_CSV_PATH}")
        exit()
    
    df = df_orig.copy()
    # Prepare text data
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna('')
    # Keep original for exploration report
    df['Original_Comment'] = df[TEXT_COLUMN] 
    df['Processed_Comment'] = df['Original_Comment'].apply(preprocess_text)
    
    # Filter out empty comments as they can't be processed
    df = df[df['Processed_Comment'] != ''].copy()

    # Parse and filter dates
    try:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
        df = df.dropna(subset=[DATE_COLUMN])
        
        df_filtered = df[(df[DATE_COLUMN] >= START_DATE) & (df[DATE_COLUMN] <= END_DATE)].copy()

        if df_filtered.empty:
            print(f"No data found for the period {START_DATE.date()} to {END_DATE.date()}. Exiting.")
            exit()
        
        print(f"Filtered data for the last 6 full months ({START_DATE.date()} to {END_DATE.date()}). Shape: {df_filtered.shape}")
        df_filtered['YearMonth'] = df_filtered[DATE_COLUMN].dt.to_period('M').astype(str)
    except Exception as e:
        print(f"Error during date parsing or filtering: {e}")
        exit()

    # 2. Unsupervised Topic Modeling with BERTopic
    print("\nStarting unsupervised topic modeling with BERTopic (this may take time)...")
    
    # Prepare documents for BERTopic
    docs = df_filtered['Processed_Comment'].tolist()
    timestamps = df_filtered[DATE_COLUMN].tolist() # For potential time-aware modeling

    # Load sentence transformer model for embeddings
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Initialize BERTopic
    # We can pass `language="multilingual"` if we suspect mixed languages.
    # min_topic_size can be adjusted to get more or fewer topics.
    topic_model = BERTopic(embedding_model=embedding_model, 
                           language="multilingual", 
                           min_topic_size=15, # Minimum size of a topic
                           verbose=True)

    # Fit the model and transform documents to topics
    topics, probs = topic_model.fit_transform(docs)

    # Add results back to the DataFrame
    df_filtered['Topic_ID'] = topics
    print("\nTopic modeling complete.")

    # Get information about the discovered topics
    # Topic -1 is for outliers and will be ignored in the trend analysis
    topic_info = topic_model.get_topic_info()
    print("Discovered topics (Top 10):")
    print(topic_info.head(11))

    # Add Topic Keywords to the main DataFrame for easy reference
    # Create a mapping from Topic_ID to a representative name/keywords
    topic_info['Topic_Label'] = topic_info['Name'].apply(lambda x: "_".join(x.split('_')[1:4])) # Shorten name
    topic_map = topic_info.set_index('Topic')['Topic_Label'].to_dict()
    df_filtered['Topic_Keywords'] = df_filtered['Topic_ID'].map(topic_map)
    
    # 3. Analyze Trends and Identify Top Topics
    # Exclude outliers (Topic -1)
    df_analysis = df_filtered[df_filtered['Topic_ID'] != -1].copy()
    
    if df_analysis.empty:
        print("No topics other than outliers were found. Cannot proceed with trend analysis.")
        exit()

    # Get overall top N topics
    top_n_topics_ids = df_analysis['Topic_ID'].value_counts().nlargest(TOP_N_TOPICS_TO_EXPLORE).index
    
    # Get monthly counts for these top topics
    monthly_trends = df_analysis[df_analysis['Topic_ID'].isin(top_n_topics_ids)] \
                      .groupby(['YearMonth', 'Topic_ID', 'Topic_Keywords']) \
                      .size().reset_index(name='Count') \
                      .sort_values(by=['YearMonth', 'Count'], ascending=[True, False])
    
    # 4. Generate Manual Exploration Report
    print("\nGenerating manual exploration report...")
    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("--- Manual Exploration Report for Top Complaint Topics ---\n")
        f.write(f"Analysis Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}\n")
        f.write("="*60 + "\n\n")

        for topic_id in top_n_topics_ids:
            topic_data = df_analysis[df_analysis['Topic_ID'] == topic_id]
            topic_keywords = topic_data['Topic_Keywords'].iloc[0]
            total_count = len(topic_data)

            f.write(f"--- Topic #{topic_id}: {topic_keywords} ---\n")
            f.write(f"Total Complaints in Period: {total_count}\n\n")

            # Write monthly trend for this topic
            f.write("Monthly Trend:\n")
            topic_trend_data = monthly_trends[monthly_trends['Topic_ID'] == topic_id]
            for _, row in topic_trend_data.iterrows():
                f.write(f"  - {row['YearMonth']}: {row['Count']} complaints\n")
            
            f.write("\n")

            # Write example comments for manual verification
            f.write(f"Example Comments (up to {NUM_EXAMPLE_COMMENTS}):\n")
            example_comments = topic_data['Original_Comment'].head(NUM_EXAMPLE_COMMENTS).tolist()
            for i, comment in enumerate(example_comments):
                f.write(f"  {i+1}. \"{comment.strip()}\"\n")
            
            f.write("\n" + "="*60 + "\n\n")
    
    print(f"Manual exploration report saved to {OUTPUT_REPORT_PATH}")

    # 5. Save enhanced CSV and plot trends
    # Prepare data for plotting
    plot_data = monthly_trends.rename(columns={'Topic_Keywords': 'Topic_Label'})
    plot_topic_trends(plot_data, OUTPUT_PLOTS_DIR)

    # Save the dataframe with topic assignments
    try:
        output_cols = ['Original_Comment', DATE_COLUMN, 'YearMonth', 'Topic_ID', 'Topic_Keywords']
        df_filtered_to_save = df_filtered[output_cols]
        df_filtered_to_save.to_csv(OUTPUT_ENHANCED_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"Enhanced data with unsupervised topic assignments saved to {OUTPUT_ENHANCED_CSV_PATH}")
    except Exception as e:
        print(f"Error saving enhanced CSV: {e}")
        
    print("\n--- Unsupervised Analysis Finished ---")
    print("Next Steps:")
    print(f"1. Open and review '{OUTPUT_REPORT_PATH}' to manually explore the top 5 complaint topics.")
    print("2. Use the report to verify if the discovered topics are coherent and meaningful.")
    print(f"3. For deeper data dives, use '{OUTPUT_ENHANCED_CSV_PATH}' in Excel or other BI tools.")
    print(f"4. View the trend visualization at '{OUTPUT_PLOTS_DIR}top_5_topic_trends.png'.")

