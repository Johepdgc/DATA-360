# Install necessary libraries if you haven't already:
# pip install pandas nltk sentence-transformers scikit-learn matplotlib seaborn wordcloud yellowbrick transformers torch openpyxl pysentimiento

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer # For Elbow method
from scipy.stats import chi2_contingency
from pysentimiento import create_analyzer # For sentiment analysis

# --- Configuration ---
INPUT_CSV_PATH = 'categorized_complaints.csv' # Assumes output from previous categorization script
OUTPUT_ENHANCED_CSV_PATH = 'enhanced_complaints_analysis.csv'
OUTPUT_PLOTS_DIR = 'analysis_plots/' # Directory to save plots

# Ensure output directory exists
import os
if not os.path.exists(OUTPUT_PLOTS_DIR):
    os.makedirs(OUTPUT_PLOTS_DIR)

# Columns from your CSV (adjust if names are different)
TEXT_COLUMN = 'Comentarios'
PROCESSED_TEXT_COLUMN = 'processed_comentarios' # Assumed from previous script
MOTIVE_COLUMN = 'Motivo de su solicitud'
NLP_CLUSTER_COLUMN = 'nlp_cluster'
NLP_KEYWORDS_COLUMN = 'nlp_cluster_keywords'
COUNTRY_COLUMN = 'País de donde se contacta'
# Date column for time series - check your CSV for the correct one
# 'Marca temporal' or 'Fecha de interacción'
DATE_COLUMN = 'Marca temporal' # Or 'Fecha de interacción'
# If 'Marca temporal' is like "2/12/2022 12:34:48", pandas should parse it.
# If it's Excel serial date, more complex parsing might be needed.

TOP_N_CATEGORIES_PLOT = 10
TOP_N_CLUSTERS_WORDCLOUD = 5

# --- NLTK Resource Check ---
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')

# --- Helper Functions ---

def plot_elbow_method(embeddings, max_k=15, metric='distortion', timings=False):
    """
    Plots the Elbow Method to help find optimal K for K-Means.
    Requires sentence embeddings as input.
    This function is for guidance; you'd use the chosen K in the main categorization.
    """
    if embeddings is None or len(embeddings) < max_k :
        print("Not enough data points or embeddings not provided for Elbow method.")
        return

    print(f"\nPlotting Elbow Method (up to K={max_k})...")
    model = KMeans(random_state=42, n_init='auto')
    visualizer = KElbowVisualizer(model, k=(2, max_k), metric=metric, timings=timings)
    
    visualizer.fit(embeddings) # Fit the data to the visualizer
    plt.title(f'Elbow Method for Optimal K ({metric})')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel(metric.capitalize() if metric != 'distortion' else 'Distortion Score')
    
    plot_filename = os.path.join(OUTPUT_PLOTS_DIR, f'elbow_method_{metric}.png')
    visualizer.show(outpath=plot_filename, clear_figure=True) # Saves the plot
    print(f"Elbow method plot saved to {plot_filename}")
    print(f"Optimal K suggested by Elbow Visualizer ({metric}): {visualizer.elbow_value_}")


def calculate_silhouette_scores(embeddings, range_n_clusters=[2, 3, 4, 5, 10, 15, 20]):
    """
    Calculates Silhouette Scores for different numbers of clusters.
    Requires sentence embeddings. For guidance.
    """
    if embeddings is None or len(embeddings) < max(range_n_clusters):
        print("Not enough data points or embeddings not provided for Silhouette scores.")
        return {}
        
    print("\nCalculating Silhouette Scores...")
    silhouette_scores = {}
    for n_clusters in range_n_clusters:
        if len(embeddings) <= n_clusters: # Cannot have more clusters than samples
            print(f"Skipping n_clusters={n_clusters} as it's >= number of samples.")
            continue
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores[n_clusters] = score
            print(f"Silhouette Score for {n_clusters} clusters: {score:.4f}")
        except Exception as e:
            print(f"Error calculating silhouette score for {n_clusters} clusters: {e}")
    
    if silhouette_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
        plt.title('Silhouette Scores for Different Numbers of Clusters')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plot_filename = os.path.join(OUTPUT_PLOTS_DIR, 'silhouette_scores.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Silhouette scores plot saved to {plot_filename}")
    return silhouette_scores


def plot_bar_chart(data_series, title, output_filename, top_n=TOP_N_CATEGORIES_PLOT, color='skyblue'):
    """Plots and saves a horizontal bar chart for top N categories."""
    if data_series.empty:
        print(f"No data for bar chart: {title}")
        return
    counts = data_series.value_counts().nlargest(top_n)
    plt.figure(figsize=(12, 8))
    counts.sort_values().plot(kind='barh', color=color)
    plt.title(title)
    plt.xlabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, output_filename))
    plt.close()
    print(f"Bar chart '{title}' saved.")

def plot_pareto_chart(data_series, title, output_filename, top_n=TOP_N_CATEGORIES_PLOT):
    """Plots and saves a Pareto chart."""
    if data_series.empty:
        print(f"No data for Pareto chart: {title}")
        return
    counts = data_series.value_counts().nlargest(top_n)
    df_pareto = pd.DataFrame({'count': counts})
    df_pareto = df_pareto.sort_values(by='count', ascending=False)
    df_pareto['cumulative_percentage'] = (df_pareto['count'].cumsum() / df_pareto['count'].sum()) * 100

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.bar(df_pareto.index, df_pareto['count'], color='skyblue')
    ax1.set_xlabel('Complaint Category')
    ax1.set_ylabel('Number of Complaints', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    plt.xticks(rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(df_pareto.index, df_pareto['cumulative_percentage'], color='red', marker='o', ms=5)
    ax2.set_ylabel('Cumulative Percentage (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 110]) # Ensure 100% is visible

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, output_filename))
    plt.close()
    print(f"Pareto chart '{title}' saved.")


def plot_time_series(df, date_col, value_col_series, title, output_filename, freq='M'):
    """Plots time series for a given value column, resampled by frequency."""
    if date_col not in df.columns or value_col_series.empty:
        print(f"Date column '{date_col}' or value series missing for time series: {title}")
        return
    
    # Ensure date_col is datetime
    try:
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors='coerce')
        df_ts = df_ts.dropna(subset=[date_col]) # Remove rows where date couldn't be parsed
        if df_ts.empty:
            print(f"No valid dates found in column '{date_col}' for time series plot.")
            return
        df_ts = df_ts.set_index(date_col)
    except Exception as e:
        print(f"Error processing date column for time series '{title}': {e}")
        return

    plt.figure(figsize=(14, 7))
    # Assuming value_col_series is already grouped counts by category
    # We need to resample counts of each top category over time
    top_categories = value_col_series.index # These are the categories to track
    
    for category in top_categories:
        # Filter original df for this category, then resample
        category_ts_data = df_ts[df_ts[MOTIVE_COLUMN] == category] # Assuming MOTIVE_COLUMN holds the category
        if not category_ts_data.empty:
            category_ts_data.resample(freq).size().plot(label=category, marker='o', linestyle='-')
            
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(f'Number of Complaints (Resampled by {freq})')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1)) # Move legend outside plot
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, output_filename))
    plt.close()
    print(f"Time series plot '{title}' saved.")


def generate_word_clouds(df, text_col, cluster_col, top_n_clusters, output_dir):
    """Generates word clouds for the top N clusters."""
    if text_col not in df.columns or cluster_col not in df.columns:
        print(f"Required columns ('{text_col}', '{cluster_col}') not found for word clouds.")
        return

    df_cleaned = df.dropna(subset=[text_col, cluster_col])
    if df_cleaned.empty:
        print("No data available for word cloud generation after dropping NaNs.")
        return

    # Ensure cluster_col is treated as integer if it's float due to NaNs from original load
    if pd.api.types.is_numeric_dtype(df_cleaned[cluster_col]):
         df_cleaned[cluster_col] = df_cleaned[cluster_col].astype(int)

    top_clusters = df_cleaned[cluster_col].value_counts().nlargest(top_n_clusters).index
    
    spanish_stopwords = list(stopwords.words('spanish'))
    # Add custom stopwords if needed:
    # custom_stopwords = ['cliente', 'n1co', 'app', 'favor', 'gracias', 'hola', 'buenos', 'dias', 'tardes', 'noches']
    # spanish_stopwords.extend(custom_stopwords)

    for cluster_id in top_clusters:
        cluster_texts = " ".join(df_cleaned[df_cleaned[cluster_col] == cluster_id][text_col])
        if not cluster_texts.strip():
            print(f"No text available for cluster {cluster_id} to generate word cloud.")
            continue
        try:
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  stopwords=spanish_stopwords, min_font_size=10).generate(cluster_texts)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for Cluster {cluster_id}')
            filename = os.path.join(output_dir, f'wordcloud_cluster_{cluster_id}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Word cloud for cluster {cluster_id} saved.")
        except ValueError as ve: # Handle cases where all words are stopwords
             print(f"Could not generate word cloud for cluster {cluster_id} (ValueError): {ve}")
        except Exception as e:
            print(f"Error generating word cloud for cluster {cluster_id}: {e}")


def plot_heatmap(df, rows_col, cols_col, title, output_filename):
    """Plots and saves a heatmap for the relationship between two categorical columns."""
    if rows_col not in df.columns or cols_col not in df.columns:
        print(f"Required columns ('{rows_col}', '{cols_col}') not found for heatmap.")
        return
    
    df_cleaned = df.dropna(subset=[rows_col, cols_col])
    if df_cleaned.empty:
        print(f"No data for heatmap '{title}' after dropping NaNs.")
        return

    contingency_table = pd.crosstab(df_cleaned[rows_col], df_cleaned[cols_col])
    if contingency_table.empty:
        print(f"Contingency table is empty for heatmap: {title}")
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, output_filename))
    plt.close()
    print(f"Heatmap '{title}' saved.")


def perform_sentiment_analysis(texts):
    """Performs sentiment analysis on a list of texts using pysentimiento."""
    print("\nPerforming sentiment analysis (this may take some time)...")
    # Using a general sentiment analyzer, can be adapted for specific tasks or languages
    # For Spanish, pysentimiento is excellent.
    analyzer = create_analyzer(task="sentiment", lang="es") 
    sentiments = []
    for i, text in enumerate(texts):
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            sentiments.append({'label': 'NEUTRAL', 'score': 0.0}) # Default for empty/NaN
            continue
        try:
            # Pysentimiento returns an object with output (POS, NEU, NEG) and probas
            result = analyzer.predict(text)
            sentiments.append({'label': result.output, 'score': result.probas.get(result.output, 0.0)})
        except Exception as e:
            print(f"Error during sentiment analysis for text {i}: {e}")
            sentiments.append({'label': 'ERROR', 'score': 0.0})
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(texts)} texts for sentiment...")
            
    print("Sentiment analysis complete.")
    return pd.DataFrame(sentiments)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Advanced Complaint Analysis Toolkit...")

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

    # --- Data Cleaning and Preparation ---
    # Ensure text columns are strings and handle NaNs for processing
    for col in [TEXT_COLUMN, PROCESSED_TEXT_COLUMN, MOTIVE_COLUMN, NLP_KEYWORDS_COLUMN, COUNTRY_COLUMN]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('')
        else:
            print(f"Warning: Expected column '{col}' not found. Some analyses might be affected.")
    
    # Parse date column
    if DATE_COLUMN in df.columns:
        try:
            # Attempt to parse, coercing errors to NaT
            df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
            # Check how many dates were successfully parsed
            valid_dates_count = df[DATE_COLUMN].notna().sum()
            print(f"Parsed '{DATE_COLUMN}'. Found {valid_dates_count} valid dates out of {len(df)} rows.")
            if valid_dates_count == 0:
                 print(f"Warning: No valid dates could be parsed from '{DATE_COLUMN}'. Time series analysis will be skipped.")
        except Exception as e:
            print(f"Warning: Could not parse date column '{DATE_COLUMN}': {e}. Time series analysis might fail or be inaccurate.")
    else:
        print(f"Warning: Date column '{DATE_COLUMN}' not found. Time series analysis will be skipped.")


    # --- Cluster Optimization Guidance (Example Call - requires embeddings) ---
    # Note: Embeddings are not re-generated in this script.
    # You would typically run this on the embeddings from your categorization script.
    # For demonstration, this part is commented out. If you have embeddings saved (e.g., as .npy):
    # try:
    #     embeddings_for_optim = np.load('path_to_your_embeddings.npy') # Replace with actual path
    #     if embeddings_for_optim is not None and len(embeddings_for_optim) > 0:
    #         plot_elbow_method(embeddings_for_optim, max_k=20)
    #         plot_elbow_method(embeddings_for_optim, max_k=20, metric='silhouette') # KElbow with silhouette
    #         calculate_silhouette_scores(embeddings_for_optim, range_n_clusters=[2, 5, 10, 15, 20, 25])
    # except FileNotFoundError:
    #     print("Embeddings file not found. Skipping cluster optimization guidance plots.")
    # except Exception as e:
    #     print(f"Error during cluster optimization guidance: {e}")


    # --- Advanced Visualizations ---
    print("\nGenerating visualizations...")
    # 1. Bar Charts
    if MOTIVE_COLUMN in df.columns:
        plot_bar_chart(df[MOTIVE_COLUMN],
                       f'Top {TOP_N_CATEGORIES_PLOT} Complaint Motives ({MOTIVE_COLUMN})',
                       'bar_motivo_solicitud.png')
    if NLP_CLUSTER_COLUMN in df.columns:
        # Create descriptive labels for NLP clusters if keywords are available
        # This part assumes NLP_CLUSTER_COLUMN is numeric and NLP_KEYWORDS_COLUMN has representative keywords
        cluster_counts_for_plot = df[NLP_CLUSTER_COLUMN].value_counts().nlargest(TOP_N_CATEGORIES_PLOT)
        
        # Attempt to map cluster numbers to keyword labels for the plot
        if NLP_KEYWORDS_COLUMN in df.columns:
            # Create a mapping from cluster number to its primary keyword(s)
            # This is a simplified approach; you might have a better mapping
            cluster_label_map = df.dropna(subset=[NLP_CLUSTER_COLUMN, NLP_KEYWORDS_COLUMN])\
                                  .drop_duplicates(subset=[NLP_CLUSTER_COLUMN])\
                                  .set_index(NLP_CLUSTER_COLUMN)[NLP_KEYWORDS_COLUMN]\
                                  .to_dict()
            
            # Ensure cluster IDs in counts are compatible with map keys (e.g. both int or both str)
            # If NLP_CLUSTER_COLUMN can be float due to NaNs, convert to int after dropna
            current_cluster_labels = []
            for idx in cluster_counts_for_plot.index:
                try:
                    # Attempt to convert cluster ID to the type used in cluster_label_map (usually int or float if NaNs were present)
                    # This can be tricky if the original cluster IDs had mixed types or were non-numeric.
                    # For simplicity, we assume cluster_label_map keys are compatible with cluster_counts_for_plot.index
                    # A safer way is to ensure NLP_CLUSTER_COLUMN is consistently typed before this.
                    key = int(float(idx)) # Try to make it int if it's numeric
                    label = cluster_label_map.get(key, f"Cluster {idx}")
                    # Take first few keywords for brevity
                    label_short = ', '.join(str(label).split(',')[:2]).strip()
                    current_cluster_labels.append(f"C{idx}: {label_short}")
                except (ValueError, TypeError):
                     current_cluster_labels.append(f"Cluster {idx}") # Fallback
            
            if len(current_cluster_labels) == len(cluster_counts_for_plot):
                 cluster_counts_for_plot.index = current_cluster_labels
            else: # Fallback if mapping failed
                 cluster_counts_for_plot.index = "Cluster " + cluster_counts_for_plot.index.astype(str)
        else:
            cluster_counts_for_plot.index = "Cluster " + cluster_counts_for_plot.index.astype(str)

        plt.figure(figsize=(12, 8))
        cluster_counts_for_plot.sort_values().plot(kind='barh', color='coral')
        plt.title(f'Top {TOP_N_CATEGORIES_PLOT} NLP Clusters')
        plt.xlabel('Number of Complaints')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'bar_nlp_clusters.png'))
        plt.close()
        print(f"Bar chart 'Top {TOP_N_CATEGORIES_PLOT} NLP Clusters' saved.")


    # 2. Pareto Chart
    if MOTIVE_COLUMN in df.columns:
        plot_pareto_chart(df[MOTIVE_COLUMN],
                          f'Pareto Chart for Top {TOP_N_CATEGORIES_PLOT} Complaint Motives',
                          'pareto_motivo_solicitud.png')

    # 3. Trend Analysis (Time Series)
    if DATE_COLUMN in df.columns and MOTIVE_COLUMN in df.columns and df[DATE_COLUMN].notna().any():
        top_motives_for_ts = df[MOTIVE_COLUMN].value_counts().nlargest(5) # Track top 5 motives
        plot_time_series(df, DATE_COLUMN, top_motives_for_ts,
                         f'Trend of Top 5 Complaint Motives ({MOTIVE_COLUMN})',
                         'timeseries_top_motives.png', freq='M') # Monthly frequency

    # 4. Word Clouds
    # Using PROCESSED_TEXT_COLUMN for cleaner word clouds if available, else TEXT_COLUMN
    text_col_for_wc = PROCESSED_TEXT_COLUMN if PROCESSED_TEXT_COLUMN in df.columns and df[PROCESSED_TEXT_COLUMN].notna().any() else TEXT_COLUMN
    if NLP_CLUSTER_COLUMN in df.columns and text_col_for_wc in df.columns :
        generate_word_clouds(df, text_col_for_wc, NLP_CLUSTER_COLUMN, TOP_N_CLUSTERS_WORDCLOUD, OUTPUT_PLOTS_DIR)

    # 5. Heatmap
    if NLP_CLUSTER_COLUMN in df.columns and COUNTRY_COLUMN in df.columns:
        plot_heatmap(df, NLP_CLUSTER_COLUMN, COUNTRY_COLUMN,
                     f'Heatmap: NLP Clusters vs. {COUNTRY_COLUMN}',
                     'heatmap_nlp_cluster_vs_country.png')

    # --- Deeper Analytical Approaches ---
    # 1. Sentiment Analysis
    if TEXT_COLUMN in df.columns:
        sentiment_results_df = perform_sentiment_analysis(df[TEXT_COLUMN].tolist())
        df['sentiment_label'] = sentiment_results_df['label']
        df['sentiment_score'] = sentiment_results_df['score']
        print("\nSentiment Analysis Results (sample):")
        print(df[['sentiment_label', 'sentiment_score']].head())
        
        # Plot sentiment distribution
        if not df['sentiment_label'].empty:
            plt.figure(figsize=(8, 6))
            df['sentiment_label'].value_counts().plot(kind='bar', color=['red', 'lightgreen', 'skyblue', 'orange']) # Adjust colors as needed
            plt.title('Sentiment Distribution of Complaints')
            plt.ylabel('Number of Complaints')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'sentiment_distribution.png'))
            plt.close()
            print("Sentiment distribution plot saved.")


    # 2. Cross-Tabulation and Chi-squared Test
    if NLP_CLUSTER_COLUMN in df.columns and COUNTRY_COLUMN in df.columns:
        print(f"\nCross-Tabulation: NLP Clusters vs. {COUNTRY_COLUMN}")
        # Ensure columns are clean for crosstab (no NaNs in these specific columns for chi2)
        df_crosstab = df.dropna(subset=[NLP_CLUSTER_COLUMN, COUNTRY_COLUMN])

        if not df_crosstab.empty:
            contingency_table = pd.crosstab(df_crosstab[NLP_CLUSTER_COLUMN], df_crosstab[COUNTRY_COLUMN])
            print(contingency_table)

            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1: # Chi2 needs >1 row/col
                try:
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    print(f"\nChi-squared Test Results (NLP Cluster vs. {COUNTRY_COLUMN}):")
                    print(f"  Chi2 Statistic: {chi2:.4f}")
                    print(f"  P-value: {p:.4f}")
                    print(f"  Degrees of Freedom: {dof}")
                    if p < 0.05:
                        print("  Result: Significant association (p < 0.05)")
                    else:
                        print("  Result: No significant association (p >= 0.05)")
                except ValueError as ve_chi2: # Can occur if expected frequencies are too low
                    print(f"  Chi-squared test could not be performed: {ve_chi2}")
            else:
                print("  Chi-squared test not performed (table too small).")
        else:
            print("Not enough data for cross-tabulation after dropping NaNs.")


    # --- Save Enhanced DataFrame ---
    try:
        df.to_csv(OUTPUT_ENHANCED_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\nEnhanced data with sentiment saved to {OUTPUT_ENHANCED_CSV_PATH}")
    except Exception as e:
        print(f"Error saving enhanced CSV: {e}")

    print("\nAdvanced Complaint Analysis Toolkit finished.")
    print(f"Plots saved in: {OUTPUT_PLOTS_DIR}")

