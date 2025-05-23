# Install necessary libraries if you haven't already:
# pip install pandas matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CATEGORIZED_COMPLAINTS_CSV = 'categorized_complaints.csv'
CLUSTER_DISTRIBUTION_CSV = 'cluster_distribution.csv' # Path to your second CSV
OUTPUT_CHART_MOTIVO = 'top_motivos_pie_chart.png'
OUTPUT_CHART_NLP_CLUSTER = 'top_nlp_clusters_pie_chart.png'
OUTPUT_CHART_DISTRIBUTION = 'cluster_distribution_pie_chart.png'

TOP_N_CATEGORIES = 10 # Number of top categories to show in pie charts (plus 'Other')

# --- Helper Function to Create Pie Chart ---
def create_pie_chart(data_series, title, output_filename, top_n=TOP_N_CATEGORIES):
    """
    Creates a pie chart from a pandas Series (value counts) and saves it.
    Shows top_n categories and groups the rest into 'Other'.
    """
    if data_series.empty:
        print(f"No data to plot for '{title}'. Skipping chart generation.")
        return

    # Get top N categories and sum the rest into 'Other'
    if len(data_series) > top_n:
        top_categories = data_series.nlargest(top_n)
        other_sum = data_series.nsmallest(len(data_series) - top_n).sum()
        if other_sum > 0:
            top_categories['Other'] = other_sum
        plot_data = top_categories
    else:
        plot_data = data_series

    plt.figure(figsize=(12, 9)) # Adjusted for better label visibility
    
    # Use a colormap for more distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))

    wedges, texts, autotexts = plt.pie(
        plot_data,
        labels=None, # We'll create a custom legend
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.85, # Distance of percentage text from center
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='w') # For a donut-like effect
    )

    plt.title(title, fontsize=16, pad=20)
    
    # Create a legend with labels and percentages
    legend_labels = [f'{label} ({percent:.1f}%)' for label, percent in zip(plot_data.index, (plot_data/plot_data.sum()*100))]
    plt.legend(wedges, legend_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout(rect=[0, 0, 0.75, 1]) # Adjust layout to make space for legend

    try:
        plt.savefig(output_filename)
        print(f"Pie chart '{title}' saved as {output_filename}")
    except Exception as e:
        print(f"Error saving chart {output_filename}: {e}")
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting pie chart generation process...")

    # 1. Analyze 'categorized_complaints.csv'
    try:
        df_categorized = pd.read_csv(CATEGORIZED_COMPLAINTS_CSV)
        print(f"Successfully loaded {CATEGORIZED_COMPLAINTS_CSV}. Shape: {df_categorized.shape}")

        # Pie chart for 'Motivo de su solicitud'
        if 'Motivo de su solicitud' in df_categorized.columns:
            motivo_counts = df_categorized['Motivo de su solicitud'].value_counts()
            create_pie_chart(motivo_counts,
                             f'Top {TOP_N_CATEGORIES} Complaint Motives (Motivo de su solicitud)',
                             OUTPUT_CHART_MOTIVO)
        else:
            print(f"Column 'Motivo de su solicitud' not found in {CATEGORIZED_COMPLAINTS_CSV}.")

        # Pie chart for 'nlp_cluster'
        if 'nlp_cluster' in df_categorized.columns and 'nlp_cluster_keywords' in df_categorized.columns:
            # Create more descriptive labels for NLP clusters if keywords are available
            # This part attempts to create meaningful labels from keywords.
            # It takes the first keyword as a representative label.
            # You might want to refine this based on your keywords.
            
            # Handle potential NaN values in nlp_cluster before value_counts
            df_categorized_not_na_nlp = df_categorized.dropna(subset=['nlp_cluster'])
            if not df_categorized_not_na_nlp.empty:
                # Ensure nlp_cluster is integer for proper mapping if it's float due to NaNs
                df_categorized_not_na_nlp['nlp_cluster'] = df_categorized_not_na_nlp['nlp_cluster'].astype(int)

                nlp_cluster_counts = df_categorized_not_na_nlp['nlp_cluster'].value_counts()
                
                # Create a mapping from cluster number to a descriptive label (e.g., first keyword)
                # Group by cluster and take the first keyword string for that cluster
                # Assuming 'nlp_cluster_keywords' contains comma-separated keywords
                cluster_label_map = {}
                if 'nlp_cluster_keywords' in df_categorized_not_na_nlp.columns:
                    # Ensure keywords are strings
                    df_categorized_not_na_nlp['nlp_cluster_keywords'] = df_categorized_not_na_nlp['nlp_cluster_keywords'].astype(str)
                    
                    # Get the most frequent keyword string for each cluster as its label
                    # This might not always be the 'best' keyword, but it's an approach.
                    # A more robust way would be to use the keywords generated by the previous script directly if they were saved per cluster.
                    
                    # For simplicity, we'll use the keyword string associated with the first occurrence of each cluster.
                    # This assumes the keywords are somewhat consistent for a cluster.
                    unique_clusters_df = df_categorized_not_na_nlp.drop_duplicates(subset=['nlp_cluster'])
                    cluster_label_map = pd.Series(
                        unique_clusters_df.nlp_cluster_keywords.values,
                        index=unique_clusters_df.nlp_cluster
                    ).to_dict()
                    
                    # Extract first keyword or a short representation
                    for k, v in cluster_label_map.items():
                        first_keyword = v.split(',')[0].strip()
                        cluster_label_map[k] = f"Cluster {k}: {first_keyword}" if first_keyword and first_keyword != "N/A - Original comment was empty or not clustered" else f"Cluster {k}"


                # Rename index of nlp_cluster_counts with these descriptive labels
                nlp_cluster_counts.index = nlp_cluster_counts.index.map(lambda x: cluster_label_map.get(x, f"Cluster {x}"))
                
                create_pie_chart(nlp_cluster_counts,
                                 f'Top {TOP_N_CATEGORIES} NLP Clusters (from Comentarios)',
                                 OUTPUT_CHART_NLP_CLUSTER)
            else:
                print(f"No valid 'nlp_cluster' data to plot in {CATEGORIZED_COMPLAINTS_CSV} after dropping NaNs.")

        elif 'nlp_cluster' in df_categorized.columns: # Fallback if keywords column is missing
            nlp_cluster_counts = df_categorized['nlp_cluster'].value_counts()
            nlp_cluster_counts.index = "Cluster " + nlp_cluster_counts.index.astype(str)
            create_pie_chart(nlp_cluster_counts,
                             f'Top {TOP_N_CATEGORIES} NLP Clusters (from Comentarios)',
                             OUTPUT_CHART_NLP_CLUSTER)
        else:
            print(f"Column 'nlp_cluster' not found in {CATEGORIZED_COMPLAINTS_CSV}.")


    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {CATEGORIZED_COMPLAINTS_CSV}")
    except Exception as e:
        print(f"Error processing {CATEGORIZED_COMPLAINTS_CSV}: {e}")

    # 2. Analyze 'cluster_distribution.csv'
    try:
        df_distribution = pd.read_csv(CLUSTER_DISTRIBUTION_CSV)
        print(f"\nSuccessfully loaded {CLUSTER_DISTRIBUTION_CSV}. Shape: {df_distribution.shape}")

        # Assuming 'cluster_distribution.csv' has two columns:
        # One for category/cluster labels (e.g., 'ClusterLabel' or 'Category')
        # One for counts (e.g., 'Count' or 'Frequency')
        # You might need to adjust these column names based on your actual file.
        
        # Try to infer label and count columns
        label_col = None
        count_col = None

        if len(df_distribution.columns) >= 2:
            # Attempt to find a string column for labels and a numeric column for counts
            for col in df_distribution.columns:
                if df_distribution[col].dtype == 'object' and label_col is None:
                    label_col = col
                elif pd.api.types.is_numeric_dtype(df_distribution[col]) and count_col is None:
                    count_col = col
            
            if label_col and count_col:
                print(f"Using '{label_col}' as label column and '{count_col}' as count column from {CLUSTER_DISTRIBUTION_CSV}.")
                # Create a Series suitable for the pie chart function
                distribution_counts = pd.Series(df_distribution[count_col].values, index=df_distribution[label_col])
                distribution_counts = distribution_counts.sort_values(ascending=False) # Ensure it's sorted for top_n

                create_pie_chart(distribution_counts,
                                 f'Top {TOP_N_CATEGORIES} from Cluster Distribution File',
                                 OUTPUT_CHART_DISTRIBUTION)
            else:
                print(f"Could not automatically determine label and count columns in {CLUSTER_DISTRIBUTION_CSV}.")
                print("Please ensure it has one text column for labels and one numeric column for counts.")
                print(f"Detected column types: {df_distribution.dtypes}")

        else:
            print(f"{CLUSTER_DISTRIBUTION_CSV} does not have at least two columns. Cannot generate pie chart.")


    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {CLUSTER_DISTRIBUTION_CSV}")
    except Exception as e:
        print(f"Error processing {CLUSTER_DISTRIBUTION_CSV}: {e}")

    print("\nPie chart generation process complete.")
    print(f"Charts saved as: {OUTPUT_CHART_MOTIVO}, {OUTPUT_CHART_NLP_CLUSTER}, {OUTPUT_CHART_DISTRIBUTION} (if data was available).")

