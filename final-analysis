## CMPT 353
## Greg Baker
## Final Project
## Due Date: August 13st, 2025
## Author: Stella Maltcheva 

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pyspark.sql import SparkSession, types, functions as F
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# ----------------- NLTK Setup -----------------
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("VADER lexicon not found, downloading...")
    nltk.download('vader_lexicon', quiet=True)

sid = SentimentIntensityAnalyzer()

# ----------------- Helper Functions -----------------
def vader_sentiment(text):
    """Calculates the VADER compound sentiment score for a given text."""
    if not isinstance(text, str):
        return 0.0 # Return neutral sentiment for non-string input
    return float(sid.polarity_scores(text.lower())['compound'])

def ensure_dir(path):
    """Ensures a directory exists, creating it if necessary."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sns_lineplot_safe(*args, **kwargs):
    """Wrapper for seaborn.lineplot to handle different versions."""
    try:
        return sns.lineplot(*args, errorbar=None, **kwargs)
    except TypeError:
        return sns.lineplot(*args, ci=None, **kwargs)

# ----------------- Main Analysis Function -----------------
def main(input_dir, output_dir):
    """
    Main function to perform Reddit data analysis and generate plots.
    
    Args:
        input_dir (str): The directory containing the input data files.
        output_dir (str): The directory where output plots and files will be saved.
    """
    ensure_dir(output_dir)
    spark = SparkSession.builder.appName("reddit_comment_analysis").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Define schema for the input data
    schema = types.StructType([
        types.StructField('author', types.StringType()),
        types.StructField('body', types.StringType()),
        types.StructField('score', types.LongType()),
        types.StructField('created_utc', types.StringType()),
        types.StructField('subreddit', types.StringType())
    ])

    # Load and preprocess data with PySpark
    print("Loading data...")
    comments = spark.read.json(f"{input_dir}/comments_2", schema=schema).cache()
    comments = comments.filter(
        (F.col("author") != "[deleted]") & 
        (~F.col("body").rlike(r"^The above submission has been \*\*removed\*\*"))
    ).cache()

    # Convert to pandas DataFrame for detailed analysis and plotting
    print("Converting to Pandas DataFrame...")
    pdf = comments.toPandas()

    # Subreddit to Category mapping with the new categories
    category_map = {
        "AskReddit":"General","relationships":"General","todayilearned":"General",
        "politics":"Politics","worldnews":"Politics","SandersForPresident":"Politics",
        "gaming":"Entertainment","movies":"Entertainment","leagueoflegends":"Entertainment",
    }
    pdf['category'] = pdf['subreddit'].map(category_map)

    # Filter out subreddits not in our mapping
    pdf = pdf.dropna(subset=['category'])

    print("Generating new features...")
    pdf['sentiment'] = pdf['body'].fillna("").apply(vader_sentiment)
    
    # Calculate Relative Score
    subreddit_mean_scores = pdf.groupby('subreddit')['score'].mean()
    pdf['rel_score'] = pdf.apply(lambda row: row['score'] / subreddit_mean_scores.get(row['subreddit'], 1), axis=1)

    # Convert UTC to PST
    pdf['created_ts'] = pd.to_numeric(pdf['created_utc'], errors='coerce')
    pdf['created_dt'] = pd.to_datetime(pdf['created_ts'], unit='s', utc=True)
    pdf['created_pst'] = pdf['created_dt'].dt.tz_convert('US/Pacific')
    pdf['hour_pst'] = pdf['created_pst'].dt.hour.astype(int)
    pdf['dow_pst'] = pd.Categorical(
        pdf['created_pst'].dt.day_name(),
        categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
        ordered=True
    )

    sns.set(style="whitegrid")
    
        # ----------------- Entertainment: Thursday vs Monday -----------------
    entertainment = pdf[pdf['category'] == 'Entertainment']
    thursday_scores = entertainment[entertainment['dow_pst'] == 'Thursday']['rel_score']
    monday_scores = entertainment[entertainment['dow_pst'] == 'Monday']['rel_score']

    if len(thursday_scores) > 1 and len(monday_scores) > 1:
        t_ent, p_ent = stats.ttest_ind(thursday_scores, monday_scores, equal_var=False)
        print(f"Entertainment relative score: Thursday vs Monday | t={t_ent:.3f} | p={p_ent:.4f}")

    # ----------------- Politics sentiment: Politics vs rest -----------------
    politics = pdf[pdf['category'] == 'Politics']
    rest = pdf[pdf['category'] != 'Politics']['sentiment']

    if len(politics) > 1 and len(rest) > 1:
        t_pol, p_pol = stats.ttest_ind(politics['sentiment'], rest, equal_var=False)
        print(f"Politics sentiment: Politics vs rest | t={t_pol:.3f} | p={p_pol:.4f}")
        
        
    # ----------------- Overall average scores: 3AM+5AM vs 5PM (PST) -----------------
    early_hours = pdf[pdf['hour_pst'].isin([3, 5])]['rel_score']
    evening_hour = pdf[pdf['hour_pst'] == 17]['rel_score']

    if len(early_hours) > 1 and len(evening_hour) > 1:
        t_time, p_time = stats.ttest_ind(early_hours, evening_hour, equal_var=False)
        print(f"Overall relative score: 3AM+5AM vs 5PM | t={t_time:.3f} | p={p_time:.4f}")
        
    # ----------------- Politics: 3AM vs 8AM + 5PM -----------------
    politics = pdf[pdf['category'] == 'Politics']
    politics_3am = politics[politics['hour_pst'] == 3]['rel_score']
    politics_8_17 = politics[politics['hour_pst'].isin([8, 17])]['rel_score']

    if len(politics_3am) > 1 and len(politics_8_17) > 1:
        t_pol_time, p_pol_time = stats.ttest_ind(politics_3am, politics_8_17, equal_var=False)
        print(f"Politics relative score: 3AM vs 8AM+5PM | t={t_pol_time:.3f} | p={p_pol_time:.4f}")
        
    # ----------------- General: Thursday vs Monday -----------------
    general = pdf[pdf['category'] == 'General']
    gen_thursday = general[general['dow_pst'] == 'Thursday']['rel_score']
    gen_monday = general[general['dow_pst'] == 'Monday']['rel_score']

    if len(gen_thursday) > 1 and len(gen_monday) > 1:
        t_gen_tm, p_gen_tm = stats.ttest_ind(gen_thursday, gen_monday, equal_var=False)
        print(f"General relative score: Thursday vs Monday | t={t_gen_tm:.3f} | p={p_gen_tm:.4f}")

    # ----------------- General: Thursday vs Rest -----------------
    gen_rest = general[general['dow_pst'] != 'Thursday']['rel_score']

    if len(gen_thursday) > 1 and len(gen_rest) > 1:
        t_gen_tr, p_gen_tr = stats.ttest_ind(gen_thursday, gen_rest, equal_var=False)
        print(f"General relative score: Thursday vs Rest | t={t_gen_tr:.3f} | p={p_gen_tr:.4f}")
    

    # Collect all raw p-values
    raw_pvals = [
        p_ent,      # Entertainment Thursday vs Monday
        p_pol,      # Politics vs rest
        p_time,     # Overall 3AM+5AM vs 5PM
        p_pol_time, # Politics 3AM vs 8AM+5PM
        p_gen_tm,   # General Thursday vs Monday
        p_gen_tr    # General Thursday vs Rest
    ]

    # Apply Bonferroni correction
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method='bonferroni')

    # Print results
    test_names = [
        "Entertainment Thursday vs Monday",
        "Politics vs rest",
        "Overall 3AM+5AM vs 5PM",
        "Politics 3AM vs 8AM+5PM",
        "General Thursday vs Monday",
        "General Thursday vs Rest"
    ]
    # Assunming for loop is is okay because values are only being printed
    for name, p_raw, p_corr, rej in zip(test_names, raw_pvals, pvals_corrected, reject):
        print(f"{name}: raw p={p_raw:.4f}, Bonferroni adj p={p_corr:.4f}, reject H0={rej}")
        
        
    # ----------------- Summary Table: Sentiment, Comment Length, Word Length -----------------
    pdf['comment_length_chars'] = pdf['body'].fillna("").apply(len)
    pdf['comment_length_words'] = pdf['body'].fillna("").apply(lambda x: len(x.split()))

    summary_table = pdf.groupby('category').agg(
        mean_sentiment=('sentiment', 'mean'),
        mean_length_chars=('comment_length_chars', 'mean'),
        mean_length_words=('comment_length_words', 'mean'),
        count_comments=('body', 'count')
    ).reset_index()
    
    # ----------------- Heatmap: Number of Comments per Day -----------------
    comment_counts = pdf.groupby(['dow_pst', 'hour_pst']).size().unstack(fill_value=0)

    plt.figure(figsize=(12,6))
    sns.heatmap(comment_counts, cmap="YlGnBu", linewidths=0.5, annot=True, fmt='d')
    plt.title("Number of Comments by Day and Hour (PST)")
    plt.xlabel("Hour (PST)")
    plt.ylabel("Day of Week (PST)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_comments_day_hour.png"))
    plt.close()

    # Save to CSV
    summary_csv_path = os.path.join(output_dir, "summary_stats_per_category.csv")
    summary_table.to_csv(summary_csv_path, index=False)

    print("\n📊 Summary statistics per category:")
    print(summary_table.to_string(index=False))
    print(f"Saved summary table to {summary_csv_path}")
    
    #------------Regression Model for comment count and comment length---------------
    from sklearn.linear_model import LinearRegression

    # Using summary_table
    X = summary_table[['count_comments']] 
    y = summary_table['mean_sentiment'] 

    model = LinearRegression()
    model.fit(X, y)

    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"R^2 score: {model.score(X, y):.4f}")
    
    ## Results show the above is not significant

    #----------------- Plotting Section -----------------
    print("Generating plots...")
    
    # Plot 1: Average Relative Score by Hour (PST) by Category
    plt.figure(figsize=(12,6))
    sns_lineplot_safe(data=pdf, x='hour_pst', y='rel_score', hue='category', estimator='mean')
    plt.title("Average Relative Score by Hour (PST) by Category")
    plt.xlabel("Hour (PST)")
    plt.ylabel("Average Relative Score")
    plt.savefig(os.path.join(output_dir, "avg_rel_score_hour_pst_by_category.png"))
    plt.close()

    # Plot 2: Average Relative Score by Day (PST) by Category
    plt.figure(figsize=(12,6))
    avg_rel_day_cat = pdf.groupby(['category', 'dow_pst'])['rel_score'].mean().reset_index()
    sns.barplot(x='dow_pst', y='rel_score', hue='category', data=avg_rel_day_cat, errorbar=None)
    plt.title("Average Relative Score by Day (PST) by Category")
    plt.xlabel("Day of Week (PST)")
    plt.ylabel("Average Relative Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_rel_score_day_pst_by_category.png"))
    plt.close()

# Plot 3: Average Sentiment: Top 20% vs Bottom 20% Scores
    def plot_top_bottom_sentiment(df, output_path):
        # Compute overall quantiles
        overall_top_q = df['rel_score'].quantile(0.8)
        overall_bottom_q = df['rel_score'].quantile(0.2)
        
        # Compute top and bottom 20% sentiment for all categories without apply
        top_q = df.groupby('category')['rel_score'].transform(lambda x: x.quantile(0.8))
        bottom_q = df.groupby('category')['rel_score'].transform(lambda x: x.quantile(0.2))

        # Top 20% per category
        pdf_top = df[df['rel_score'] >= top_q].groupby('category')['sentiment'].mean().reset_index()
        pdf_top['group'] = 'Top 20% Score'

        # Bottom 20% per category
        pdf_bottom = df[df['rel_score'] <= bottom_q].groupby('category')['sentiment'].mean().reset_index()
        pdf_bottom['group'] = 'Bottom 20% Score'

        # Combine category results
        top_bottom_data = pd.concat([pdf_top, pdf_bottom], ignore_index=True)
        top_bottom_data.rename(columns={'sentiment': 'avg_sentiment'}, inplace=True)

        # Add overall top/bottom
        overall_data = pd.DataFrame([
            {'category': 'Overall', 'avg_sentiment': df[df['rel_score'] >= overall_top_q]['sentiment'].mean(), 'group': 'Top 20% Score'},
            {'category': 'Overall', 'avg_sentiment': df[df['rel_score'] <= overall_bottom_q]['sentiment'].mean(), 'group': 'Bottom 20% Score'}
        ])
        
        # Combine with category data
        plot_df = pd.concat([top_bottom_data, overall_data], ignore_index=True)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='category', y='avg_sentiment', hue='group', data=plot_df, palette="tab10")
        plt.title('Average Sentiment: Top 20% vs Bottom 20% Scores')
        plt.ylabel('Average Sentiment Score')
        plt.xlabel('Category')
        plt.legend(title='Relative Score Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Call the function outside to generate the plot
    plot_top_bottom_sentiment(pdf, os.path.join(output_dir, "sentiment_top_bottom.png"))

    
    # Plot 4: Sentiment vs Relative Score by Category (Scatter Plot)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='sentiment', y='rel_score', hue='category', data=pdf, alpha=0.4, s=20)
    plt.title("Sentiment vs Relative Score by Category")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Relative Score")
    plt.legend(title='Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_sentiment_vs_rel_score.png"))
    plt.close()

    print(f"All plots saved to: {output_dir}")
    
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reddit_analysis.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    main(input_directory, output_directory)
