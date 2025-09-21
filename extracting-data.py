## CMPT 353
## Greg Baker
## Final Project
## Due Date: August 13st, 2025
## Author: Stella Maltcheva 

import sys
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.functions import col, rand, row_number
from pyspark.sql import Window

spark = SparkSession.builder.appName('reddit extracter').getOrCreate()

reddit_submissions_path = '/courses/datasets/reddit_submissions_repartitioned/year=2024/month={09,10,11,12}'
reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/year=2024/month={09,10,11,12}'
output = 'reddit-subset'

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

submissions_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('created', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('domain', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.BooleanType()),
    types.StructField('from', types.StringType()),
    types.StructField('from_id', types.StringType()),
    types.StructField('from_kind', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('hide_score', types.BooleanType()),
    types.StructField('id', types.StringType()),
    types.StructField('is_self', types.BooleanType()),
    types.StructField('link_flair_css_class', types.StringType()),
    types.StructField('link_flair_text', types.StringType()),
    types.StructField('media', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('num_comments', types.LongType()),
    types.StructField('over_18', types.BooleanType()),
    types.StructField('permalink', types.StringType()),
    types.StructField('quarantine', types.BooleanType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('saved', types.BooleanType()),
    types.StructField('score', types.LongType()),
    types.StructField('secure_media', types.StringType()),
    types.StructField('selftext', types.StringType()),
    types.StructField('stickied', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('thumbnail', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('url', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

def main():
    chosen_subreddits = [
        'AskReddit', 'relationships', 'todayilearned',       # Social
        'politics', 'worldnews', 'SandersForPresident',      # News/Politics
        'gaming', 'movies', 'leagueoflegends'                # Entertainment/Gaming
    ]

    SAMPLE_FRACTION = 0.01  # 1% sampling per subreddit
    MAX_COMMENTS_PER_SUBREDDIT = 5000  # cap per subreddit

    # fractions dict without for loop
    fractions = dict(zip(chosen_subreddits, [SAMPLE_FRACTION] * len(chosen_subreddits)))

    print("Reading submissions...")
    reddit_submissions = spark.read.json(reddit_submissions_path, schema=submissions_schema)

    print("Reading comments...")
    reddit_comments = spark.read.json(reddit_comments_path, schema=comments_schema)

    print("Filtering comments to chosen subreddits...")
    filtered_comments = reddit_comments.filter(col('subreddit').isin(chosen_subreddits))

    print("Sampling approx 1% comments per subreddit...")
    sampled_comments = filtered_comments.sampleBy('subreddit', fractions=fractions, seed=42)

    print(f"Count after sampling: {sampled_comments.count()}")

    print(f"Capping to max {MAX_COMMENTS_PER_SUBREDDIT} comments per subreddit randomly...")
    window = Window.partitionBy('subreddit').orderBy(rand())
    capped_comments = (
        sampled_comments
        .withColumn('row_num', row_number().over(window))
        .filter(col('row_num') <= MAX_COMMENTS_PER_SUBREDDIT)
        .drop('row_num')
    )

    print(f"Count after capping: {capped_comments.count()}")

    print("Filtering submissions to chosen subreddits...")
    filtered_submissions = reddit_submissions.filter(col('subreddit').isin(chosen_subreddits))

    print("Writing submissions...")
    filtered_submissions.write.json(output + '/submissions_2', mode='overwrite', compression='gzip')

    print("Writing capped comments...")
    capped_comments.write.json(output + '/comments_2', mode='overwrite', compression='gzip')

    print("✅ All done!")

main()
