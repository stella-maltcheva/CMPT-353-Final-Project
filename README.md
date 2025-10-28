# CMPT 353 Final Project - Reddit Comment Analysis

**Author:** Stella Maltcheva  
**Instructor:** Greg Baker  
**Course:** CMPT 353 - Computational Data Science  
**Due Date:** August 13, 2025  

---

## Project Overview
This project explores Reddit comment data to investigate patterns in user engagement, comment sentiment, and posting behavior across different subreddit categories. The analysis focuses on:

- How comment timing (hour of day, day of week) relates to relative scores.
- The relationship between comment sentiment and scores/comment count.
- Differences in scores and sentiment between subreddit categories (General, Politics, Entertainment).

The dataset consists of Reddit comments and submissions.

## Instructions
Please run the extracting-data.py file on the SFU cluster, ensuring you have all libraries installed. An output directory is required for this file and it will extract around 40,000 comments from 9 specific subreddits in three categories. Afterwards, run the final-analysis.py file. This file requires an input directory (where you saved the Reddit subset) and an output directory for plots and visualizations.


---

## Requirements
The project relies on the following Python libraries:

- pandas  
- matplotlib  
- seaborn  
- scipy  
- statsmodels  
- pyspark  
- nltk  
- scikit-learn  
