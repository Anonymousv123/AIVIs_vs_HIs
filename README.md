# Analyzing the Dynamics of User Interactions with AI-Virtual vs. Human Influencers

## Overview
As AI virtual influencers (AIVIs) proliferate, understanding human-AI emotional engagement is vital for advancing affective computing. To examine whether AIVIs evoke comparable affective responses to human influencers (HIs), we analyzed 156,918 Instagram comments across 12 million interactions using a multi-signal computational framework. We integrated Large Language Model (LLM)-derived knowledge graphs, emotion analysis, and machine learning. Results reveal structural interaction asymmetries. HI-directed discourse exhibits cohesive semantic networks and highly differentiated emotional expressions, whereas AIVI-directed discourse spans a fragmented semantic space characterized by emotional detachment and neutral affect. Our Multi-Signal Embedding Network (MSEN) model, incorporating behavioral metrics and OpenAI semantic embeddings, successfully classified influencer types based on these discourse signatures (F1=0.89), significantly outperforming baselines, including a state-of-the-art zero-shot LLM. Advancing the Computers Are Social Actors (CASA) paradigm, these findings demonstrate that human-like visual presentation does not guarantee human-like emotional connection, offering critical design implications for affect-aware systems in human-computer interaction.

## Data Availability

Due to Instagram’s content sharing policies, we cannot share the raw dataset. 
However, a list of publicly available Instagram post URLs used in this study is provided to enable data reconstruction for reproducibility purposes.

## Running the code
`data_analysis.R` - 
This R script performs sentiment and popularity analysis on Instagram comment data, comparing interactions between users and two types of influencers: human (HI) and AI-based (AIVI).
Cleans comment text by removing usernames,
Applies sentiment analysis,
Computes comment length,
Performs statistical comparisons across groups (HI vs. AIVI) for:
Popularity,
Likes per post,
Likes per comment,
Sentiment scores,
Comment length, and
Visualizes results

Output: A CSV file with sentiment results and several plots highlighting group differences.


`EP_analysis.py` -
Analysis of Estimated Earnings per Post

`Neural_Net.py` -
Loads a JSON file containing Instagram comment features and BERT-style sentence embeddings.
Cleans and prepares the dataset:
Unpacks embedded vectors,
One-hot encodes sentiment,
Label-encodes used type (AIVI vs. HUMAN),
Trains a feed-forward neural network (MLPClassifier) to predict whether a comment was made in response to an AI or human influencer,
Evaluates the model using accuracy, classification report, confusion matrix, and ROC AUC.

`Text_to_Vector.py` -
Loads a JSON file (output_Final.json) containing user comments (likely scraped from Instagram).
Starts processing at a given index (start_index = 150000) in case of a crash or to resume processing.
Uses OpenAI’s `text-embedding-3-large` model to generate embeddings from user comment text (posts.comments.text).
Stores the result in a new column: embedded.posts.comments.text.
Implements retrying failed embedding requests with backoff.
Periodically saves output to JSON files named by index (OutPutUntilXXXX.json), plus a final export.

`graph_analysis.R` - 
Reads subgraph edge lists and builds directed igraph graphs.
Computes multiple centrality metrics (Indegree, Outdegree, Closeness, etc.).
Applies log transformation to normalize skewed centrality distributions.
Performs Wilcoxon tests to compare groups (AIVI vs. HI).
Visualizes the results using ggplot2 boxplots with overlayed mean (blue dots) and median (red triangles).
Annotates significance (p-values) on each facet.

`model.R` - 
Train classification models (Logistic Regression and Neural Networks) to differentiate between user interactions with AI-based Virtual Influencers (AIVIs) and Human Influencers (HIs) based on comment data, metadata, sentiment, and vector embeddings.
Data Preparation & Sentiment Analysis:
Loads Excel/CSV comment data.
Cleans usernames and calculates sentiment using sentimentr.
Computes derived features like comment length and popularity.
Saves enriched data.
Statistical Testing (t-tests and Wilcoxon):
Tests for group differences in popularity, likes per post/comment, text length, and sentiment.
Visualizes these differences with ggplot2 boxplots and overlays mean/error bars.

Logistic Regression with caret:
Trains a logistic regression model on selected features (likes_count, popularity, sentiment).
Evaluates performance via confusion matrix, F1 score, and ROC/AUC.
Exploratory Model with BERT and TF-IDF:
Includes textEmbed for BERT embeddings and tm for TF-IDF vectorization.
Trains logistic regression using comment text.
Makes predictions and evaluates performance.

`df2Neo4j.ipynb` -
Create a Neo4j graph object from comments using LangChain and LlaMA.
Then, create the two subgraphs of AIVI and HI.


## Citing
If you find this paper useful for your research, please consider citing us:
```
@article{j2026AIVI,
  title={Emotional and Behavioral Asymmetries in User Responses to Virtual vs. Human Influencers},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={2026},
  publisher={}
}
```


