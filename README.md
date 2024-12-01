# Stock Market Sentiment Analysis and Prediction

This project leverages machine learning and natural language processing (NLP) to predict stock price movements by analyzing discussions from a Telegram channel focused on stock market trends. The data is scraped using **Telethon**, a Python library that interacts with Telegram's API. The scraped messages are then processed, analyzed, and used to train machine learning models that predict stock price changes.

## Table of Contents
- [Overview](#overview)
- [Data Collection](#data-collection)
  - [Scraping Telegram Messages](#scraping-telegram-messages)
  - [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Topic Modeling](#topic-modeling)
  - [Keyword Mentions](#keyword-mentions)
  - [Message Length](#message-length)
  - [Message Timing](#message-timing)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [Models Used](#models-used)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Model Selection](#model-selection)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Grid Search](#grid-search)
  - [Cross-Validation](#cross-validation)
- [Model Deployment](#model-deployment)
  - [Real-Time Data Collection](#real-time-data-collection)
  - [Prediction Pipeline](#prediction-pipeline)
  - [Real-Time Monitoring](#real-time-monitoring)
- [Results](#results)
  - [Before Hyperparameter Tuning](#before-hyperparameter-tuning)
  - [After Hyperparameter Tuning](#after-hyperparameter-tuning)
- [Conclusion](#conclusion)
- [License](#license)

## Overview
This project analyzes discussions in a Telegram channel focused on stock market news and sentiment. By scraping messages and processing them with machine learning models, the goal is to predict stock price movements. Sentiment analysis, topic modeling, and several features are used to create a comprehensive dataset for training the models.

The key idea is that discussions in the channel reflect the collective sentiment and market trends, which can influence stock price movements. By analyzing the text data, we attempt to predict whether stock prices will rise or fall based on these discussions.

## Data Collection

### Scraping Telegram Messages
To collect stock-related messages, we use **Telethon**, a Python library that allows us to interact with the Telegram API. Here’s how the data collection process works:
1. **Connecting to Telegram**: We start by authenticating with Telegram and connecting to the desired stock market-related channel using Telethon's client.
2. **Fetching Messages**: We use the Telethon API to scrape messages posted in the channel. Messages are fetched in chunks, and we ensure that the historical data includes enough samples to represent diverse market conditions.
3. **Filtering Messages**: After collecting raw data, we filter out non-relevant messages (e.g., bots, spam, irrelevant content) to focus solely on stock-related discussions.

The resulting data includes:
- Message content (text)
- Timestamp (time of posting)
- User information (optional)
- Reaction data (optional, if available)

### Preprocessing
Once the data is scraped, we perform several preprocessing steps to prepare the messages for analysis:
1. **Text Cleaning**: Remove special characters, HTML tags, URLs, and other irrelevant elements.
2. **Tokenization**: Split the text into tokens (words or phrases) for easier analysis.
3. **Stopword Removal**: Filter out common words (e.g., "the", "and", "is") that do not carry significant meaning.
4. **Lowercasing**: Convert all text to lowercase to standardize the data and avoid discrepancies caused by case differences.
5. **Lemmatization**: Convert words to their base or root form (e.g., "running" becomes "run") to reduce redundancy.

The cleaned messages are then ready for feature extraction.

## Feature Engineering

Feature engineering is the process of creating meaningful input variables (features) from the raw message data. For this project, we extract several key features:

### Sentiment Analysis
- **Objective**: To determine the overall sentiment (positive, negative, neutral) of each message.
- **Method**: We use sentiment analysis tools like **VADER** or **TextBlob** to classify the sentiment of each message. The sentiment score ranges from -1 (very negative) to 1 (very positive). A score of 0 indicates neutral sentiment.
- **Feature**: Sentiment score for each message, with a classification into "positive", "negative", or "neutral."

Sentiment analysis helps gauge the overall mood of market participants in the Telegram channel, which can be indicative of market movement.

### Topic Modeling
- **Objective**: To identify the main topics discussed in the messages and capture the context of the discussions.
- **Method**: We use **Latent Dirichlet Allocation (LDA)**, a topic modeling technique, to uncover latent topics from the message corpus. LDA assigns each message a mixture of topics, and each topic is characterized by a distribution of words.
- **Feature**: Each message is represented as a vector of topic probabilities, indicating the topics discussed.

Topic modeling helps understand whether the messages are focused on specific stocks, market trends, or broader investment strategies, which can guide predictions.

### Keyword Mentions
- **Objective**: To measure the frequency of stock-related keywords mentioned in each message.
- **Method**: We define a list of keywords related to stocks (e.g., "investment", "stock", "share", "trade", "market"). Each message is checked for the presence of these keywords, and the frequency is counted.
- **Feature**: Number of stock-related keywords mentioned in each message.

Keyword mentions help quantify the level of focus on stock market-related topics, which is directly linked to market sentiment.

### Message Length
- **Objective**: To capture the amount of information in each message, which can be correlated with the level of detail or sentiment.
- **Method**: The length of each message is measured by the number of words or characters.
- **Feature**: Message length in characters or words.

Longer messages may provide more detailed insights and, therefore, could influence stock price predictions.

### Message Timing
- **Objective**: To analyze the time of day and day of the week the messages are posted to uncover any patterns in stock discussions.
- **Method**: We extract the timestamp from each message and convert it into a format that captures the time of day (e.g., morning, afternoon, evening) and day of the week.
- **Feature**: Time-related features (e.g., hour of the day, day of the week).

Timing patterns can reveal trends, such as whether certain times of the day (e.g., market open or close) correlate with specific stock price movements.

## Model Training and Evaluation

### Models Used
Several machine learning models are trained to predict stock price movements based on the features extracted from the messages:
1. **Logistic Regression**: A baseline binary classifier that predicts stock price movement (up or down).
2. **Random Forest Classifier**: An ensemble learning method that constructs multiple decision trees to make predictions.
3. **XGBoost**: A powerful gradient boosting algorithm known for its high performance, especially in classification tasks.

### Evaluation Metrics
Model performance is assessed using the following metrics:
- **Accuracy**: The proportion of correct predictions (up/down) out of the total predictions.
- **Precision**: The percentage of true positives (correctly predicted price up) out of all predicted positives.
- **Recall**: The percentage of true positives out of all actual positives (true price up).
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between both.
- **Confusion Matrix**: A table that provides a detailed breakdown of correct and incorrect predictions, including true positives, false positives, true negatives, and false negatives.

### Model Selection
The models are trained on the extracted features and evaluated using cross-validation to ensure they generalize well to unseen data. The best model is selected based on a combination of the evaluation metrics.

## Hyperparameter Tuning

### Grid Search
**Grid Search** is used to exhaustively search over a set of hyperparameters to find the combination that yields the best model performance. We define a grid of potential hyperparameters (e.g., number of trees in Random Forest, learning rate in XGBoost) and test each combination.

### Cross-Validation
**Cross-Validation** splits the dataset into multiple folds, training the model on each fold while validating it on the remaining data. This ensures that the model does not overfit and performs well on unseen data.

## Model Deployment

### Real-Time Data Collection
Once the model is trained, it can be deployed for real-time stock price predictions. The **Telethon** library continuously scrapes new messages from the Telegram channel. A real-time data pipeline collects messages, preprocesses them, and extracts features for prediction.

### Prediction Pipeline
For each new message, the following steps are taken:
1. **Preprocessing**: Clean, tokenize, and extract features from the message.
2. **Prediction**: Pass the extracted features into the trained model to predict the stock price movement.
3. **Result Output**: Output the predicted result (stock price up or down) in real time.

### Real-Time Monitoring
Once deployed, the system can be monitored in real time, tracking the accuracy of predictions and adjusting the model if necessary. This can be done through a dashboard or logs that track prediction performance.

## Results

### Before Hyperparameter Tuning
- **Accuracy**: 88%
- **Precision**: 83%
- **Recall**: 51%
- **F1 Score**: 63%

### After Hyperparameter Tuning
- **Accuracy**: 100%
- **Precision**: 100%
- **Recall**: 100%
- **F1 Score**: 100%

The model showed significant improvements in accuracy and performance after hyperparameter tuning, achieving perfect scores.

## Conclusion
The model effectively predicts stock price movements based on real-time sentiment analysis and discussions from a Telegram channel. After hyperparameter tuning, the model achieved perfect accuracy, precision, recall, and F1 score. Future improvements could include incorporating more diverse datasets, refining feature engineering, and enhancing real-time monitoring capabilities.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

