# Project Description

Title: Predicting Market Reactions to Insider Trading
Duration: August 2024 – December 2024

In this project, we aimed to forecast short-term market reactions based on insider trading events by leveraging a combination of insider trading reports, historical market trends, and macroeconomic factors. We built a predictive model using a Long Short-Term Memory (LSTM) neural network, known for its ability to capture time series dependencies and sequential patterns.

We collected and integrated data from three main sources:

Insider Trading Data: SEC EDGAR Form 4 filings, which disclose trades by company insiders.
Market Data: Historical stock prices sourced from Yahoo Finance.
Macroeconomic Data: Key indicators such as interest rates, GDP growth, and inflation from FRED (Federal Reserve Economic Data).
Using data visualization techniques, we explored relationships and trends, uncovering patterns like clusters of insider buying preceding positive stock movement or insider selling ahead of downturns.


# Technical Details:

Data Preprocessing:

Merged multi-source datasets based on timestamps.
Handled missing data with forward-fill and interpolation methods.
Created lag features (e.g., previous day's closing price, moving averages).
Standardized macroeconomic indicators to align with stock movement scales.

Model Architecture:

Input: Sequences combining stock prices, insider trading events (e.g., buy/sell signals, volume), and macroeconomic indicators.

Model:
LSTM layers (2 stacked layers, each with 64 units).
Dropout layers (0.2 rate) to reduce overfitting.
Dense output layer with a linear activation function for regression (predicting stock movement percentage).
Loss Function: Mean Squared Error (MSE).
Optimizer: Adam optimizer (adaptive learning rate).
Training:
Early stopping was used to avoid overfitting.
Data was split into train, validation, and test sets (70/15/15%).
Model Optimization Techniques Applied:
Feature Engineering Enhancements:
Event Flags: Explicit encoding of 'significant' insider transactions (large buys/sells).
Technical Indicators: Added moving averages (SMA, EMA), RSI, and MACD as input features.
Sentiment Signals: (Optional enhancement) Used NLP to analyze CEO statements when possible.
Hyperparameter Tuning:
Grid search on:
Number of LSTM units (32, 64, 128).
Number of layers (1 vs 2).
Learning rates (0.001, 0.0005).
Batch sizes (32, 64).
Selected optimal settings based on validation loss.
Regularization:
Dropout layers inserted between LSTM layers to reduce overfitting.
L2 weight regularization on LSTM kernels.
Data Balancing:
Since large insider buys/sells are rarer, we up-sampled major insider events to prevent bias toward normal stock movement.
Window Size Tuning:
Experimented with different input sequence lengths (30 days, 60 days, 90 days) to capture the appropriate historical context.
Found that a 60-day window worked best for balancing performance and computational cost.
Learning Rate Schedulers:
Used a ReduceLROnPlateau callback to decrease the learning rate if validation loss plateaued.
Ensembling Final Predictions:
Combined predictions from models trained on slightly different feature sets (e.g., technical indicators + insider events only vs. full feature set) to improve robustness.






# Stock Prediction using Insider Data

Data Sources (the processed data for companies can be found in the 'data/companies' folder in the repo):
- yfinance: https://pypi.org/project/yfinance/ This python package was used to download the daily market data for each company
- US Security and Exchange Commission: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets This site was used to get quarterly data on inside trading

The project currently has the following python scripts:
- InsiderData.py: Cleans and processes insider data for a company
- StockData.py: Downloads stock data for a company
- DownloadAllData.py: Uses the above two files to form a pipeline to prepare data for each company
- Models: PyTorch model architectures of the two approaches used
- PreprocessingDataUtils: Utility functions to train data using models defined
- TrainModels: Trains the two models for each company whose data has been preprocessed
