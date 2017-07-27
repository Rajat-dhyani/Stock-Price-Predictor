# Stock Price Predictor
*Udacity - Machine learning Nano Degree Program : Project-6 (Capstone project)*

## Project Overview
*This is sixth and final capstone project in the series of the projects listed in Udacity- Machine Learning Nano Degree Program.*

Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.

Can we actually predict stock prices with machine learning? Investors make educated guesses by analyzing data. They'll read the news, study the company history, industry trends and other lots of data points that go into making a prediction. The prevailing theories is that stock prices are totally random and unpredictable but that raises the question why top firms like Morgan Stanley and Citigroup hire quantitative analysts to build predictive models. We have this idea of a trading floor being filled with adrenaline infuse men with loose ties running around yelling something into a phone but these days they're more likely to see rows of machine learning experts quietly sitting in front of computer screens. In fact about 70% of all orders on Wall Street are now placed by software, we're now living in the age of the algorithm.

This project utilizes Deep Learning models, Long-Short Term Memory (LSTM) Neural Network algorithm, to predict stock prices. For data with timeframes recurrent neural networks (RNNs) come in handy but recent researches have shown that LSTM, networks are the most popular and useful variants of RNNs. 

I have used Keras to build a LSTM to predict stock prices using historical closing price and trading volume and visualize both the predicted price values over time and the optimal parameters for the model.

![Stock Price Predictor](https://github.com/Rajat-dhyani/Stock-Price-Predictor/blob/master/data_visualization_lstm_improved.png)


## Problem Highlights
*The challenge of this project is to accurately predict the future closing value of a given stock across a given period of time in the future.  For this project I have used a Long Short Term Memory networks – usually just called “LSTMs” to predict the closing price of the S&P 500 using a dataset of past prices*

* **Achievements:**
  * Built a model to accurately predict the future closing price of a given stock, using Long Short Term Memory Neural net algorithm. 
  * Achieved Mean Squared Error rating of just 0.00093063.

Things i have learnt by completing this project:
* How to apply deep learning techniques: Long Short Term Memory Neural Network algorithms.
* How to use keras-tensorflow library.
* How to collect and preprocess given data.
* How to analyze model's performance.
* How to optimise Long Short Term Memory Neural Network algortithm, to ensure increase in postive results.

### Other Related Projects:
* <strong> Project 0 : </strong> *[Titanic Survivals Prediction](https://github.com/Rajat-dhyani/titanic_survival)*
* <strong> Project 1 : </strong> *[Boston's Houses Prediction](https://github.com/Rajat-dhyani/boston_housing)*
* <strong> Project 2 : </strong> *[Charity Donors Prediction](https://github.com/Rajat-dhyani/charity_donors)*
* <strong> Project 3 : </strong> *[Creating Customer Segments](https://github.com/Rajat-dhyani/creating_customer_segments)*
* <strong> Project 4 : </strong> *[Smart Cab](https://github.com/Rajat-dhyani/smart-cab)*
* <strong> Project 5 : </strong> *[ImageNetBot](https://github.com/Rajat-dhyani/ImageNetBot)*

## Software and Libraries
This project uses the following software and Python libraries:

* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [NumPy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [Keras](https://keras.io/)
* [Tensor-flow](https://www.tensorflow.org)
* [Jupyter Notebook](http://ipython.org/notebook.html)

