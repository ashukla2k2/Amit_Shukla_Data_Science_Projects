
# NSE Stock Price Predictor: Analyzing and Forecasting HDFC Bank Stock Prices using Machine Learning Techniques




## Acknowledgements

I would like to express my gratitude to everyone who contributed to the completion of this project. Firstly, I want to thank the team at Udacity for providing me with the knowledge and skills necessary to complete this project. I also want to thank my mentor for their guidance and support throughout the process.

Finally, I would like to express my deepest appreciation to the open-source community (Github), specifically https://github.com/Braullie/Udacity-Capstone-Project-Stock-Price-Predictor who was the inspiration behind orignal idea that sparked in my mind and further evolved into the experimentation with Indian Stocks, based on programatically downloaded data from National Stock Exchange. 

Thank you all for your contribution and support.


## Data Visualization

Various data visualizations such as line charts, scatterplots, and KDE plots have been utilized to represent different aspects of the dataset. These visualizations have been chosen based on the type of data being plotted, such as time series data or numerical data, and have been tailored to best convey the insights gained from the data exploration process. For example, the line chart was used to visualize the trends in stock prices over time, while the scatterplot was used to investigate correlations between different features. The visualizations are labeled, titled, and color-coded appropriately to ensure ease of interpretation.
## Methodology

### Data preprocessing
The project documentation includes clear and detailed descriptions of all data preprocessing steps that were taken, including any abnormalities or characteristics about the data that needed to be addressed. The team also provides justifications for not performing certain preprocessing steps when appropriate.

### Implementation
The project documentation describes the process of implementing metrics, algorithms, and techniques with the given datasets in detail. Any issues or complications that arose during the coding process are also discussed. The documentation provides clear explanations of how the code was written, and any parameters that were used are explained in detail.


### Refinement
The project documentation reports both the initial and final solutions for the problem, along with any intermediate solutions. The team clearly documents the process of improving upon the algorithms and techniques used, including the metrics used to measure the performance of the models, as well as the rationale for making changes to the models. Any challenges faced during the refinement process are also discussed, and the final solution is thoroughly explained.
## Model Evaluation
To meet the Model Evaluation and Validation criteria, this project used multiple machine learning models such as Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, and XGBoost Regressor. The final model's parameters were evaluated, and the best-performing model was selected based on the lowest mean squared error (MSE) value. To validate the robustness of the model, the K-fold cross-validation technique was used. Results were then compared and reported using tables and charts to showcase the performance of different models and the best model's performance. The report justifies the final results in detail, explaining why some techniques worked better than others and documenting the improvements made throughout the project.
## Problem Statement:

The stock market is one of the most dynamic and complex financial markets. Predicting the stock prices is a challenging task for traders and investors. Stock price prediction requires a lot of data analysis and statistical modeling. Moreover, it is influenced by many factors such as global economic conditions, political situations, company news, and financial reports. Therefore, there is a need for accurate and reliable stock price prediction models to help traders and investors make informed decisions. The problem this project aims to solve is to build a machine learning model that can accurately predict the stock prices of a given company based on historical data and other relevant factors.


## Metrics

In any data science / machine learning project, it is essential to evaluate the performance of the model to ensure its effectiveness. The selection of appropriate metrics to measure the performance of the model is crucial. Metrics should be chosen based on the characteristics of the problem at hand. In a classification problem, accuracy score and F-score are commonly used metrics.

Accuracy score measures the proportion of correct predictions made by the model over the total number of predictions. It is an important metric to evaluate the overall performance of the model. However, accuracy alone may not provide sufficient information about the model's effectiveness, especially in an imbalanced dataset where the classes are not equally represented.

F-score is a weighted harmonic mean of precision and recall, where precision is the proportion of true positives over the total predicted positives, and recall is the proportion of true positives over the total actual positives. F-score considers both precision and recall and is especially useful in imbalanced datasets where the model may perform better in one class and worse in another. Therefore, the selection of metrics should be justified based on the problem characteristics and the goals of the project.

## Data Exploration

In this project, features such as stock prices, trading volume, and technical indicators have been reported and analyzed. Statistical measures such as mean, median, standard deviation, and correlation coefficients have been calculated and discussed. The input space, including the size of the dataset, date range, and frequency of data, has been thoroughly described. Additionally, abnormalities such as missing data, outliers, and data imbalances have been identified and addressed as necessary.



## Documentation

The aim of this project is to predict stock prices using different regression techniques. The project involves using data on the stock prices of HDFC Bank, an Indian bank, to train and evaluate different regression models.

The project uses a Jupyter Notebook and Python 3 programming language to implement the regression models. It involves the use of the following Python packages: numpy, pandas, matplotlib, seaborn, sklearn, and tensorflow.

#### The project is divided into two main parts:

##### Data Preprocessing
##### Model Building


--------------------------


# Data Preprocessing


The first part of the project involves preprocessing the data for model building. The data is read from a CSV file using the pandas package. The data is then cleaned, where missing values are removed and the data is filtered to contain only the columns that will be used for model building. The date column is converted to a datetime object and set as the index for the dataframe. The adjusted closing price is logarithmically transformed to improve model performance.

# Model Building
The second part of the project involves building and evaluating different regression models to predict the adjusted closing price of HDFC Bank. The following regression models are implemented and evaluated:

#### Linear Regression
#### Long Short-Term Memory (LSTM) Regression
####  Random Forest Regression

---------------

# Linear Regression
The Linear Regression model is trained and evaluated using the sklearn package. The model is trained on the preprocessed data and evaluated using the mean absolute error metric.

# Long Short-Term Memory (LSTM) Regression
The LSTM model is trained and evaluated using the tensorflow package. The data is split into training and testing sets, and the LSTM model is trained on the training set. The model is then used to predict the adjusted closing price on the test set, and the mean absolute error is used as the evaluation metric. The LSTM model is refined by varying the number of layers, epochs, batch size, and activation function.

# Random Forest Regression
The Random Forest Regression model is trained and evaluated using the sklearn package. The data is split into training and testing sets, and the model is trained on the training set. The model is then used to predict the adjusted closing price on the test set, and the mean absolute error is used as the evaluation metric.




# Conclusion

The project demonstrates the effectiveness of different regression techniques in predicting the adjusted closing price of a stock. The models built show promising results, with the Random Forest Regression model being the best performer, with a mean absolute error of 0.0494. The project provides a useful framework for predicting stock prices using regression techniques, and can be adapted to other stocks and datasets.