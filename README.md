README

This repository contains a Python script airbnb_analysis.py that performs some basic data analysis on an Airbnb listings dataset. The analysis aims to answer the following questions:

What is the average price of listings by neighborhood?
What is the most common room type and its percentage in the dataset?
Is there a relationship between the number of reviews and the availability of listings?
Prerequisites
Python 3.6 or later
pandas, pandas-profiling, ydata-profiling, and sklearn Python packages
Installation
To run this script, you need to install the required Python packages. You can install them using the following command:

Copy code
pip install pandas pandas-profiling ydata-profiling scikit-learn
Usage
To run the script, navigate to the directory where the script is located and run the following command:

Copy code
python airbnb_analysis.py
The script will read the abnbuklistings.csv file and perform the analysis. The results will be printed to the console and some visualizations will be displayed.

Code explanation
The Python script starts by importing the necessary packages, including pandas, pandas-profiling, ydata-profiling, and sklearn.

It then reads in the Airbnb listings dataset from a CSV file using pandas.

Next, it creates a profile report of the dataset using the ProfileReport function from the pandas-profiling package. This report provides an overview of the dataset's variables, including data types, missing values, and basic statistics.

The script then drops the neighbourhood_group and license columns since they have no values.

The last_review column is converted to a datetime type using the pd.to_datetime function.

Next, the script checks for missing values in the last_review and reviews_per_month columns. It uses the SimpleImputer class from sklearn to fill in missing values with the mean value of the column.

The script then answers the three questions listed above using pandas functions such as groupby, value_counts, and visualizations from seaborn and matplotlib.

Conclusion
This Python script provides some basic insights into the Airbnb listings dataset. However, more in-depth analysis can be performed on this dataset to gain a better understanding of the factors that affect the prices and availability of Airbnb listings.