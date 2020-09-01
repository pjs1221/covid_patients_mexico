# COVID-19 Mortality Predictor: Project Overview

* Created a tool that predicts people at risk of mortality from COVID-19 using health data and preconditions released by the Mexican Government
* Cleaned over 500,000 entries for use in machine learning algorithms
* Optimized Linear, Lasso, and Random Forest Regressors using GridSearchCV to reach the best model
* Built a client facing API using flask



## Code and Resources Used
<strong>Python Version</strong>: 3.7

<strong>Packages</strong>: pandas, numpy, sklearn, matplotlib, seaborn

<strong>Kaggle Dataset</strong>: https://www.kaggle.com/tanmoyx/covid19-patient-precondition-dataset

<strong>Flask Productionization</strong>: 

## Data Cleaning
After acquiring the data, I needed to clean it so that it was usable for the model. I made the following changes:

* Removed rows without clear indicators for answer on preconditions
* Removed columns based off of post-hospital factors (e.g. Intubation, ICU, etc.)
* Parsed symptoms and admit dates
* Transformed admit dates to number of days from first symptoms to entry to hospital
* Converted entries to make them model-usable (e.g. Changing 2 to 0 for false entries)

## Exploratory Data Analysis

## Model Building

## Model Performance

## Productionization
