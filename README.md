# COVID-19 Mortality Predictor: Project Overview

* Created a tool that predicts people at risk of death from COVID-19 using health data and preconditions released by the Mexican Government
* Cleaned over 500,000 entries for use in machine learning algorithms
* Optimized Logisitic Regression, Decision Trees, Random Forest, Gradient Boosted Classifiers, and XGboost using GridSearchCV to reach the best model
* Performed analysis on the null accuracy and created different metrics for evaluating models
* Built a client facing API using flask

## Code and Resources Used
<strong>Python Version</strong>: 3.7

<strong>Packages</strong>: pandas, numpy, scikit-learn, matplotlib, seaborn, flask, json, pickle

<strong>For Web Framework Requirements</strong>: pip install -r requirements.txt

<strong>Kaggle Dataset</strong>: https://www.kaggle.com/tanmoyx/covid19-patient-precondition-dataset

<strong>Flask Productionization</strong>: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Data Cleaning

After acquiring the data, I needed to clean it so that it was usable for the model. I made the following changes:

* Removed rows without clear indicators for answer on preconditions
* Removed columns based off of post-hospital factors (e.g. Intubation, ICU, etc.)
* Parsed symptoms and admit dates
* Transformed admit dates to number of days from first symptoms to entry to hospital
* Converted entries to make them model-usable (e.g. Changing 2 to 0 for false entries)

## Exploratory Data Analysis

## Model Building

A train-test split was performed on the dataset with a test size of 20%. Furthermore, k-fold cross validation was utilized as a means of estimating the in-sample accuracy with k = 10. 

At this stage, performance was evaluated by a simple accuracy metric.

Five machine learning algorithms were considered for this data including:

* Logistic Regression - Baseline for the model
* Decision Trees - Utilized due the nature of the data
* Random Forest - Ensemble method for decision trees
* Gradient Boosted Classifier - Ensemble method more optimized for performance
* XGBoost - Similar reasons to GBC model

Gradient boosted classifier and XGboost were found to be the best models. I used GridSearchCV to tune the hyperparameters of the model.

## Model Performance

### Accuracy As A Metric

At first, I used accuracy as the metric with which to analyze the success of the models. All models performed similarly with decision trees and random forest performing slightly worse than the other models. 

* Logistic Regression: Accuracy = 93.16%
* Decision Trees: Accuracy = 92.15%
* Random Forest: Accuracy = 92.44%
* Gradient Boosted Classifier: Accuracy = 93.30%
* XGBoost: Accuracy = 93.31%

Unfortunately, the accuracy alone is not a sufficient metric for the evaluation of the models.


### Null Accuracy and Utilizing Different Thresholds

To put it shortly, the accuracy of the classifiers must be evaluated based off the null accuracy. All models perform little better than a model simply choosing everyone to live. To further highlight this problem, XGboost, the best performing model, only predicted 13% of those who would die from COVID-19. 

The sensitivity of the model at this stage was not high enough.

My solution was to change the threshold with which the model determined whether a patient would die or not. By lowering the threshold, the model would begin to predict that more people would die and, in return, identify those truly at risk.

However, this would also increase the number of those falsely predicted to die. The specificity would be reduced in return for the sensitivity to increase.

To balance the tradeoff between sensitivity and specificity, I tested a range of thresholds and maximized the product of sensitivity and specificity, where:

* Sensitivity = True Positives / Total Positives
* Specificity = True Negatives / Total Negatives

A range of values from 0 to 0.5 with interval 0.001 were tested for the threshold.

### Performance After Changing Threshold

From testing different thresholds, the gradient boosted classifier and XGboost were found to have performed the best.

The accuracy was reduced to approximately 83%, but the sensitivity of the model rose from 13% to 88%.

While this increases the number of false positives, the usefulness of the model skyrockets.

## Productionization

In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values for patient data and returns a prediction if the patient will die from COVID-19.
