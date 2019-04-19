### Predicting income using Linear model with Estimators
This model uses the tf.estimator API in TensorFlow to solve a benchmark binary classification problem.

##### Overview
Using census data which contains data a person's age, education, marital status, and occupation (the features), we will try to predict whether or not the person earns more than 50,000 dollars a year (the target label). We will train a logistic regression model that, given an individual's information, outputs a number between 0 and 1—this can be interpreted as the probability that the individual has an annual income of over 50,000 dollars.
