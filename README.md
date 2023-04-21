# SC1015 Project - *The Titanic Disaster*

Welcome to Titanic Analysis repository.
This is a Mini-project for SC1015 - Introduction to Data Science and Artifical Intelligence. <br>
This project focuses on predicting who survives The Titanic based on passenger data.

## Contributers:
1.  @jdengoh: Data pre-processing, Data Splitting and Feature Analysis, KNN, XGBoost, Conclusion
2.  [@ananyakapoor12](https://github.com/ananyakapoor12): Data pre-processing, Data Visualisation and EDA, Random Forest Classifier, Conclusion

## About/Problem Definition

Our team's objective is to analyse and predict the likely survivors of the titanic disasters using passengers' data.

The underlying motivation is provide new insights on the titanic incident by:
- Identifying factors that may influence passengers' survivor rate.
- Predicting the likelihood of an individual's survival

Such insights may be able to tell us more about different demographics' chances of survival and factors that affect their survival rate.

We hope that such insights can be useful in the future for:
- Identifying undiscovered loopholes in safety measures or precautions.
- Further improving existing safety infrastructure to prevent such a disaster from occuring in the future.

## Datasets
https://www.kaggle.com/competitions/titanic/data

## Repository Overview
1. [Data Cleaning](https://github.com/jdengoh/Titanic-Analysis/blob/main/Data%20Cleaning.ipynb)
2. [Data Visualisation and EDA](https://github.com/jdengoh/Titanic-Analysis/blob/main/Data%20Visualisation%20and%20EDA.ipynb)
3. [Data Splitting and Feature Analysis](https://github.com/jdengoh/Titanic-Analysis/blob/main/Data%20Splitting%20and%20Feature%20Analysis.ipynb)
4. [Prediction Models](https://github.com/jdengoh/Titanic-Analysis/blob/main/Prediction%20Models.ipynb)

## Prediction models used
1. KNN
2. XGBoost
3. Random Forest Classifier

## Conclusion

## Learning Points
Our key learnings from this project are as follows:

1. Data pre processing and cleaning to prepare it for model fitting.
2. Feature analysis on data.
3. Encoding of data when needed.
4. New prediction models - K-Nearest Neighbour, XGBoost, Random Forest Classifier.
5. Using best features for predictions in different models, cross-validation.
6. Overall pipeline of machine learning: from getting data to making our predictions.

## References

##### KNN
- https://towardsdatascience.com/k-nearest-neighbors-and-the-curse-of-dimensionality-e39d10a6105d#:~:text=The%20%E2%80%9CCurse%20of%20Dimensionality%E2%80%9D%20is,to%20keep%20the%20same%20density.
- https://medium.com/swlh/stop-one-hot-encoding-your-categorical-features-avoid-curse-of-dimensionality-16743c32cea4#:~:text=One%2Dhot%20encoding%20categorical%20variables,problem%20of%20parallelism%20and%20multicollinearity.
- https://neptune.ai/blog/knn-algorithm-explanation-opportunities-limitations

##### Random Forest Classifier
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://crunchingthedata.com/random-forest-overfitting/#:~:text=Overfitting%20happens%20when%20a%20model,not%20generalize%20to%20other%20observations.

##### XGBoost
- https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/
https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d

##### Chi-Squared test
- https://datascience.stackexchange.com/questions/68931/scikit-learn-onehotencoder-effect-on-feature-selection
- https://github.com/aswintechguy/Data-Science-Concepts/blob/main/Machine%20Learning/
- https://www.analyticsvidhya.com/blog/2021/06/decoding-the-chi-square-test%E2%80%8A-%E2%80%8Ause-along-with-implementation-and-visualization/
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html

##### EDA
- https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/
- https://seaborn.pydata.org/
