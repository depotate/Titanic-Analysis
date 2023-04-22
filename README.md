<img width="1000" alt="image" src="https://user-images.githubusercontent.com/114278389/233688759-b8f0b5ff-d2d6-48c7-ab48-fad52b97ef7d.png">

# SC1015 Project - *The Titanic Disaster*

Welcome to Titanic Analysis repository.
This is a Mini-project for SC1015 - Introduction to Data Science and Artifical Intelligence. <br>
This project focuses on predicting who survives The Titanic based on passenger data.

## Contributers:
1.  [@jdengoh](https://github.com/jdengoh): Data pre-processing, Data Splitting and Feature Analysis, KNN, XGBoost, Conclusion
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

### Relating back to our problem definition and real world context:

- Children and Women having a higher chance of survival could be attributed to passengers aboard prioritising the saving of woman and children first.

- In reality, while ticket class should not have much impact on one's survival rate, it is interesting to see that first class passengers have a higher survival rate.
    - This may be linked to their cabin location and accessibility to safety infrastructure during the time of the disaster.
    - However, more data will be required to be able to confidently explain how ticket class affects survival rate directly.

- Passengers who travel with smaller families seem to have a higher chance in surviving.
    - Perhaps those who have many family members aboard are unwilling to leave them behind since it is unlikely that everyone in their family could be saved.
    - For those travelling alone, it could be possible that they may prioritise saving smaller families.
    - Another important factor is that, most large families were travelling in third class which was also another reason, they had a lower survival rate.
 - First class passengers were more likely to survive than second or third class ones. Since women were more likely to survive than men, almost all female passengers in first class survived.
    
### From our prediction models, we can conclude that:

1. Using KNN model, our best model was using 8 best features likely due to the Curse of Dimensionality. In this attempt, our False Negatives were quite lower than our other attempts and False Positives were manageable too.

2. Using XGBoost, our best model was using all features for prediction likely because it is a robust model in itself. While our first attempt has a high accuracy, it also has relatively higher number of False Negatives compared to Attempt 3 and Attempt 4 with best 10 and 8 features, respectively. 

3. Using Random Forest Classifier, our attempts with 10 and 8 best features yielded the best result and the issue here is likely to be that of overfitting. False positives in both these attempts were much higher than false negatives, a general trend observed in all our models.

4. Overall, from the 4 attempts on each of our three models, the highest test accuracy of 0.83708 was yielded by our **XGBoost Model using all features for prediction**. This is likely becaise XGBoost is a robust model with built-in methods of improving errors, reducing over-fitting and increasing accuracy.


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
- https://www.simplilearn.com/tutorials/machine-learning-tutorial/knn-in-python
- https://realpython.com/knn-python/#tune-and-optimize-knn-in-python-using-scikit-learn

##### Random Forest Classifier
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://crunchingthedata.com/random-forest-overfitting/#:~:text=Overfitting%20happens%20when%20a%20model,not%20generalize%20to%20other%20observations.

##### XGBoost
- https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/
- https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d

##### Chi-Squared test
- https://datascience.stackexchange.com/questions/68931/scikit-learn-onehotencoder-effect-on-feature-selection
- https://github.com/aswintechguy/Data-Science-Concepts/blob/main/Machine%20Learning/
- https://www.analyticsvidhya.com/blog/2021/06/decoding-the-chi-square-test%E2%80%8A-%E2%80%8Ause-along-with-implementation-and-visualization/
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html

##### EDA
- https://www.analyticsvidhya.com/blog/2021/05/feature-engineering-how-to-detect-and-remove-outliers-with-python-code/
- https://seaborn.pydata.org/

##### Notebooks
- https://www.kaggle.com/code/allohvk/titanic-missing-age-imputation-tutorial-advanced
- https://www.kaggle.com/code/jirakst/titanic-auc-92/notebook
- https://www.kaggle.com/code/nhlr21/complete-titanic-tutorial-with-ml-nn-ensembling/notebook
- https://www.kaggle.com/code/dantefilu/keras-neural-network-a-hitchhiker-s-guide-to-nn/notebook#Appendix
