# FacebookLikePrediction
I analyzed the UCI facebook metrics dataset and built 2 models: 
   
    1. Predict how many likes a post will have based on features in dataset
    
    2. Predict how many lifetime users a post will engage
    
There are 2 project notebooks, each following the below structure. The first is the like prediction model, and the second is for the lifetime engaged users model. 

## Project Structure
1. Project Question and Dataset Description
2. Loading and viewing dataset
3. Pre-processing
4. Exploratory Data Analysis
5. Modeling 
    - Outlier removal and re-shaping dataframe
    - Linear Regression
    - Random Forest Regression
6. Takeaways
7. References

## Results

### Like Prediction Model

The Linear Regression Model was tuned for robustness, yet the best R-squared value for test was 0.15, with 0.20 on the training set. 

The Random Forest Model overfit, and I was unable to achieve a R-squared test value above .13. Correlation metrics (Spearman, Pearson) had similar drops between training score and test score. 

The Linear Regression model modestly outperformed the Random Forest model, yet neither could be a suitably performing production model. This suggests that the information in this dataset does not contain enough useful information. 

### Lifetime Engaged Users Model

The Lifetime Engaged Users model included the same variables and modeling process as the Like Prediction Model, but saw the addition of the total interactions feature (likes + comments + shares). Using a linear regression model as a benchmark, the model reached a R^2 value of .683 for the test data vs. .646 in the train data. 

The final tuned Random Forest regression Model performed higher than the benchmarked linear regression, with R^2 values of .72 on the test dataset vs. .75 for the train dataset, with high correlation metrics (test Spearmans Correlation value = .868).

## Conclusion

While the model predicting likes performed slightly better than just guessing,the final model for predicting lifetime engaged users had much higher performance. This model could be used by organizations trying to estimate the number of engaged users per post, in order to estimate lifetime post performance, perhaps as a KPI. 

## Source:
https://archive.ics.uci.edu/ml/datasets/Facebook+metrics

(Moro et al., 2016) Moro, S., Rita, P., & Vala, B. (2016). Predicting social media performance metrics and evaluation of the impact on brand building: A data mining approach. Journal of Business Research, 69(9), 3341-3351. 
