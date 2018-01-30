# FacebookLikePrediction
I analyzed the UCI facebook metrics dataset and built 2 models to predict how many likes a post will have based on features in dataset.

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
The Linear Regression Model was tuned for robustness, yet the best R-squared value for test was 0.15, with 0.20 on the training set. 

The Random Forest Model overfit, and I was unable to achieve a psuedo-R-squared value above .13. Correlation metrics (Spearman, Pearson) had similar drops between training score and test score. 

Thus, the Linear Regression model outperformed the Random Forest, yet neither model is close to a adequately performing production model. This suggests that the information in this dataset does not contain enough useful information.

## Source:
https://archive.ics.uci.edu/ml/datasets/Facebook+metrics

(Moro et al., 2016) Moro, S., Rita, P., & Vala, B. (2016). Predicting social media performance metrics and evaluation of the impact on brand building: A data mining approach. Journal of Business Research, 69(9), 3341-3351. 
