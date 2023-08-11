# Introduction: Predicting Airbnb Ratings using Decision Tree Analysis

Welcome to my project, where I dive into the world of data analysis and prediction with a focus on Airbnb ratings. As a student, I've taken on the challenge of exploring how Decision Tree classifiers can help us anticipate whether an Airbnb client will give a rating above or below 90. This endeavor is an essential part of my portfolio, showcasing my skills and passion for data-driven insights.

### What's the Project About?
In this repository, I've chosen to work with Barcelona Airbnb data, examining various factors that might influence client ratings. These factors range from the type of room and its capacity to the cleanliness rating and even a restaurant index. By analyzing this dataset, I aim to create a predictive model using Decision Trees that can help hosts understand guest satisfaction and tailor their offerings accordingly.

### The Decision Tree Approach:
I've taken a step-by-step approach to build and fine-tune the Decision Tree model:
1. **Data Preparation:** I've carefully cleaned and organized the data, ensuring it's ready for analysis.
2. **Feature Selection:** I've chosen the most important features that can affect client ratings, simplifying the model while retaining its effectiveness.
3. **Model Building:** I've created a Decision Tree classifier, a tool that learns patterns in the data to make predictions.
4. **Parameter Optimization:** I've used GridSearchCV to find the best parameters for the model, ensuring optimal performance.
5. **Evaluation:** I've thoroughly assessed the model's accuracy and effectiveness in predicting ratings.

### Comparing Two-Dimensional Data:
As an extra step, I've explored a unique technique. I've transformed the data into a two-dimensional representation using t-SNE, enabling a visual comparison of the original and reduced-dimensional datasets. I've then repeated the Decision Tree process on both datasets to understand how this impacts the model's predictions.

### Why It Matters:
This project isn't just about building a model—it's about gaining insights. As a student, I'm excited to showcase my ability to use data analysis and machine learning to solve real-world problems. Predicting Airbnb ratings has practical implications for hosts and provides a tangible way to understand the power of data-driven decision-making.

# Data 

The Barcelona Airbnb dataset contains various details about properties available for short-term stays in Barcelona. It includes features like price, room type, capacity, cleanliness rating and more. This dataset enables us to explore factors that impact guest experiences and ratings.

## Features:
File - airbnb_data.csv.
"realSum" - total Airbnb listing price (numeric).
"room_type" - type of room offered (private room, shared room, entire home/apt) (categorical).
"person_capacity" - maximum number of accommodated residents (numeric).
"host_is_superhost" - whether the host is a superhost (ordinal 0 or 1).
"multi" - whether multiple rooms are included in the listing (binary 0-1).
"biz" - whether the listing is intended for business purposes (binary 0-1).
"cleanliness_rating" - cleanliness rating (numeric).
"guest_satisfaction_overall" - overall guest satisfaction rating (numeric).
"bedrooms" - number of bedrooms (numeric).
"dist" - distance to the city center (numeric).
"metro_dist" - distance to the metro (numeric).
"attr_index" - attractiveness index (numeric).
"attr_index_norm" - normalized attractiveness indexes.
"rest_index" - restaurant index (numeric).
"rest_index_norm" - normalized restaurant index (numeric).
"lng" - longitude coordinate (numeric).
"lat" - latitude coordinate (numeric).

# Research Objective and Goals

## Research Objective: 
Classify the original data and two-dimensional data.

## Goals:
Choose a variable with two classes for classification.
Examine descriptive statistics for the data and the dependent variable classes.
Perform classification using decision tree algorithm.
Visualize results with ROC curves.
Calculate model statistics.


# Comparison of Data Characteristics: Group 0 (Rating < 90) vs Group 1 (Rating > 90)

In this analysis, we delve into two distinct groups of Airbnb property data based on client ratings in the captivating city of Barcelona. Group 0 comprises properties with client ratings below 90, while Group 1 includes properties that have received ratings exceeding 90. These tables provide a glimpse into the key attributes of each group, allowing us to uncover noteworthy differences between them.

## Group 0 (Rating < 90):
The first table presents statistical summaries of attributes for properties within Group 0. These properties have received client ratings below 90, and the table sheds light on the characteristics associated with such ratings.

|        | realSum | person_capacity | cleanliness_rating | bedrooms | dist | Metro_dist | Attr_index | Rest_index | lng | lat |
| ------ | ------- | --------------- | ------------------ | -------- | ---- | ---------- | ---------- | ---------- | --- | --- |
| Count  | 598.00  | 598.00          | 598.00             | 598.00   | 598.00 | 598.00     | 598.00     | 598.00     | 598.00 | 598.00 |
| Mean   | 311.49  | 2.95            | 8.60               | 1.26     | 2.07 | 0.42       | 469.58     | 883.97     | 2.17 | 41.39 |
| Std    | 464.61  | 1.40            | 1.21               | 0.65     | 1.29 | 0.26       | 264.67     | 454.44     | 0.02 | 0.02 |
| Min    | 69.59   | 2.00            | 2.00               | 0.00     | 0.14 | 0.01       | 93.82      | 159.84     | 2.12 | 41.35 |
| Median | 196.90  | 2.00            | 9.00               | 1.00     | 1.72 | 0.36       | 381.59     | 816.56     | 2.17 | 41.39 |
| Max    | 6943.70 | 6.00            | 10.00              | 6.00     | 8.44 | 1.68       | 2065.07    | 2608.33    | 2.23 | 41.46 |

## Group 1 (Rating > 90):
The second table provides a parallel analysis, focusing on properties in Group 1 with client ratings exceeding 90. By examining the attributes of this group, we aim to uncover the distinctive features associated with higher client ratings.

|        | realSum | Person_capacity | Cleanliness_rating | bedrooms | dist | Metro_dist | Attr_index | Rest_index | lng | lat |
| ------ | ------- | --------------- | ----------------- | -------- | ---- | ---------- | ---------- | ---------- | --- | --- |
| Count  | 957.00  | 957.00          | 957.00            | 957.00   | 957.00 | 957.00     | 957.00     | 957.00     | 957.00 | 957.00 |
| Mean   | 273.96  | 2.63            | 9.72              | 1.19     | 2.15 | 0.45       | 461.12     | 873.72     | 2.17 | 41.39 |
| Std    | 179.83  | 1.18            | 0.50              | 0.51     | 1.39 | 0.28       | 270.66     | 465.72     | 0.02 | 0.02 |
| Min    | 69.59   | 2.00            | 6.00              | 0.00     | 0.12 | 0.01       | 98.66      | 167.93     | 2.11 | 41.35 |
| Median | 215.28  | 2.00            | 10.00             | 1.00     | 1.78 | 0.38       | 390.76     | 792.92     | 2.17 | 41.39 |
| Max    | 1770.66 | 6.00            | 10.00             | 3.00     | 8.06 | 2.40       | 2934.13    | 4542.75    | 2.22 | 41.46 |


# The following metrics are used for evaluating model suitability:

**Accuracy:** This measures the overall correctness of the model's predictions. It calculates the ratio of correctly classified examples to the total number of examples. However, relying solely on accuracy may not be sufficient when there is class imbalance or when the cost of false positives and false negatives differs.

**Precision:** This metric quantitatively evaluates the model's ability to correctly identify positive examples. It calculates the ratio of true positive results to the sum of true positive results and false positive results.

**Recall:** This metric measures the model's ability to find all positive examples. It calculates the ratio of true positive examples to the sum of true positive examples and false negative examples.

**F1 Score:** The F1 score combines the precision metric and recall into a single indicator that balances both metrics. It is the harmonic mean of precision and recall, resulting in a single value that indicates the overall performance of the model.

# Classification Using Decision Tree Algorithm

## What is a Decision Tree Algorithm?

A decision tree is a supervised learning method that can be used for both classification and regression tasks, although it is most commonly chosen for solving classification problems. It is a tree-like classifier where internal nodes represent features of a dataset, branches depict decision rules, and each leaf node corresponds to an outcome.

## Key Hyperparameters

**Max Depth:** Specifies the maximum depth of the decision tree. A large value may lead to overfitting and reduced generalization, while a small value may result in underfitting.

**Min Samples per Leaf:** Sets the minimum number of samples required in a leaf node. A high value may simplify the tree but potentially lead to underfitting, while a low value may do the opposite.

**Min Samples per Split:** Sets the minimum number of samples required to split a node into new nodes.

**Criterion:** Determines the function the decision tree uses to decide when to stop growing the tree.

**Max Features:** Specifies the number of features that can be considered for splitting. A higher value may provide more information to the model.

## Algorithm Stages

In decision trees, to predict the class of a given dataset, the algorithm starts from the root node and compares attribute values with real dataset values. Based on comparisons, it traverses the tree and moves to the next node.

The process continues until a leaf node is reached. The algorithm can be better understood using the following steps:

1. Start with the root node containing the entire dataset.
2. Find the best attribute using an attribute selection mechanism.
3. Divide the dataset into subsets with the best attribute values.
4. Create a decision tree node with the best attribute.
5. Recursively create new decision trees for subsets in step 3. Continue until you can't classify values further, resulting in a leaf node.

## Advantages and Disadvantages

### Advantages:
Decision tree classifiers are fast and efficient, handling large datasets and requiring minimal resources.
They are robust to noise and capable of recognizing unclear and nonlinear relationships between features.

### Disadvantages:
Decision tree classifiers can tend to create overly complex trees that fit training data well but perform poorly on new data.
They may be sensitive to small changes in the dataset, requiring retraining to maintain predictive accuracy.
Scaling features may pose challenges, necessitating multiple trials and tree variations for optimal results.

# Application of the Algorithm in Practice

## Initial Model

To begin, we created two models using both the original and reduced-dimensional training datasets, without altering the classifier's hyperparameters. We calculated the accuracy of the models using the 'sklearn' library's accuracy calculation function. The results showed that our initial models correctly classified 72% and 0.69% of the values, respectively.

## Search for Optimal Hyperparameters

For the search for optimal parameters, we chose one of the most popular algorithms - GridSearchCV (Grid Search Cross-Validation).

Given that the decision tree algorithm allows for modifications of several hyperparameters, we selected five key parameters as described in the previous section.

The hyperparameters provided to the GridSearchCV algorithm were:

**Max Depth**
**Min Samples per Leaf**
**Max Features**
**Criterion**
**Min Samples per Split**

We obtained results where the best hyperparameter values remained the same for both the original dataset and the reduced-dimension dataset.

**GridSearchCV results using original data**
| Max Depth | Min Samples per Leaf | Max Features | Criterion | Min Samples per Split |
|-----------|----------------------|--------------|-----------|----------------------|
| 3         | 1                    | None         | entropy   | 2                    |

**GridSearchCV results using two-dimensional data**
| Max Depth | Min Samples per Leaf | Max Features | Criterion | Min Samples per Split |
|-----------|----------------------|--------------|-----------|----------------------|
| None      | 3                    | log2         | log_loss  | 10                   |

With these hyperparameters, the models showed an improvement of 8% and 1% compared to the initial models that used default parameters. We will discuss the classification quality results along with the evaluation methods for quality.

## Methods for Evaluating Classification Quality

### Cross-Validation Method

To assess the quality of this model, the cross-validation method was employed. This approach divides the dataset into five equal parts, using each part as a testing set while the rest are used as training sets.

Our average cross-validation result using both the model with the original dataset and the model with the two-dimensional dataset is 0.79, indicating that the decision tree classifier with the selected parameters correctly classified approximately 79% of the test dataset.

### Holdout Validation Method

Holdout validation is an alternative method used to assess the performance of the decision tree classifier. When applying this method to the classifier with the selected parameters and the original dataset, we achieved an accuracy of 79% in correctly classifying the test dataset. In contrast, when using the two-dimensional dataset, the accuracy achieved through holdout validation was 70%.

It's important to note that these results provide insight into how well the classifier performs on new, unseen data. The model trained on the original dataset exhibited an accuracy of 79%, while the model using the two-dimensional dataset achieved an accuracy of 70% using holdout validation.

## Confusion Matrix

These results represent the confusion matrix outcomes. The purpose of this matrix is to assess the classifier's performance by comparing combinations of actual and predicted values.

In this case, the model using the original dataset classified 62 instances correctly as class 0, and 184 instances correctly as class 1. However, it made 52 instances where class 0 was predicted when it was actually class 1 (false positives), and 13 instances where class 1 was predicted when it was actually class 0 (false negatives).

![image](https://github.com/Voleisho/Airbnb-Client-Rating-Prediction/assets/141240910/2dd4e8cb-6b2c-4887-8c66-2d3592e1c374)


Similarly, for the model using the two-dimensional dataset, 75 instances were correctly classified as class 0, and 142 instances were correctly classified as class 1. However, there were 39 instances falsely predicted as class 0 (false positives) and 55 instances falsely predicted as class 1 (false negatives).

![image](https://github.com/Voleisho/Airbnb-Client-Rating-Prediction/assets/141240910/61683672-384b-4b7d-b575-5db940b49ba1)

In terms of classifying Class 0, the model using the two-dimensional dataset performed slightly better with 75 true negatives (correct predictions of Class 0) compared to 62 true negatives by the model using the original dataset. However, in terms of classifying Class 1, the model using the original dataset performed better with 184 true positives compared to 142 true positives by the model using the two-dimensional dataset.

Therefore, based on these results, the classifier's performance is not very satisfactory, as there are numerous errors present.

## Classification Metrics

These results represent classification model quality metrics used to assess the accuracy and performance of the model using both original and reduced-dimension data.

According to the "Precision" metric, in 60% of cases where the model predicted a negative event (value 0), the prediction was correct, while in 40% of cases, the predictions were incorrect. Even fewer accurate zeros were predicted using the two-dimension data. However, more accurate ones were predicted compared to using the original data.

Using the original dataset, the "Recall" value indicates that in 46% of cases where the value is truly 0, the model correctly identified it, while in 54% of cases, the predictions were incorrect. The results for the original and reduced-dimension datasets significantly differ in this aspect.

Furthermore, the "F1-score" fluctuates between 52% and 77%, suggesting that the model's performance is moderately satisfactory.

"Accuracy" is an overall metric indicating how many percent of events were correctly classified by the model. In this case, the accuracy is 69%, meaning that the first model correctly classifies only 69% of all events. The second model performs slightly better, with an accuracy of 0.70%.

|   | Precision | Recall | F1-Score | Support | Accuracy |
|---|-----------|--------|----------|---------|----------|
| 0 |   0.60    |  0.46  |   0.52   |   114   | 0.69     |
| 1 |   0.72    |  0.82  |   0.77   |   197   |          |

|           | precision | recall | f1-score | support | Accuracy |
|-----------|-----------|--------|----------|---------|----------|
|       0   |    0.58   |  0.66  |   0.61   |   114   | 0.70     |
|       1   |    0.78   |  0.72  |   0.75   |   197   |          |
 

In summary, the overall quality of the models is not good, as there are many incorrect predictions and the accuracy is moderate.

## ROC Curve

Visualizing the graphical representation, the ROC curve illustrates how classifier predictions change based on the decision threshold (refer to Figure 8). AUC is an evaluation metric that measures how well a classifier can distinguish between positive and negative values. Our AUC result is 0.7016 and 0.6944, when using the classifier for original data and two-dimensional data. This indicates that the classifier has a moderate ability to distinguish between the two classes. The results are satisfactory, but not sufficiently high to consider the models highly successful.

![image](https://github.com/Voleisho/Airbnb-Client-Rating-Prediction/assets/141240910/a302c2b9-01cf-4009-9276-83ffb1a90eaa)

![image](https://github.com/Voleisho/Airbnb-Client-Rating-Prediction/assets/141240910/22b346dc-ebf0-4186-a7f2-cfc15a46f4c1)
