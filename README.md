4# Introduction: Predicting Airbnb Ratings using Decision Tree Analysis

In this project, I'm exploring data analysis and prediction, focusing on Airbnb ratings. As a student, I'm looking into how Decision Tree classifiers can help predict if an Airbnb client will rate their experience above or below 90. This project is an important example of my skills and understanding how classifiers operate.

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
- File - airbnb_data.csv.
- "realSum" - total Airbnb listing price (numeric).
- "room_type" - type of room offered (private room, shared room, entire home/apt) (categorical).
- "person_capacity" - maximum number of accommodated residents (numeric).
- "host_is_superhost" - whether the host is a superhost (ordinal 0 or 1).
- "multi" - whether multiple rooms are included in the listing (binary 0-1).
- "biz" - whether the listing is intended for business purposes (binary 0-1).
- "cleanliness_rating" - cleanliness rating (numeric).
- "guest_satisfaction_overall" - overall guest satisfaction rating (numeric).
- "bedrooms" - number of bedrooms (numeric).
- "dist" - distance to the city center (numeric).
- "metro_dist" - distance to the metro (numeric).
- "attr_index" - attractiveness index (numeric).
- "attr_index_norm" - normalized attractiveness indexes.
- "rest_index" - restaurant index (numeric).
- "rest_index_norm" - normalized restaurant index (numeric).
- "lng" - longitude coordinate (numeric).
- "lat" - latitude coordinate (numeric).

# Research Objective and Goals

## Research Objective: 
Classify the original data and two-dimensional data.

## Goals:
- Choose a variable with two classes for classification.
- Examine descriptive statistics for the data and the dependent variable classes.
- Perform classification using decision tree algorithm.
- Visualize results with ROC curves.
- Calculate model statistics.


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

 - **Max Depth:** Specifies the maximum depth of the decision tree. A large value may lead to overfitting and reduced generalization, while a small value may result in underfitting.

- **Min Samples per Leaf:** Sets the minimum number of samples required in a leaf node. A high value may simplify the tree but potentially lead to underfitting, while a low value may do the opposite.

- **Min Samples per Split:** Sets the minimum number of samples required to split a node into new nodes.

- **Criterion:** Determines the function the decision tree uses to decide when to stop growing the tree.

- **Max Features:** Specifies the number of features that can be considered for splitting. A higher value may provide more information to the model.

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
- Decision tree classifiers are fast and efficient, handling large datasets and requiring minimal resources.
- They are robust to noise and capable of recognizing unclear and nonlinear relationships between features.

### Disadvantages:
- Decision tree classifiers can tend to create overly complex trees that fit training data well but perform poorly on new data.
- They may be sensitive to small changes in the dataset, requiring retraining to maintain predictive accuracy.
- Scaling features may pose challenges, necessitating multiple trials and tree variations for optimal results.

# Application of the Algorithm in Practice

## Initial Model

To begin, we created two models using both the original and reduced-dimensional training datasets, without altering the classifier's hyperparameters. We calculated the accuracy of the models using the 'sklearn' library's accuracy calculation function. The results showed that our initial models correctly classified 72% and 64% of the values, respectively.

## Search for Optimal Hyperparameters

For the search for optimal parameters, we chose one of the most popular algorithms - GridSearchCV (Grid Search Cross-Validation).

Given that the decision tree algorithm allows for modifications of several hyperparameters, we selected five key parameters as described in the previous section.

The hyperparameters provided to the GridSearchCV algorithm were:

- **Max Depth**
- **Min Samples per Leaf**
- **Max Features**
- **Criterion**
- **Min Samples per Split**

We obtained results where the best hyperparameter values remained the same for both the original dataset and the reduced-dimension dataset.

**GridSearchCV results using original data**
| Max Depth | Min Samples per Leaf | Max Features | Criterion | Min Samples per Split |
|-----------|----------------------|--------------|-----------|----------------------|
| ų         | 5                    | log2         | log_loss  | 9                    |

**GridSearchCV results using two-dimensional data**
| Max Depth | Min Samples per Leaf | Max Features | Criterion | Min Samples per Split |
|-----------|----------------------|--------------|-----------|----------------------|
| 10        | 1                    | None         | Gini      | 4                    |

With these hyperparameters, the models showed an improvement of 8% and 9% compared to the initial models that used default parameters. We will discuss the classification quality results along with the evaluation methods for quality.

## Methods for Evaluating Classification Quality

### Cross-Validation Method

To assess the quality of this model, the cross-validation method was employed. This approach divides the dataset into five equal parts, using each part as a testing set while the rest are used as training sets.

The outcomes of our cross-validation analysis were as follows: the initial model utilizing the original dataset displayed a 75% accuracy, while the subsequent model employing the two-dimensional dataset achieved a 69% accuracy. 

### Holdout Validation Method

An alternative approach called "holdout validation" was employed to evaluate the decision tree classifier's performance. Employing this technique on the classifier, coupled with the designated parameters and the original dataset, yielded a 73% accuracy in correctly categorizing the test dataset. Conversely, utilizing the two-dimensional dataset led to a 71% accuracy through the holdout validation method.

## Confusion Matrix

These results represent the confusion matrix outcomes. The purpose of this matrix is to assess the classifier's performance by comparing combinations of actual and predicted values.

In this case, the model using the original dataset correctly identified 157 instances where the predictions matched the actual positive cases. Additionally, it accurately recognized 69 instances of negative cases. However, there were 45 instances where the model mistakenly classified positive cases when they were actually negative, and 40 instances where it missed identifying actual positive cases. 

![image](https://github.com/Voleisho/Airbnb-Client-Rating-Prediction/assets/141240910/7809faa8-9567-4e88-9fe3-23d581eb7a6b)



Similarly, for the model using the two-dimensional dataset, 147 instances where the predictions aligned with the actual positive cases. Furthermore, it accurately recognized 74 instances of negative cases. However, there were 40 instances where the model erroneously classified positive cases when they were actually negative, and 50 instances where it failed to identify actual positive cases. These metrics are vital for assessing the classifier's precision, recall, and overall effectiveness in distinguishing between the two classes. Comparing these results with those of the first model could offer valuable insights into the impact of using the two-dimensional dataset on the model's performance.


![image](https://github.com/Voleisho/Airbnb-Client-Rating-Prediction/assets/141240910/e1bb3500-f0f9-476a-98fd-84eb0e367b07)


Based on the confusion matrix comparison, both models exhibit strengths and weaknesses. Model 1 demonstrates better performance in terms of true positives and false negatives, indicating its proficiency in identifying actual positive cases. On the other hand, Model 2 excels in true negatives and false positives, implying it's better at recognizing negative cases. The choice between these models depends on the specific goal of the analysis: whether minimizing false positives (Model 2) or false negatives (Model 1) is more critical. Additionally, these results emphasize the impact of dataset dimensions on the model's ability to classify instances accurately.
Therefore, based on these results, the classifier's performance is not very satisfactory, as there are numerous errors present.

## Classification Metrics

These results represent classification model quality metrics used to assess the accuracy and performance of the model using both original and reduced-dimension data.

These metrics show that the model performed decently on Class 1 (with higher precision, recall, and F1-score), but there's room for improvement in Class 0 where the values are comparatively lower.

**Classification metrics results using original data** 
|    | Precision | Recall  | F1-Score | Support |
|----|-----------|---------|----------|---------|
| 0  | 0.63      | 0.61    | 0.62     | 114     |
| 1  | 0.78      | 0.80    | 0.79     | 197     |
|    |           |         |          |         |
| Accuracy |       |         | 0.73     | 311     |


In this case, the model's performance seems to have improved across both classes, with increased precision, recall, and F1-score for Class 0 and notably improved recall for Class 1. The overall accuracy has also slightly increased.

**Classification metrics results using two-dimensional data** 
|    | Precision | Recall  | F1-Score | Support |
|----|-----------|---------|----------|---------|
| 0  | 0.58      | 0.66    | 0.61     | 114     |
| 1  | 0.78      | 0.72    | 0.75     | 197     |
|    |           |         |          |         |
| Accuracy |       |         | 0.70     | 311     |

In summary, the overall quality of the models is not good, as there are many incorrect predictions and the accuracy is moderate.

## ROC Curve

Visualizing the graphical representation, the ROC curve illustrates how classifier predictions change based on the decision threshold (refer to Figure 8). AUC is an evaluation metric that measures how well a classifier can distinguish between positive and negative values. Our AUC result is 0.70, when using the classifier for original data and two-dimensional data too. This indicates that the classifier has a moderate ability to distinguish between the two classes. The results are satisfactory, but not sufficiently high to consider the models highly successful.

![image](https://github.com/Voleisho/Airbnb-Client-Rating-Prediction/assets/141240910/78cd864d-77cb-4a7a-888d-8025feb2796b)



![image](https://github.com/Voleisho/Airbnb-Client-Rating-Prediction/assets/141240910/1eeaf84f-a5c6-4638-9b78-17e62c1dcba7)

