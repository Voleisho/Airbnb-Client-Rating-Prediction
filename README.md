**Introduction: Predicting Airbnb Ratings using Decision Tree Analysis**

Welcome to my project, where I dive into the world of data analysis and prediction with a focus on Airbnb ratings. As a student, I've taken on the challenge of exploring how Decision Tree classifiers can help us anticipate whether an Airbnb client will give a rating above or below 90. This endeavor is an essential part of my portfolio, showcasing my skills and passion for data-driven insights.

**What's the Project About?**
In this repository, I've chosen to work with Barcelona Airbnb data, examining various factors that might influence client ratings. These factors range from the type of room and its capacity to the cleanliness rating and even a restaurant index. By analyzing this dataset, I aim to create a predictive model using Decision Trees that can help hosts understand guest satisfaction and tailor their offerings accordingly.

**The Decision Tree Approach:**
I've taken a step-by-step approach to build and fine-tune the Decision Tree model:
1. **Data Preparation:** I've carefully cleaned and organized the data, ensuring it's ready for analysis.
2. **Feature Selection:** I've chosen the most important features that can affect client ratings, simplifying the model while retaining its effectiveness.
3. **Model Building:** I've created a Decision Tree classifier, a tool that learns patterns in the data to make predictions.
4. **Parameter Optimization:** I've used GridSearchCV to find the best parameters for the model, ensuring optimal performance.
5. **Evaluation:** I've thoroughly assessed the model's accuracy and effectiveness in predicting ratings.

**Comparing Two-Dimensional Data:**
As an extra step, I've explored a unique technique. I've transformed the data into a two-dimensional representation using t-SNE, enabling a visual comparison of the original and reduced-dimensional datasets. I've then repeated the Decision Tree process on both datasets to understand how this impacts the model's predictions.

**Why It Matters:**
This project isn't just about building a model—it's about gaining insights. As a student, I'm excited to showcase my ability to use data analysis and machine learning to solve real-world problems. Predicting Airbnb ratings has practical implications for hosts and provides a tangible way to understand the power of data-driven decision-making.

### Data

The Barcelona Airbnb dataset contains various details about properties available for short-term stays in Barcelona. It includes features like price, room type, capacity, cleanliness rating and more. This dataset enables us to explore factors that impact guest experiences and ratings.

#### Features:
File - airbnb_data.csv
"realSum" - total Airbnb listing price (numeric)
"room_type" - type of room offered (private room, shared room, entire home/apt) (categorical)
"person_capacity" - maximum number of accommodated residents (numeric)
"host_is_superhost" - whether the host is a superhost (ordinal 0 or 1)
"multi" - whether multiple rooms are included in the listing (binary 0-1)
"biz" - whether the listing is intended for business purposes (binary 0-1)
"cleanliness_rating" - cleanliness rating (numeric)
"guest_satisfaction_overall" - overall guest satisfaction rating (numeric)
"bedrooms" - number of bedrooms (numeric)
"dist" - distance to the city center (numeric)
"metro_dist" - distance to the metro (numeric)
"attr_index" - attractiveness index (numeric)
"attr_index_norm" - normalized attractiveness indexes
"rest_index" - restaurant index (numeric)
"rest_index_norm" - normalized restaurant index (numeric)
"lng" - longitude coordinate (numeric)
"lat" - latitude coordinate (numeric)

#** Research Objective and Goals

**Research Objective:** Classify the original data and reduced-dimension data.

**Goals:**
Choose a variable with two classes for classification.
Examine descriptive statistics for the data and the dependent variable classes.
Perform classification using three algorithms.
Visualize results with ROC curves.
Calculate model statistics.


### Comparison of Data Characteristics: Group 0 (Rating < 90) vs Group 1 (Rating > 90)

In this analysis, we delve into two distinct groups of Airbnb property data based on client ratings in the captivating city of Barcelona. Group 0 comprises properties with client ratings below 90, while Group 1 includes properties that have received ratings exceeding 90. These tables provide a glimpse into the key attributes of each group, allowing us to uncover noteworthy differences between them.

#### Group 0 (Rating < 90):
The first table presents statistical summaries of attributes for properties within Group 0. These properties have received client ratings below 90, and the table sheds light on the characteristics associated with such ratings.

|        | realSum | person_capacity | cleanliness_rating | bedrooms | dist | Metro_dist | Attr_index | Rest_index | lng | lat |
| ------ | ------- | --------------- | ------------------ | -------- | ---- | ---------- | ---------- | ---------- | --- | --- |
| Count  | 598.00  | 598.00          | 598.00             | 598.00   | 598.00 | 598.00     | 598.00     | 598.00     | 598.00 | 598.00 |
| Mean   | 311.49  | 2.95            | 8.60               | 1.26     | 2.07 | 0.42       | 469.58     | 883.97     | 2.17 | 41.39 |
| Std    | 464.61  | 1.40            | 1.21               | 0.65     | 1.29 | 0.26       | 264.67     | 454.44     | 0.02 | 0.02 |
| Min    | 69.59   | 2.00            | 2.00               | 0.00     | 0.14 | 0.01       | 93.82      | 159.84     | 2.12 | 41.35 |
| Median | 196.90  | 2.00            | 9.00               | 1.00     | 1.72 | 0.36       | 381.59     | 816.56     | 2.17 | 41.39 |
| Max    | 6943.70 | 6.00            | 10.00              | 6.00     | 8.44 | 1.68       | 2065.07    | 2608.33    | 2.23 | 41.46 |

#### Group 1 (Rating > 90):
The second table provides a parallel analysis, focusing on properties in Group 1 with client ratings exceeding 90. By examining the attributes of this group, we aim to uncover the distinctive features associated with higher client ratings.

|        | realSum | Person_capacity | Cleanliness_rating | bedrooms | dist | Metro_dist | Attr_index | Rest_index | lng | lat |
| ------ | ------- | --------------- | ----------------- | -------- | ---- | ---------- | ---------- | ---------- | --- | --- |
| Count  | 957.00  | 957.00          | 957.00            | 957.00   | 957.00 | 957.00     | 957.00     | 957.00     | 957.00 | 957.00 |
| Mean   | 273.96  | 2.63            | 9.72              | 1.19     | 2.15 | 0.45       | 461.12     | 873.72     | 2.17 | 41.39 |
| Std    | 179.83  | 1.18            | 0.50              | 0.51     | 1.39 | 0.28       | 270.66     | 465.72     | 0.02 | 0.02 |
| Min    | 69.59   | 2.00            | 6.00              | 0.00     | 0.12 | 0.01       | 98.66      | 167.93     | 2.11 | 41.35 |
| Median | 215.28  | 2.00            | 10.00             | 1.00     | 1.78 | 0.38       | 390.76     | 792.92     | 2.17 | 41.39 |
| Max    | 1770.66 | 6.00            | 10.00             | 3.00     | 8.06 | 2.40       | 2934.13    | 4542.75    | 2.22 | 41.46 |
