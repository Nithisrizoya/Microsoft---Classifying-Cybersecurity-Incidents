# Microsoft---Classifying-Cybersecurity-Incidents

# Introduction

In the rapidly evolving cybersecurity landscape, the increasing volume of incidents has overwhelmed Security Operation Centers (SOCs). To address this, there is a pressing need for solutions that can automate or support the remediation process effectively. This project leverages the GUIDE dataset—a groundbreaking collection of real-world cybersecurity incidents—to develop machine learning models for predicting significant cybersecurity incidents and facilitating informed decision-making.

# Problem Statement:

As a data scientist at Microsoft, he/she is tasked with enhancing the efficiency of Security Operation Centers (SOCs) by developing a machine learning model that can accurately predict the triage grade of cybersecurity incidents. Utilizing the comprehensive GUIDE dataset, the goal is to create a classification model that categorizes incidents as true positive (TP), benign positive (BP), or false positive (FP) based on historical evidence and customer responses. The model should be robust enough to support guided response systems in providing SOC analysts with precise, context-rich recommendations, ultimately improving the overall security posture of enterprise environments.

# Business Use Cases:

The solution developed in this project can be implemented in various business scenarios, particularly in the field of cybersecurity. Some potential applications include:

**Security Operation Centers (SOCs):** Automating the triage process by accurately classifying cybersecurity incidents, thereby allowing SOC analysts to prioritize their efforts and respond to critical threats more efficiently.

**Incident Response Automation:** Enabling guided response systems to automatically suggest appropriate actions for different types of incidents, leading to quicker mitigation of potential threats.

**Threat Intelligence:** Enhancing threat detection capabilities by incorporating historical evidence and customer responses into the triage process, which can lead to more accurate identification of true and false positives.

**Enterprise Security Management:** Improving the overall security posture of enterprise environments by reducing the number of false positives and ensuring that true threats are addressed promptly.

# Dataset Overview :

GUIDE_train.csv (2.43 GB) GUIDE_test.csv (1.09 GB) **link :** https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction

The GUIDE dataset consists of over 13 million pieces of evidence across three hierarchical levels:

  1.Evidence: Individual data points supporting an alert (e.g., IP addresses, user details).

  2.Alert: Aggregated evidences indicating potential security incidents.

  3.Incident: A comprehensive narrative representing one or more alerts.

  4.Size: Over 1 million annotated incidents with triage labels, and 26,000 incidents with remediation action labels.

  5.Telemetry: Data from over 6,100 organizations, including 441 MITRE ATT&CK techniques.
  
  6.Training/Testing: The dataset is divided into a training set (70%) and a test set (30%), ensuring stratified representation of triage grades and identifiers.

 # Tools Used

  **IDE & Notebooks:** Google Colab 

  **Programming Language:** Python

  **Libraries:** scikit-learn, Pandas, Matplotlib, Seaborn, NumPy

  **Cloud Services:** BigQuery, Google Cloud Storage, Google Compute Engine

  **Version Control:** Git, Github

# Approach

# 1. Data Exploration and Understanding

**Initial Inspection:**

  * Loaded the GUIDE_train.csv and 'GUIDE_test.csv' dataset and performed an initial inspection to understand the structure of the data, including the number of features, types of variables (categorical, numerical), and the distribution of the target variable (TP, BP, FP).
  
  * Train Dataset contained - rows x columns (9494278 x 45)
  
  * Test Dataset contained - rows x columns (4147888 x 46) one extra column called usage(Public or Private, which won't be utilized as it isn't in our train data)

**Exploratory Data Analysis (EDA):**

Used visualizations and statistical summaries to identify patterns, correlations, and potential anomalies in the data.

# 2. Data Preprocessing

**Handling Missing Data and Duplicates:** Identified missing values and droped columns where missing values were more than 50% of the total rows. Duplicates were removed as well.Conversion of datatypes, for example string time to datetime done as well.

**Feature Engineering**: Created new features or modified existing ones to improve model performance. For example, combined related features, derived new features from timestamps (like hour of the day or day of the week), and normalized numerical variables.

# 3. Exploratory Data Analysis (EDA)

**EDA:** Further a closer look at the distribution of incidents across time were as done. Clearly shows fluctuations across time which would be useful for our model.

        * time1.png
        * time2.png
        * time3.png

And the co-relation heatmap as well to understand co-linearity among the features

        * correlation_heatmap.png

**Multi-colinearity** Removing one of the columns where pairs are highly co-related to avoid multi-colinearity.

# 4. Model Selection and Training

**Machine Learning Models:** Experimented with more sophisticated models such as Random Forests, Gradient Boosting Machines (e.g., XGBoost, LightGBM). Each model was tuned using techniques like grid search or random search over hyperparameters.

**Train-Validation Split:** Before diving into model training, split the train.csv data into training and validation sets. This allowed for tuning and evaluating the model before final testing on test.csv. A typical 70-30 or 80-20 split was used, varying depending on the dataset's size.

**Encoding Categorical Variables:** Converted categorical features into numerical representations using techniques like one-hot encoding, and label encoding depending on the nature of the feature and its relationship with the target variable.

**Stratification:** Can use stratified sampling to ensure that both the training and validation sets had similar class distributions, especially since the target variable was imbalanced. But in this case we can attempt to proceed with as is at Random Forest and XGBoost are well equiped with dealing with imbalance data and it also represents real world scenario. We can look into other methods after testing this way and evaluating it's metrics.

# 5. Model Evaluation and Tuning

**Performance Metrics:** Evaluated the model using the validation set, focusing on macro-F1 score, precision, and recall. Analyzed these metrics across different classes (TP, BP, FP) to ensure balanced performance.

**Hyperparameter Tuning:** Fine-tuned hyperparameters based on the initial evaluation to optimize model performance. Adjusted learning rates, regularization parameters, tree depths, and the number of estimators, depending on the model type.

# 6. Final Evaluation on Test Set

**Testing:** Once the model was finalized and optimized, it was evaluated on the test.csv dataset. Reported the final macro-F1 score, precision, and recall to assess how well the model generalized to unseen data.

# 7. Final Results and Model Performance Analysis

**Training Dataset Performance:**

Trained using ensemble methods XGBoost and Random Forest; Random Forest performed the best.

      * RF_train_metrics.png
      * XGB_train_metrics.png

**Test Dataset Performance:**

Selected the Random Forest Classifier and applied it to the test dataset.

       * RF_test_metrics.png

# 8. Inferences

**High Training Performance:** The model exhibits very high performance on the training dataset, indicating it has learned the patterns in the data well.

**Good Generalization:** The model performs robustly on the test dataset, suggesting it generalizes well to new, unseen data. The slight decrease in accuracy from training to testing is typical and indicates good generalization without significant overfitting.

**Class-wise Variations:** While the model maintains strong performance across most classes, there is a noticeable drop in performance for benign positive incidents in the test set. This could be an area for further investigation and improvement. Overall, the Random Forest model demonstrates strong capabilities in classifying cybersecurity incidents, with good generalization to real-world data. Future improvements could focus on enhancing performance for specific classes and continuing to monitor and adjust the model as more data becomes available.


        


