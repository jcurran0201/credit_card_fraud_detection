# Credit Card Fraud Detection 
Machine learning fraud detection system using behavioral, temporal, and geographic features with financial impact evaluation and model comparison.
# Fraud Detection Project Overview https://www.kaggle.com/datasets/kartik2112/fraud-detection 
The Kaggle Fraud Detection dataset is a large, synthetic credit card transactions dataset designed to support supervised fraud detection modeling. It simulates realistic consumer spending behavior across roughly 1,000 customers. The dataset is highly imbalanced, reflecting real-world fraud scenarios where fraudulent transactions are a small minority. Although synthetic, the data closely mirrors real financial patterns, making it well-suited for feature engineering, EDA, and machine learning model development.
## Data Cleaning
  No missing values or duplicates were present
  The non-informative Unnamed: 0 column was removed
  Training and testing datasets were kept separate to prevent leakage
  Feature Engineering
All feature engineering logic was implemented using reusable def() functions to maintain consistency across training and testing datasets.
Behavioral features: Features about spending habits in specific time frames before the transaction occurred. We created features for spending habits (how many times card was used and how much was spent on the card) an hour before transaction, day before transaction, and a week before the transaction. 
Temporal features: Weekend and off-peak transactions (10 PM–6 AM) were found and used features. 
Demographic features: Cardholder age at transaction time computed from date of birth and spending habits based on the subject’s job title were used 
Spatial features: Distance between cardholder residence and transaction location computed with the Haversine formula using latitude/longitude  coordinates provided. 
## Exploratory Data Analysis 
Fraud transactions were mostly under $500 
Fraud victims tended to be slightly older, but age was not a strong predictor
Most merchants had relatively small fraud amounts, with a few outliers exceeding $10,000
Cards with multiple fraudulent transactions generally resulted in higher total losses
Certain merchants (e.g., Kuhic LLC, Kozey-Boehm, Boyer PLC, Terry-Huel) appeared frequently in the top fraudulent spend categories


## Machine Learning Approach
### Features Used
### Modeling included a combination of:
Transactional: amount, transaction category, timestamp, merchant details
Demographic: age, gender, spending habits based on job title
Geographic: customer and merchant latitude/longitude, city, ZIP code, state
Behavioral: number of unique states per card, card transaction history over recent days/weeks
### Handling Class Imbalance
Tree-based models (Decision Trees, Random Forest, XGBoost) were used because they can handle class imbalance via class weighting and boosting mechanisms. SMOTE was avoided as it can introduce unrealistic synthetic transactions and distort temporal patterns. Instead, class weights and probability thresholds were tuned to balance precision and recall while preserving natural transaction sequences.
### Hyperparameter Tuning
RandomizedSearchCV was used for efficiency on the large dataset
Optimization focused on improving PR AUC, precision, recall, and F1 scores while considering computational constraints
Model Performance
### Decision Tree
#### Baseline model
Default probability threshold (0.5) led to excessive false positives
Threshold increased to 0.9 to reduce false positives
Even after tuning, it is highly sensitive to outliers, resulting in the model being overly aggressive in detecting fraud. Due the model’s overagression, it was decided to not investigate its financial impact because the model was clearly not usable in a real world scenario
### Random Forest
Initial performance: precision 0.82, recall 0.64, F1 0.72, PR AUC 0.76
After RandomizedSearchCV: slight F1 decrease to 0.68, but recall and PR AUC improved
Better than Decision Trees due to using a much wider range of features than decision trees
### XGBoost
Initial performance: precision 0.65, recall 0.78, F1 0.71, PR AUC ~0.80
After RandomizedSearchCV: slight increase in precision, minor drop in recall, F1 unchanged, PR AUC dropped by 0.0075
Probability threshold raised to 0.94 before RandomizedSearchCV the 0.95 after the RandomizedSearchCV to balance precision and recall
Strongest model overall due to effective integration of behavioral, temporal, and demographic features
### Insights
Decision Trees relied heavily on immediate transaction amounts and short-term totals, making them overly aggressive
Random Forest and XGBoost incorporated broader context: time of day, cardholder age, and transaction history over several days/weeks
Precision-recall tradeoff is critical: adjusting thresholds and tuning hyperparameters is essential for production-ready fraud detection models 
Among the evaluated models, the Random Forest after RandomizedSearchCV was the most effective at maximizing fraud loss prevention, as it recovered the highest dollar value of fraudulent transactions (~$1.00M) and reduced missed fraud at the lowest value (~131K). This indicates the strongest detection capability and the greatest direct financial protection. However, this improvement came with a substantial increase in legitimate transactions incorrectly flagged (~$384K), meaning higher customer friction and operational review burden. By comparison, the tuned XGBoost model produced fewer false positives than the tuned Random Forest but allowed more fraud to go undetected and recovered less total fraud value. Therefore, if the primary objective is minimizing financial loss from fraud, the tuned Random Forest is the most effective. If the goal is a more balanced tradeoff between fraud prevention and customer experience, the tuned XGBoost model may be operationally preferable.
