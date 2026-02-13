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
### <img width="1016" height="626" alt="Money spent in fraud transactions" src="https://github.com/user-attachments/assets/5e212b60-0779-45d8-a69f-bc663bd285df" />

Fraud victims had a slightly older average age, but it did not indicate that a specific age group is more vulnerable to becoming a victim of fraud
### <img width="371" height="472" alt="Age" src="https://github.com/user-attachments/assets/30dbdff6-51d9-4372-a4b1-a5e741f32711" />

As merchants have more fraud transactions, the amount of money lost to fraud at the store tends to increase, especially after 20 fraud transactions.
### <img width="865" height="585" alt="Fraud Frequency by store" src="https://github.com/user-attachments/assets/373a53e1-6d71-44a1-95df-45bdb6749216" />

Cards with more fraudulent transactions generally resulted in higher total losses 
### <img width="870" height="546" alt="Screenshot 2026-02-13 at 1 24 32 PM" src="https://github.com/user-attachments/assets/2f7e1f26-4184-4e0e-9a60-85e3ac7b4441" />

Certain merchants (e.g., Kuhic LLC, Kozey-Boehm, Boyer PLC, Terry-Huel) appeared frequently in the top fraudulent spend categories
### <img width="1007" height="573" alt="Most fraud transactions by store" src="https://github.com/user-attachments/assets/54e986f2-1d13-428d-b3fc-17d1a399ed55" />
### <img width="666" height="581" alt="Stores that lost most money in fraud" src="https://github.com/user-attachments/assets/b9dd9b23-5c99-4f32-987d-854d53571b1f" />
The following merchants appear in most fraud transactions by store and most money lost by store : Kozey-Boehm, Kuhic LLC, Terry-Huel, and Boyer PLC

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
Decision trees were used here as a baseline model to determine which features might be more heavily valued in more competitive models. 
Default probability threshold (0.5) led to excessive false positives. Threshold increased to 0.9 to reduce false positives 
#### Decision Trees before RandomizedSearchCV  
The initial hyperparamaters used were DecisionTreeClassifier(
    criterion='entropy',
    max_depth=7,
    min_samples_leaf=500,
    min_samples_split=100,
    class_weight='balanced',
    random_state=42) 

In the training set of the decision tree model the precision is 0.35 and the recal is 0.86. In the testing set the percision dropped 0.08 to 0.27 and the recall dropped 0.03 and the f1 score had a 0.1 point decrease. This suggests that the model does not have an overfitting problem. It should be noted that the decision tree model is extremely agressive in calling a transaction fraud. This is expected because of the sensitivity of decision trees to different features, class imbalance, hyperparamaters, and probability thresholds of fraud     
### <img width="346" height="665" alt="d_tree classificaiton report before tuning" src="https://github.com/user-attachments/assets/a16a7f84-0465-4e80-b807-0ea977542b8f" /> 
The root node in the tree starts with amount spent and as the tree gets deeper, it begins to use more unique features to decipher between fraud and normal transactions. 
### <img width="1137" height="466" alt="Screenshot 2026-02-13 at 2 05 52 PM" src="https://github.com/user-attachments/assets/1f23734d-61c5-4b1b-ba92-758d77e571fc" />   
As expected in a heavily imbalanced dataset, a high accuracy score is expected and was achieved with an ROC score of 0.98.
### <img width="739" height="577" alt="Screenshot 2026-02-13 at 4 02 09 PM" src="https://github.com/user-attachments/assets/d60ba689-5f7b-4fd4-bb44-d3174156a3c8" /> 
The PRC score was significantly lower at 0.69 than the ROC score, which is expected. The PRC score of the decision trees is expected to be lower than in other tree based models such as Random Forest and XGBoost
### <img width="758" height="568" alt="Screenshot 2026-02-13 at 4 00 11 PM" src="https://github.com/user-attachments/assets/d07e38f3-20b9-4218-a635-6a824b92f933" /> 
Primary features that were used in the model: amount spent in the transaction, total amount of money spent of the card in the last 24 hours, if the transaction occured durng off peak hours. Some of the secondary features that were used are the amount of transactions on the card, the time since the last transaction, and age of the card owner.  
### <img width="965" height="630" alt="Screenshot 2026-02-13 at 2 07 02 PM" src="https://github.com/user-attachments/assets/db8a7d87-b180-4854-a2ff-dd6dcc886a1c" />

#### Decision Trees after RandomizedSearchCV  
Even after tuning, it is highly sensitive to outliers, resulting in the model being overly aggressive in detecting fraud. Due the model’s overagression, it was decided to not investigate its financial impact because the model was clearly not usable in a real world scenario 
### <img width="385" height="673" alt="Screenshot 2026-02-13 at 2 11 01 PM" src="https://github.com/user-attachments/assets/202ceddb-317b-4b3f-9479-f8c3b5e67c1f" />
### <img width="1280" height="494" alt="Screenshot 2026-02-13 at 2 11 53 PM" src="https://github.com/user-attachments/assets/396555b1-c2db-46af-aecc-a8b0218a0e17" />
### <img width="719" height="572" alt="Screenshot 2026-02-13 at 2 24 41 PM" src="https://github.com/user-attachments/assets/82cbf537-f960-40d4-9a36-6dc21dfbdbc5" />
### <img width="655" height="485" alt="Screenshot 2026-02-13 at 2 13 56 PM" src="https://github.com/user-attachments/assets/9226cf9b-4126-4da6-bd22-c4fdea816e87" />
### <img width="971" height="665" alt="Screenshot 2026-02-13 at 2 12 35 PM" src="https://github.com/user-attachments/assets/45379159-0189-4107-a996-7d3db6a5c3f6" />


### Random Forest
Initial performance: precision 0.82, recall 0.64, F1 0.72, PR AUC 0.76 
### <img width="524" height="666" alt="Screenshot 2026-02-13 at 4 09 53 PM" src="https://github.com/user-attachments/assets/bba1aac4-9ae9-47ed-9695-0198f6f061c6" />
### <img width="469" height="576" alt="Screenshot 2026-02-13 at 2 26 31 PM" src="https://github.com/user-attachments/assets/3cf8415e-37aa-4dba-97bd-41155b172ae2" />
### <img width="714" height="479" alt="Screenshot 2026-02-13 at 3 32 17 PM" src="https://github.com/user-attachments/assets/ab6e613d-abbe-4862-913f-8a1f68ce9894" />
After RandomizedSearchCV: slight F1 decrease to 0.68, but recall and PR AUC improved
Better than Decision Trees due to using a much wider range of features than decision trees 
###  <img width="330" height="666" alt="Screenshot 2026-02-13 at 2 27 22 PM" src="https://github.com/user-attachments/assets/0a012e6f-8abb-4898-bce9-3642cb91da22" />
### <img width="533" height="430" alt="Screenshot 2026-02-13 at 2 28 49 PM" src="https://github.com/user-attachments/assets/92379a77-d17e-40b8-8b96-5b0b718807fb" />

### <img width="708" height="569" alt="Screenshot 2026-02-13 at 2 29 15 PM" src="https://github.com/user-attachments/assets/8c0ffb5d-8ce7-4a5d-ada9-7a294189bc3e" />
### <img width="1016" height="617" alt="Screenshot 2026-02-13 at 4 04 17 PM" src="https://github.com/user-attachments/assets/b913ce5a-2be2-4db9-a4cd-f24615850315" />

### XGBoost
Initial performance: precision 0.65, recall 0.78, F1 0.71, PR AUC ~0.80
  
### <img width="531" height="641" alt="Screenshot 2026-02-13 at 2 35 18 PM" src="https://github.com/user-attachments/assets/86431cc1-d53f-47d2-8587-6c4721f17247" />
### <img width="509" height="330" alt="Screenshot 2026-02-13 at 2 36 17 PM" src="https://github.com/user-attachments/assets/56e050d5-fa62-4d42-9b76-6b14c1967c3c" />
### <img width="421" height="578" alt="Screenshot 2026-02-13 at 2 37 46 PM" src="https://github.com/user-attachments/assets/aa5579c5-94fd-4b0b-aac6-c1a4aa912e36" /> 
### <img width="695" height="440" alt="Screenshot 2026-02-13 at 4 06 44 PM" src="https://github.com/user-attachments/assets/cebc06b8-0842-4b68-bf7f-b745297a7405" />
After RandomizedSearchCV: slight increase in precision, minor drop in recall, F1 unchanged, PR AUC dropped by 0.0075
### <img width="372" height="665" alt="Screenshot 2026-02-13 at 4 14 42 PM" src="https://github.com/user-attachments/assets/6e911f95-c9db-4814-a77e-42f3be8f8901" />
### <img width="367" height="284" alt="Screenshot 2026-02-13 at 2 41 04 PM" src="https://github.com/user-attachments/assets/01ccd333-4007-4352-8820-d8867772a0d5" /> 
### <img width="397" height="338" alt="Screenshot 2026-02-13 at 2 41 47 PM" src="https://github.com/user-attachments/assets/e260b9c5-0f73-444a-8588-b4f0358c11af" />
### <img width="626" height="377" alt="Screenshot 2026-02-13 at 2 40 29 PM" src="https://github.com/user-attachments/assets/0ba8d8ad-f288-4cdd-a9ff-949b33cf7f4b" />

Probability threshold raised to 0.94 before RandomizedSearchCV the 0.95 after the RandomizedSearchCV to balance precision and recall
Strongest model overall due to effective integration of behavioral, temporal, and demographic features
### Insights
Decision Trees relied heavily on immediate transaction amounts and short-term totals, making them overly aggressive
Random Forest and XGBoost incorporated broader context: time of day, cardholder age, and transaction history over several days/weeks
Precision-recall tradeoff is critical: adjusting thresholds and tuning hyperparameters is essential for production-ready fraud detection models 
Among the evaluated models, the Random Forest after RandomizedSearchCV was the most effective at maximizing fraud loss prevention, as it recovered the highest dollar value of fraudulent transactions (~$1.00M) and reduced missed fraud at the lowest value (~131K). This inticates the strongest detection capability and the greatest direct financial protection. However, this improvement came with a substantial increase in legitimate transactions being flagged incorrectly (~384K) that review burden will be higher and customers might be more frustrated. By comparison, the tuned XGBoost model produced fewer false positives than the Random Forest, but it allowed more fraud to go undetected and recovered less total fraud in monetary value. If the primary objective is minimizing financial loss from fruad the tuned Random Forest is the most effective model. If the goal is a more balanced tradeoff between fraud prevention and customer experience, the tune XGBoost model might be preferable 


