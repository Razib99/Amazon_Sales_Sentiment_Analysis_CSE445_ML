# Amazon_Sales_Sentiment_Analysis_CSE445_ML


 Step 1: Setting Up the Environment 
•	Installing Python 
•	Installing Anaconda 
•	Creating a New Conda Environment by opening the Anaconda Prompt 
•	Installing necessary libraries using pip 
•	Opening Jupyter Notebook 

Step 2: Data Collection and Loading 
•	Downloading the Dataset from Kaggle 
•	Importing the libraries 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
•	Loading the Dataset 

df = pd.read_csv('amazon.csv') df.head(5) 
Step 3: Data Inspection and Cleaning 
•	Inspecting the data 

df.info() 
df.isnull().sum() 
•	Dropping the missing values 

df.dropna(inplace=True) 
Step 4: Text Preprocessing 
•	Installing the NLTK data 

import nltk 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
import re 

•	Cleaning and Tokenizing the text 

stop_words = set(stopwords.words('english')) 
ps = PorterStemmer() 
def preprocess_text(text): 
text = re.sub('[^a-zA-Z]', ' ', text) 
text = text.lower() 
text = text.split() 
text = [ps.stem(word) for word in text if not word in stop_words] 
return ' '.join(text) 
df['cleaned_reviews'] = df['review_title'].apply(preprocess_text) 
Step 5: Labeling 
•	Assigning Sentiment Labels 

def assign_sentiment(rating): 
# Converting 'rating' column to float type 
df['rating'] = pd.to_numeric(df['rating'], errors='coerce') 
# Rounding 'rating' column to nearest integer 
df['rating'] = df['rating'].round().astype(int) 
if rating >= 4: 
return 1 # Positive 
elif rating == 3: 
return 0 # Neutral 
else: 
return -1 # Negative 
df['sentiment'] = df['rating'].apply(assign_sentiment) 
Step 6: Model Training and Evaluation 
•	Defining and Training Models 

# Importing necessary libraries 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.preprocessing import LabelEncoder 
# Encoding the labels 
label_encoder = LabelEncoder() 
y_train_encoded = label_encoder.fit_transform(y_train) 
y_test_encoded = label_encoder.transform(y_test) 
# Defining the models 
models = { 
'Logistic Regression': LogisticRegression(), 
'Naive Bayes': MultinomialNB(), 
'SVM': SVC(), 
'KNN': KNeighborsClassifier(n_neighbors=3), # Adjust n_neighbors to a suitable value 
'Random Forest': RandomForestClassifier(), 
'Gradient Boosting': GradientBoostingClassifier(), 
'XGBoost': XGBClassifier() 
} 
results = {} 
# Fitting the models and storing the results 
for name, model in models.items(): 
model.fit(X_train, y_train_encoded) 
y_pred = model.predict(X_test) 
y_pred_encoded = label_encoder.inverse_transform(y_pred) 
accuracy = accuracy_score(y_test, y_pred_encoded) 
report = classification_report(y_test, y_pred_encoded, zero_division=0) 
confusion = confusion_matrix(y_test, y_pred_encoded) 
results[name] = { 
'accuracy': accuracy, 
'classification_report': report, 
'confusion_matrix': confusion 
} 
# Printing the results 
for name, result in results.items(): 
print(f"Model: {name}") 
print(f"Accuracy: {result['accuracy']}") 
print(f"Classification Report:\n{result['classification_report']}") 
print(f"Confusion Matrix:\n{result['confusion_matrix']}\n") 
Result: 
1.	Model: Logistic Regression 

Accuracy: 0.9590443686006825 
Classification Report: 
precision recall f1-score support 
0 0.00 0.00 0.00 12 
1 0.96 1.00 0.98 281 
accuracy 0.96 293 
macro avg 0.48 0.50 0.49 293 
weighted avg 0.92 0.96 0.94 293 
Confusion Matrix: 
[[ 0 12] 
[ 0 281]] 
1.	Model: Naive Bayes 

Accuracy: 0.9590443686006825 
Classification Report: 
precision recall f1-score support 
0 0.00 0.00 0.00 12 
1 0.96 1.00 0.98 281 
accuracy 0.96 293 
macro avg 0.48 0.50 0.49 293 
weighted avg 0.92 0.96 0.94 293 
Confusion Matrix: 
[[ 0 12] 
[ 0 281]] 

1.	Model: SVM 

Accuracy: 0.962457337883959 
Classification Report: 
precision recall f1-score support 
0 1.00 0.08 0.15 12 
1 0.96 1.00 0.98 281 
accuracy 0.96 293 
macro avg 0.98 0.54 0.57 293 
weighted avg 0.96 0.96 0.95 293 
Confusion Matrix: 
[[ 1 11] 
[ 0 281]] 
1.	Model: KNN 

Accuracy: 0.9590443686006825 
Classification Report: 
precision recall f1-score support 
0 0.50 0.08 0.14 12 
1 0.96 1.00 0.98 281 
accuracy 0.96 293 
macro avg 0.73 0.54 0.56 293 
weighted avg 0.94 0.96 0.94 293 
Confusion Matrix: 
[[ 1 11] 
[ 1 280]] 

1.	Model: Random Forest 

Accuracy: 0.9658703071672355 
Classification Report: 
precision recall f1-score support 
0 1.00 0.17 0.29 12 
1 0.97 1.00 0.98 281 
accuracy 0.97 293 
macro avg 0.98 0.58 0.63 293 
weighted avg 0.97 0.97 0.95 293 
Confusion Matrix: 
[[ 2 10] 
[ 0 281]] 
1.	Model: Gradient Boosting 

Accuracy: 0.9658703071672355 
Classification Report: 
precision recall f1-score support 
0 1.00 0.17 0.29 12 
1 0.97 1.00 0.98 281 
accuracy 0.97 293 
macro avg 0.98 0.58 0.63 293 
weighted avg 0.97 0.97 0.95 293 
Confusion Matrix: 
[[ 2 10] 
[ 0 281]] 

1.	Model: XGBoost 

Accuracy: 0.9658703071672355 
Classification Report: 
precision recall f1-score support 
0 1.00 0.17 0.29 12 
1 0.97 1.00 0.98 281 
accuracy 0.97 293 
macro avg 0.98 0.58 0.63 293 
weighted avg 0.97 0.97 0.95 293 
Confusion Matrix: 
[[ 2 10] 
[ 0 281]]
