import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def impute_credit_history(cols):
    credit_history = cols[0]
    applicant_income = cols[1]
    
    if pd.isnull(credit_history):
        if applicant_income >= 10000:
            return 1.0

    else:
        return credit_history

# Read dataset and store in dataframe
train = pd.read_csv('library/data/train_ds.csv')

## BEGIN DATA CLEANSING

# For credit_history == NaN, if applicant_income >= 10000, then set credit_history = 1 
train['Credit_History'] = train[['Credit_History','ApplicantIncome']].apply(impute_credit_history,axis=1)

train.drop_duplicates(inplace=True)
train.dropna(inplace=True)

# Remove outlier (i.e the 8k outlier)
train = train[train['ApplicantIncome'] < 40000] 
train.drop(['Loan_ID'],axis=1,inplace=True)

# Clean up - Replace categorical variable
cleanup_nums = {"Gender":     {"Male": 1, "Female": 0}, 
                "Married":     {"Yes": 1, "No": 0}, 
                "Education":     {"Graduate": 1, "Not Graduate": 0},
                'Dependents': {"0": 0, "1": 1, "2": 2, "3+": 3},
                "Self_Employed":     {"Yes": 1, "No": 0},
                "Property_Area": {"Urban": 0, "Semiurban": 1, "Rural": 2 },
                "Loan_Status":     {"Y": 0, "N": 1},}

train.replace(cleanup_nums, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Loan_Status'],axis=1), 
                                                    train['Loan_Status'], test_size=0.30, 
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

#serializing our model to a file called model.pkl
pickle.dump(logmodel, open("model.pkl","wb"))

# print(classification_report(y_test,predictions))




