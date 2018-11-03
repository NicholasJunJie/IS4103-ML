import flask
import pickle
import pandas as pd
from flask import Flask
from flask import request


#code which helps initialize our server
app = Flask(__name__)

#Load our ML model
model = pickle.load(open("model.pkl","rb"))

@app.route('/predict',methods=['POST'])
def predict():
    d = request.get_json()

    frame = pd.DataFrame([
        [d["gender"], d["married"], d["dependents"], d["education"], d["selfEmployed"], d["applicantIncome"], d["coApplicantIncome"], d["loanAmount"], d["loanAmountTerm"], d["creditHistory"], d["propertyArea"]]
    ], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    
    cleanup_nums = {
    					"Gender":     {"Male": 1, "Female": 0}, 
                    	"Married":     {"Yes": 1, "No": 0}, 
                    	"Education":     {"Graduate": 1, "Not Graduate": 0},
                    	"Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
                    	"Self_Employed":     {"Yes": 1, "No": 0},
                    	"Property_Area": {"Urban": 0, "Semiurban": 1, "Rural": 2 },
               		}
    frame.replace(cleanup_nums,inplace=True)
    prediction = model.predict(frame).tolist() # convert numpy.ndarray to list

    #preparing a response object and storing the model's predictions
    # response = {}
    # response['predictions'] = prediction
    
    # The return type must be a string, tuple, Response instance, or WSGI callable
    return str(prediction[0]) 

if __name__ == '__main__': # only run the app when this api.py is called directly
    app.run(host='0.0.0.0', port=8000, debug=True)
