from flask import Flask
from flask import request
import pandas as pd
import pickle

#code which helps initialize our server
app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route('/predict',methods=['POST'])
def json_example():
    d = request.get_json()
    df_test = pd.DataFrame(data=d, index=[0])
    
    cleanup_nums = {
					"Gender":     {"Male": 1, "Female": 0}, 
                	"Married":     {"Yes": 1, "No": 0}, 
                	"Education":     {"Graduate": 1, "Not Graduate": 0},
                	"Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
                	"Self_Employed":     {"Yes": 1, "No": 0},
                	"Property_Area": {"Urban": 0, "Semiurban": 1, "Rural": 2 },
               		}
    df_test.replace(cleanup_nums,inplace=True)
    prediction = model.predict(df_test)
    print(prediction)

    return "312312"

if __name__ == '__main__': # only run the app when this api.py is called directly
    app.run(host='0.0.0.0', port=8000, debug=True)
