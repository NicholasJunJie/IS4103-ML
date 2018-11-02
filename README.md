# IS4103-ML
Logistic Regression Model

1. Install python
2. Create a virtual env (OPTIONAL)  ## READ MORE for the purpose https://docs.python.org/3/tutorial/venv.html
  2.1 pip install virtualenvwrapper-win 
  2.2 mkvirtualenv <NAME_OF_VIRTUAL_ENV)

3. Install the necassary python libraries
  3.1 pip install flask numpy pandas sklearn scipy scikit-learn

4. Run RESTFUL service directly
  4.1 python api.py
  
  
Files included in repository:
1) Library folder containing the dataset
2) ml_model.py containing the ML algorithm
3) api.py containing the REST API
4) model.pkl - Seralized model



BRIEF EXPLAINATION OF THE FLOW:
1) Run ml_model.py to create a copy of the model.pkl (Can skip this unless you update the algorithm)
2) Run api.py to start the REST service (Running this file will run the model.pkl)





  
  

