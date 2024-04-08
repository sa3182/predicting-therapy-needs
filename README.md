# Mental Health Prediction

This project performs several machine learning classification techniques on the mental health survey data to pick the
best classification technique

## Setup

Download and install the latest [python](https://www.python.org/downloads/)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy, pandas, seaborn, matplotlib,
scikit-learn, xgboost and jinja2.

```bash
pip install numpy
pip install pandas
pip install seaborn
pip install matplotlib
pip install scikit-learn
pip install xgboost
pip install jinja2
```

## Usage

We strongly recommend the usage of jupiter notebook to load the script mental_health_prediction.py into jupyter notebook
and
run it

If Jupyter notebook is not available, Please use below commands to run the script

After installing all the required python libraries, Copy the data file survey_data.csv in the same folder as the script and then run the script 


Run the python script mental_health_prediction.py

```bash
python3 mental_health_prediction.py
```

If above doesn't work, if alias is configured for python use the alias as shown below

```bash
py mental_health_prediction.py
```

Once the script execution starts data is loaded into application, data analysis starts, and the results are printed on
the console which
identifies and provides the best classification technique for the mental health survey data
