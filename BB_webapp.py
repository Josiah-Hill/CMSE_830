import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Analyzing baseball pitch outcomes')

st.header('By Josiah Hill')

st.write("Predicting pitch results based on pitch speed, spin rate, and break length.")

pitches = pd.read_csv(r"C:\Users\jojoh\OneDrive\Documents\School\MSDS\CMSE 830\Baseball ML App\pitchdata.csv").iloc[:,1:6].dropna(axis=0)

# Feature engineer code values for testing
pitches["code"] = pitches["code"].replace(["*B", "P", "I", "Z", "Q"], "B")
pitches["code"].replace(["E", "H"], "D")
pitches["code"].replace(["T", "L", "R"], "F")
pitches["code"].replace(["W", "M", "Q"], "S")
pitches.loc[pitches.code == "*B", "code"] = "B"
pitches.loc[pitches.code == "P", "code"] = "B"
pitches.loc[pitches.code == "I", "code"] = "B"
pitches.loc[pitches.code == "Z", "code"] = "B"
pitches.loc[pitches.code == "Q", "code"] = "B"
pitches.loc[pitches.code == "E", "code"] = "D"
pitches.loc[pitches.code == "H", "code"] = "D"
pitches.loc[pitches.code == "T", "code"] = "F"
pitches.loc[pitches.code == "L", "code"] = "F"
pitches.loc[pitches.code == "R", "code"] = "F"
pitches.loc[pitches.code == "W", "code"] = "S"
pitches.loc[pitches.code == "M", "code"] = "S"
pitches.loc[pitches.code == "Q", "code"] = "S"
pitches.loc[pitches.code == "V", "code"] = "S"
pitches.loc[pitches.code == "B", "code"] = 0
pitches.loc[pitches.code == "C", "code"] = 1
pitches.loc[pitches.code == "D", "code"] = 2
pitches.loc[pitches.code == "F", "code"] = 3
pitches.loc[pitches.code == "S", "code"] = 4
pitches.loc[pitches.code == "X", "code"] = 5

# Select pitch
pitch = st.selectbox("Select a pitch to analyze:", 
                     ["Four-seam fastball", "Slider", "Two-seam fastball", 
                      "Change-up", "Sinker", "Curveball", "Cut fastball"])

if pitch == "Four-seam fastball":
    data = pitches[pitches["pitch_type"] == "FF"]

elif pitch == "Slider":
    data = pitches[pitches["pitch_type"] == "SL"]

elif pitch == "Two-seam fastball":
    data = pitches[pitches["pitch_type"] == "FT"]

elif pitch == "Change-up":
    data = pitches[pitches["pitch_type"] == "CH"]

elif pitch == "Sinker":
    data = pitches[pitches["pitch_type"] == "SI"]

elif pitch == "Curveball":
    data = pitches[pitches["pitch_type"] == "CU"]

elif pitch == "Cut fastball":
    data = pitches[pitches["pitch_type"] == "FC"]

sample_size = st.slider('Select a sample size', 100, 10000, step = 100)
rand_st = st.slider('Set a state for sampling', 0, 100)

data_ball = data[data["code"] == 0].sample(n = sample_size, replace = False, random_state = rand_st)
data_cs = data[data["code"] == 1].sample(n = sample_size, replace = False, random_state = rand_st)
data_hit = data[data["code"] == 2].sample(n = sample_size, replace = False, random_state = rand_st)
data_foul = data[data["code"] == 3].sample(n = sample_size, replace = False, random_state = rand_st)
data_ss = data[data["code"] == 4].sample(n = sample_size, replace = False, random_state = rand_st)
data_out = data[data["code"] == 5].sample(n = sample_size, replace = False, random_state = rand_st)

data_sample = pd.concat([data_ball, data_cs, data_hit, data_foul, data_ss, data_out])

X = data_sample.iloc[:,2:]
y = data_sample.iloc[:,0]
y=y.astype('int')

# Initialize variables for train/test split
start_state = 42
test_fraction = 0.2

# Split x and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=start_state)

clf = st.selectbox("Select a model type:", ["KNN", "Decision Tree", "Random Forest", "Gaussian"])

if clf == "KNN":
    my_classifier = KNeighborsClassifier()
    
elif clf == "Decision Tree":
    my_classifier = DecisionTreeClassifier(max_depth=5)
    
elif clf == "Random Forest":
    my_classifier = RandomForestClassifier(max_depth=5)
    
elif clf == "Gaussian":
    my_classifier = GaussianNB()

# Fit the data to the model
my_classifier.fit(X_train, y_train)

# Predicts the test data using the model
my_predictions = my_classifier.predict(X_test)

# Calculates how accurate the model is at predicting the test data
st.write("Accuracy: ", accuracy_score(y_test, my_predictions))

conf_mat = confusion_matrix(y_test, my_predictions)

ConfusionMatrixDisplay.from_estimator(my_classifier, X_test, y_test, 
                                      display_labels=["Ball", "Called Strike", "Hit", "Foul", "Swing Strike", "Out"])

st.pyplot()