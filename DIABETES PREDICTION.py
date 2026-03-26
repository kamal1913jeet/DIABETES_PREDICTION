import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import tkinter as tk

import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv("FILES FOR PROJECT/diabetes.csv")
print(dataset)
#                                    -------------------DATA REFINEMENT AND VALIDATION--------------------
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset['Outcome'].value_counts())

#                                   --------------------GROUPING DIABETIC AND NON DIABETIC --------------
print(dataset.groupby('Outcome').mean())

#                                    -------------------SEAPARTING RESULT AND FACTORS---------------------
x = dataset.drop( 'Outcome' , axis = 1)
y = dataset['Outcome']
print(x)
print(y)

#                                     ---------------------DATA STANDARDIZATION---------------
scaler = StandardScaler()
scaler.fit(x)
std = scaler.transform(x)                       #data in same range
print(std)

#                                     ----------------store standardized data-------------------
x = std                                    #feed all data in x
y = dataset['Outcome']
print(x)
print(y)
#                                 --------------20% data for testing----------------
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.03 , stratify = y , random_state = 2)
print(x.shape , x_train.shape, x_test.shape)

#                    --------------------TRAINING THE MODEL ------------
cls = svm.SVC(kernel = 'linear')
print(cls.fit(x_train , y_train))

#                       ------------ training data EVALUATION--------------
x_pre = cls.predict(x_train)
train_acc = accuracy_score(x_pre , y_train)
print(train_acc)
#                       ---------------testing data evaluation -------------------
x_pre = cls.predict(x_test)
test_acc = accuracy_score(x_pre , y_test)
print(test_acc)

#                      -----------------prediction--------------
# input_data = (3,78,50,32,88,31,0.248,26)
def prediction ():
    data = np.array([[
        int(preg_entry.get()),
        float(glucose_entry.get()),
        float(bp_entry.get()),
        float(skin_entry.get()),
        float(insulin_entry.get()),
        float(bmi_entry.get()),
        float(dpf_entry.get()),
        int(age_entry.get())
    ]])

    input_res = data.reshape(1, -1)  # reshaping to avoid confusion of existing data for instance
    #                   ---------------standardize input data--------------------
    input_std = scaler.transform(input_res)
    print(input_std)

    pred = cls.predict(input_res)
    print(pred)

    if pred[0] == 0:
        output_label.config(text = "The person in not diabetic ." , fg = "green")
    else:
        output_label.config(text = "The person is diabetic ." , fg = "red")

import threading
def r_predict():
    threading.Thread(target=prediction).start()

root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("400x500")

tk.Label(root, text="Pregnancies").pack()
preg_entry = tk.Entry(root)
preg_entry.pack()
# preg_entry.insert(0, "2")

tk.Label(root, text="Glucose").pack()
glucose_entry = tk.Entry(root)
glucose_entry.pack()
# glucose_entry.insert(0, "120")

tk.Label(root, text="Blood Pressure").pack()
bp_entry = tk.Entry(root)
bp_entry.pack()
# bp_entry.insert(0, "70")

tk.Label(root, text="Skin Thickness").pack()
skin_entry = tk.Entry(root)
skin_entry.pack()

tk.Label(root, text="Insulin").pack()
insulin_entry = tk.Entry(root)
insulin_entry.pack()

tk.Label(root, text="BMI").pack()
bmi_entry = tk.Entry(root)
bmi_entry.pack()
# bmi_entry.insert(0, "28.5")

tk.Label(root, text="DPF").pack()
dpf_entry = tk.Entry(root)
dpf_entry.pack()

tk.Label(root, text="Age").pack()
age_entry = tk.Entry(root)
age_entry.pack()
# age_entry.insert(0, "30")

# Button
tk.Button(root, text="Predict", command=prediction).pack(pady=10)

# Output
output_label = tk.Label(root, text="")
output_label.pack()

root.mainloop()

