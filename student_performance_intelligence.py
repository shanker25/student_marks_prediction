import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Loading dataset
df = pd.read_csv("student_data.csv")

# Declaring Input and Output
X = df.drop("marks", axis=1)
y = df["marks"]

# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Training Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Test prediction
y_pred = model.predict(X_test)

# Error
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", round(mae,1))

# User input
print("Enter student details\n")
hours = int(input("Study hours: "))
sleep = int(input("Sleep hours: "))
phone = int(input("Phone usage: "))
attendance = int(input("Attendance: "))

# Converting input properly
input_df = pd.DataFrame([[hours, sleep, phone, attendance]], columns=X.columns)

# Prediction
prediction = model.predict(input_df)[0]
print("\n==============================")
print("🎯 STUDENT PERFORMANCE RESULT")
print("==============================")
print("Predicted Marks:", round(prediction, 2))

# Basic analysis
if prediction < 40:
    print("Risk: May fail")
elif prediction < 70:
    print("Average performance")
else:
    print("Good performance")

#Suggestions
print("Suggestions:\n")
if hours < 4:
    print("- Study a bit more")
if phone > 4:
    print("- Try reducing phone usage")
if sleep < 6:
    print("- Sleep properly")
if attendance < 75:
    print("- Attend more classes")
if hours >= 4 and phone <= 4 and sleep >= 6 and attendance >= 75:
    print("- You're doing good, keep it up")

# Graph
plt.scatter(df["hours"], df["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study vs Marks")
plt.show()