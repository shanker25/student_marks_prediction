import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "hours": [1, 2, 3, 4, 5],
    "marks": [20, 40, 50, 60, 80]
}

df = pd.DataFrame(data)

X = df[["hours"]]
y = df["marks"]

model = LinearRegression()
model.fit(X, y)
User_input=int(input("Enter the total study hours :"))
print("Prediction for",User_input," hours:", model.predict([[User_input]]))

