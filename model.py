import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "hours": [1, 2, 3, 4, 5],
    "marks": [20, 40, 50, 60, 80]
}

daf = pd.DataFrame(data)

X = daf[["hours"]]
y = daf["marks"]

model = LinearRegression()
model.fit(X, y)

print("Prediction for 6 hours:", model.predict([[6]]))