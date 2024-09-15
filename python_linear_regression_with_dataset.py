import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("mobile.csv")
print(df.head())

standard_scaler_data = MinMaxScaler()
df["Price"] = standard_scaler_data.fit_transform(df[["Price"]])

standard_scaler_data = StandardScaler()
df["Price"] = standard_scaler_data.fit_transform(df[["Price"]])

x = df[["Ratings", "Mobile_Size"]]
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
# Print the test data
print(x_test)
print(y_test)

prediction = model.predict(x_test)
print(prediction)

r2_score_output = r2_score(y_test,prediction)
print(r2_score_output)

print("---------------------------------------------")

mse_output = mean_squared_error(y_test, prediction)
print(mse_output)