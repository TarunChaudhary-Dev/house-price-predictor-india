import numpy as np
import pandas as pd

df = pd.read_csv("house_cleaned.csv")
df = df.dropna()
x = df[["area","bedRoom","bathroom"]].values
y = df["price"].values

x_max = x.max(axis = 0)
y_max = y.max()
x = x / x_max
y = y / y_max

w = np.zeros(3)
b = 0
alpha = 0.01
m = len(x)

for i in range(1001):
    y_hat = np.dot(x , w) + b
    
    dw = (1/m) * np.dot(x.T , (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)

    w = w - alpha * dw
    b = b - alpha * db

print(f"w =  {w}")
print(f"b =  {b}")
# print(f"sample-y {y[0]} , sample-x {x[0]}")

area = float(input("Area (Sqft): "))
bedrooms = float(input("BedRooms: "))
bathrooms = float(input("Bathrooms: "))

new_x = np.array([area , bedrooms , bathrooms])

new_x_simplified = new_x / x_max

y_hat_new = np.dot(new_x_simplified , w ) + b

y_hat_final = y_hat_new * y_max

print(f"Price = {round(y_hat_final)}Cr.")
