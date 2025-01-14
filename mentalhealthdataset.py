import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:/Users/akhaja@cps.edu/Documents/Intro to AI/a6-project-yf-ak/a6-project-yf-ak/mentalhealth.csv")

# Clean and prepare the data
data['Mental_Health_Challenge'] = np.where(
    (data['Do you have Depression?'] == 'Yes') |
    (data['Do you have Anxiety?'] == 'Yes') |
    (data['Do you have Panic?'] == 'Yes'), 1, 0
)

# Convert CGPA ranges to numeric (using midpoints)
data['CGPA'] = data['What is your CGPA?'].str.split('-').apply(lambda x: (float(x[0]) + float(x[1])) / 2)

# Select features (X) and target (y)
X = data['Mental_Health_Challenge'].values.reshape(-1, 1)
y = data['CGPA'].values

# Create and train the linear regression model
model = LinearRegression().fit(X, y)

# Find coefficient, intercept, and R-squared
coef = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(X, y), 2)

# Print results
print(f"Linear Equation: CGPA = {coef} * Mental_Health_Challenge + {intercept}")
print(f"R-Squared value: {r_squared}")

# Predict CGPA for a student with mental health challenges
mental_health_status = 1
prediction = model.predict([[mental_health_status]])
print(f"Predicted CGPA for a student with mental health challenges: {prediction[0]:.2f}")

# Visualize data and line of best fit
plt.scatter(X, y, c='purple', label='Data Points')
plt.plot(X, model.predict(X), c='red', label='Line of Best Fit')

plt.xlabel('Mental Health Challenge (1 = Yes, 0 = No)')
plt.ylabel('CGPA')
plt.title('CGPA vs Mental Health Challenge')
plt.legend()
plt.show()
