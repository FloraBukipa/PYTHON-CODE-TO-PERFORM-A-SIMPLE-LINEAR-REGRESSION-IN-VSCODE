import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Create the Data (Using pandas) ---
# Create a dictionary with your sample data
data = {
    'Hours_Study': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Grades': [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
}

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(data)

print("--- Data Snapshot (Head) ---")
print(df.head())
print("\n")

# --- 2. Prepare Variables for Statsmodels ---
# Define the dependent variable (Y, what we are predicting - Grades)
Y = df['Grades']

# Define the independent variable (X, what we are using to predict - Hours_Study)
X = df['Hours_Study']

# Add a constant (intercept) term to the independent variable.
# Statsmodels requires this explicitly for the regression equation: Y = mX + c
X = sm.add_constant(X)

print("--- Independent Variable (X) with Constant ---")
print(X.head())
print("\n")

# --- 3. Run the Linear Regression (Using statsmodels.api) ---
# Create the OLS (Ordinary Least Squares) model object
model = sm.OLS(Y, X)

# Fit the model to the data
results = model.fit()

# --- 4. Print the Results ---
print("--- Regression Results (Summary) ---")
print(results.summary())
print("\n")

# --- 5. Extract and Interpret Key Results ---

# Get the R-squared value
r_squared = results.rsquared

# Get the coefficients (Intercept and Slope)
intercept = results.params['const']
slope = results.params['Hours_Study']

print(f"R-squared: {r_squared:.4f}")
print(f"Intercept (c): {intercept:.2f}")
print(f"Slope (m): {slope:.2f}")

# The regression equation is: Grades = {slope} * Hours_Study + {intercept}
print(f"\nRegression Equation: Grades = {slope:.2f} * Hours_Study + {intercept:.2f}")


# --- 6. Visualize the Data and the Regression Line (Using matplotlib) ---
plt.figure(figsize=(10, 6))

# Scatter plot of the actual data points
plt.scatter(df['Hours_Study'], df['Grades'], color='blue', label='Actual Data Points')

# Plot the regression line
# Use the fitted values (the predicted Y values) for the line
plt.plot(df['Hours_Study'], results.fittedvalues, color='red', linewidth=2, label=f'Regression Line: Grades = {slope:.2f}X + {intercept:.2f}')

# Add labels and title
plt.title('Linear Regression: Study Hours vs. Grades')
plt.xlabel('Hours Spent on Study (X)')
plt.ylabel('Grades (Y)')
plt.legend()
plt.grid(True)
plt.show()