#PYTHON CODE TO PERFORM A SIMPLE LINEAR REGRESSION IN VSCODE
How Hours Spent on Study affects Grades.
#Basic Installations:
For regression graph:
python -m pip install statsmodels

pip install pandas

pip install statsmodels

pip install Numpy



#Verify the Python Interpreter in your IDE
In VS Code:
1.	Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P).
2.	Type Python: Select Interpreter.
3.	Select the Python interpreter 
Python Code (App.py)
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
Run The App from Command line
Type: python App.py

Screenshot of Output
 
--- Regression Results (Summary) ---
                            OLS Regression Results
==============================================================================
Dep. Variable:                 Grades   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 3.845e+30
Date:                Sun, 14 Dec 2025   Prob (F-statistic):          5.12e-120
Time:                        16:50:09   Log-Likelihood:                 300.89
No. Observations:                  10   AIC:                            -597.8
Df Residuals:                       8   BIC:                            -597.2
Df Model:                           1
Covariance Type:            nonrobust
============================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------
const          	      50.0000   1.58e-14   3.16e+15      0.000      50.000      50.000
Hours_Study     5.0000   2.55e-15   1.96e+15      0.000       5.000       5.000
==============================================================================
Omnibus:                        1.179   Durbin-Watson:                   0.153
Prob(Omnibus):             0.555   Jarque-Bera (JB):                0.710
Skew:                              -0.206   Prob(JB):                              0.701
Kurtosis:                         1.761   Cond. No.                               13.7
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.     
R-squared: 1.0000
Intercept (c): 50.00
Slope (m): 5.00
Regression Equation: Grades = 5.00 * Hours_Study + 50.00
Key Interpretation
Here are the most important takeaways from the results of the code above:
Metric	Value	Interpretation
R-squared	1.000	This is a perfect fit. 100% of the variability in the Grades is explained by the Hours Spent on Study. (This is expected with the perfectly linear sample data used).
Coeff (const)	50.000	This is the Y-intercept. It means a student who studies 0 hours is predicted to get a grade of 50.
Coeff (Hours_Study)	5.000	This is the slope. It means for every 1 additional hour spent studying, the grade is predicted to increase by 5 points.
P-value	0.000	This very low p-value (for both the slope and the F-statistic) indicates that the relationship between study hours and grades is statistically significant.

Pay attention: The core piece of evidence that leads to this conclusion is the positive sign of the slope (or coefficient) for the Hours_Study variable.
Coefficient for Hours_Study: +5.00
Since the coefficient is a positive number, it means the two variables move in the same direction.
In simple terms, more hours spent studying predicts higher grades, and fewer hours spent studying predicts lower grades, demonstrating a strong positive correlation.

