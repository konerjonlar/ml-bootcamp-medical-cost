# Health Insurance Cost Prediction Project

## Description
This project aims to estimate the approximate cost of a person's health insurance based on various variables. The project includes data preprocessing, model selection, hyperparameter optimization, and model evaluation.

## Installation
1. Clone this repository to your local machine.
2. Install the required libraries by running: `pip install -r requirements.txt`
3. Download the `insurance.csv` dataset and place it in the project directory.

## Usage
1. Open the Jupyter Notebook file `project.ipynb`.
2. Follow the step-by-step instructions to analyze the data, preprocess it, select and train regression models, optimize hyperparameters, and evaluate the models.
3. Run each cell in the notebook to execute the code and see the results.

## Description For Each Library/Module
- **Pandas**:  
  Used for data analysis and manipulation.
  - `pd.read_csv()`: 
    - Used to read CSV files and create DataFrames.
  - DataFrame:
    - Data structure representing data in a tabular format with columns and rows.
  - Series:
    - One-dimensional array representing a column in a DataFrame.

- **Numpy**:  
  Used for numerical computations and array manipulations.
  - `np.array()`:
    - Used to create NumPy arrays that represent vectors and matrices.

- **Seaborn**:
  Used for data visualization.
  - `sns.countplot()`:
    - Used to visualize the count of categorical data.

- **Matplotlib.Pyplot**:  
  Used for plotting graphs.
  - `plt.figure()`:
    - Creates a new plot figure.
  - `plt.title()`:
    - Sets the title of the plot.
  - `plt.xlabel()`:
    - Sets the label for the X-axis.
  - `plt.ylabel()`:
    - Sets the label for the Y-axis.
  - `plt.legend()`:
    - Adds legends to the plot.

- **Sklearn.Model_Selection**:  
  Used for model training and validation.
  - `train_test_split()`:
    - Splits data into training and testing sets.

- **Sklearn.Preprocessing**:  
  Used for data preprocessing and scaling.
  - `LabelEncoder()`:
    - Converts categorical data into numerical values.
  - `OneHotEncoder()`:
    - Converts categorical data using one-hot encoding.
  - `StandardScaler()`:
    - Scales and standardizes data.

- **Sklearn.Linear_Model**:  
  Contains classes for linear regression models.
  - `LinearRegression()`:
    - Creates a simple linear regression model.

- **Sklearn.Ensemble**:  
  Contains classes for ensemble models.
  - `RandomForestRegressor()`:
    - Creates a random forest regression model.

- **Sklearn.Svm**:  
  Contains classes for Support Vector Machines (SVM) models.
  - `SVR()`:
    - Creates a support vector regression model.

- **Sklearn.Metrics**:  
  Used for evaluating model performance.
  - `mean_squared_error()`:
    - Calculates Mean Squared Error (MSE).
  - `mean_absolute_error()`:
    - Calculates Mean Absolute Error (MAE).

- **Sklearn.Model_Selection**:  
  Used for model selection and validation.
  - `cross_val_score()`:
    - Evaluates model performance using cross-validation.

    
## Dataset 
The dataset used in this project (`insurance.csv`) contains information about individuals including age, sex, BMI, children, smoker status, region, and insurance charges.


## Dataset Information

- Observations: 1338
- Variables: 7
- Categorical Columns (cat_cols): 4
- Numeric Columns (num_cols): 3
- Categorical Variables that Behave Like Numeric (cat_but_car): 0
- Numeric Variables that Behave Like Categorical (num_but_cat): 1

---

## Data Exploration and Analysis

**Data Size:**
This indicates the dimensions of the dataset, where it has 1338 rows and 7 columns.

**Data Types:**
It shows the data types of each column. For example, 'age' and 'children' are represented as integers, while 'sex', 'smoker', and 'region' are represented as objects (likely strings), and 'bmi' and 'charges' are represented as floating-point numbers.

**Total Null Values:**
It states that there are no missing values (null values) in the dataset.

**General Statistics:**

|     |      age |       bmi |   children |      charges |
|:---:| --------:| ---------:| ----------:| -----------:|
| **Count** | 1338 | 1338 | 1338 | 1338 |
| **Mean** | 39.21 | 30.66 | 1.09 | 13270.42 |
| **Std** | 14.05 | 6.10 | 1.21 | 12110.01 |
| **Min** | 18 | 15.96 | 0 | 1121.87 |
| **25%** | 27 | 26.30 | 0 | 4740.29 |
| **50% (Median)** | 39 | 30.40 | 1 | 9382.03 |
| **75%** | 51 | 34.69 | 2 | 16639.91 |
| **Max** | 64 | 53.13 | 5 | 63770.43 |

**Children Value Counts:**
This segment displays the count of each unique value in the 'children' feature. For example, there are 574 individuals with 0 children, 324 with 1 child, 240 with 2 children, 157 with 3 children, 25 with 4 children, and 18 with 5 children.

**Smoker Value Counts:**
It presents the count of 'yes' and 'no' values in the 'smoker' feature. There are 1064 non-smokers and 274 smokers in the dataset.

**Region Value Counts:**
This part shows the distribution of individuals across different regions. The 'southeast' region has 364 individuals, 'southwest' has 325, 'northwest' has 325, and 'northeast' has 324.

Each of these sections provides insights into the dataset's characteristics, distributions, and statistics, helping us understand the data better.


## Project Structure
- `insurance_prediction.ipynb`: Jupyter Notebook containing the end-to-end data science pipeline.
- `insurance.csv`: Dataset used for the analysis and modeling.


---

## Analysis Steps
1. **Exploratory Data Analysis (EDA):**
   - Examine the distribution of BMI.
   - Investigate the relationship between "smoker" and "charges".
   - Explore the relationship between "smoker" and "region".
   - Analyze the connection between "bmi" and "sex".
   - Identify the "region" with the most "children".
   - Study the relationship between "age" and "bmi".
   - Explore the correlation between "bmi" and "children".
   - Check for outliers in the "bmi" variable.
   - Examine the relationship between "bmi" and "charges".
   - Visualize the relationship between "region", "smoker", and "bmi" using a bar plot.

2. **Data Preprocessing:**
   - Perform Label Encoding and One-Hot Encoding for categorical variables.
   - Split the dataset into training and testing sets.
   - Scale the features using Standard Scaling.

3. **Model Selection:**
   - Choose regression models (Linear Regression, Random Forest Regressor, SVR).
   - Train the models using the preprocessed data.
   - Evaluate the models using cross-validation.

4. **Hyperparameter Optimization:**
   - Tune the hyperparameters of the selected model using GridSearchCV.

5. **Model Evaluation:**
   - Evaluate the optimized model using regression metrics (MSE, MAE, RMSE).
---
## General Overview

### About the Dataset

The "insurance.csv" dataset contains information about health insurance costs. It includes various attributes of insured individuals along with their associated healthcare charges. This dataset covers factors such as smoking habits, age, gender, region, BMI (Body Mass Index), and number of children, which can impact health insurance costs.

### Dataset Contents

The dataset typically includes the following columns:

- `age`: Represents the age of the insured individual.
- `sex`: Indicates the gender of the insured individual (female or male).
- `bmi`: Represents the Body Mass Index (BMI) of the insured individual.
- `children`: Indicates the number of children the insured individual has.
- `smoker`: Indicates whether the insured individual is a smoker or not (yes or no).
- `region`: Represents the region where the insured individual resides (northeast, northwest, southeast, southwest).
- `charges`: Represents the healthcare charges of the insured individual.

### Purpose

This dataset can be used to understand and analyze factors influencing health insurance costs. For example, how do insurance costs differ between smokers and non-smokers? What is the impact of age, gender, BMI, etc., on insurance costs? This dataset can help answer such questions and assist in predicting approximate health insurance costs.

---

## Authors
- [Orhan Cansu](https://www.linkedin.com/in/orhan-cansu/)
- [Meryem Arslan](https://github.com/mrymarsln)
- [Murat Rüzgar Deniz](https://github.com/Ruzgarte)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
