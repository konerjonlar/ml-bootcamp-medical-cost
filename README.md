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

## Data Set
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


**Display Columns:**

|     |      age |       bmi |   children |      charges |    sex   |     smoker   |    region   |
|:---:| --------:| ---------:| ----------:| -----------:| --------:| ---------:| ----------:|

**Data Size:**
This indicates the dimensions of the dataset, where it has 1338 rows and 7 columns.

**Data Types:**
It shows the data types of each column. For example, 'age' and 'children' are represented as integers, while 'sex', 'smoker', and 'region' are represented as objects (likely strings), and 'bmi' and 'charges' are represented as floating-point numbers.

**First 10 Rows:**
This part provides the first 10 rows of the dataset, showcasing the values for each feature. For instance, the first entry is a 19-year-old female with a BMI of 27.900, no children, being a smoker, residing in the southwest region, and having charges of 16884.92400.

**Last 4 Rows:**
Similarly, this section displays the last 4 rows of the dataset, showing information about the individuals in the final rows.

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

## "insurance.csv" Usage Guide

This guide demonstrates step-by-step how to perform basic data analysis using the "insurance.csv" dataset.

### Step 1: Loading the Dataset

1. Firstly, ensure that you have a Python environment set up to use the Python programming language.
2. Download the "insurance.csv" dataset and save it to a specific folder.
3. Import the necessary Python libraries:

```python
import pandas as pd
```

4. Load the dataset:

```python
data = pd.read_csv("insurance.csv")
```

### Step 2: Exploring the Dataset

1. To see the first few rows of the dataset:

```python
print(data.head())
```

2. To view the columns and data types of the dataset:

```python
print(data.info())
```

3. To obtain summary statistics of the dataset:

```python
print(data.describe())
```

### Step 3: Basic Data Analysis

1. To examine the distribution of the "BMI" column:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data['bmi'], bins=20, kde=True)
plt.title('BMI Distribution')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.show()
```

2. To investigate the relationship between "smoker" individuals and "charges":

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='charges', y='smoker', data=data)
plt.title('Smoker and Charges Relationship')
plt.xlabel('Charges')
plt.ylabel('Smoker (0: No, 1: Yes)')
plt.show()
```

3. To explore the relationship between "smoker" and "region":

```python
plt.figure(figsize=(10, 6))
sns.countplot(x='region', hue='smoker', data=data)
plt.title('Smoker Status and Regions')
plt.xlabel('Region')
plt.ylabel('Number of Individuals')
plt.legend(title='Smoker')
plt.show()
```

4. To analyze the relationship between "BMI" and "sex":

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='bmi', data=data)
plt.title('BMI and Sex Relationship')
plt.xlabel('Sex')
plt.ylabel('BMI')
plt.show()
```

5. To determine the region with the most children:

```python
most_children_region = data.groupby('region')['children'].sum().idxmax()
print(f'Region with the most children: {most_children_region}')
```

By following these steps, you can load, explore, and perform basic data analysis on the "insurance.csv" dataset. This guide can serve as a starting point for better understanding your dataset and uncovering important features.

---

## Authors
- Orhan Cansu
- Meryem Arslan
- Murat Rüzgar Deniz

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
