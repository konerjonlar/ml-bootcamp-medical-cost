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

## Project Structure
- `insurance_prediction.ipynb`: Jupyter Notebook containing the end-to-end data science pipeline.
- `insurance.csv`: Dataset used for the analysis and modeling.

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

## Authors
- Orhan Cansu
- Meryem Arslan
- Rüzgar

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
