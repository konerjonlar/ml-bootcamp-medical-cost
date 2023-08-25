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
This section displays the names of the columns (features) in the dataset.

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
Here, summary statistics of the dataset are presented:
- `count`: The number of non-null values in each feature.
- `mean`: The average value of each feature.
- `std`: The standard deviation of each feature.
- `min`: The minimum value of each feature.
- `25%`, `50%`, `75%`: The quartiles of each feature.
- `max`: The maximum value of each feature.

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
Elbette, aşağıda adım adım "sigorta.csv" veri setini kullanım kılavuzu şeklinde bir README dosyası örneği yer almaktadır. Bu rehberi takip ederek veri setinizi yüklemek, incelemek ve analiz etmek için adım adım ilerleyebilirsiniz.

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
## Genel Bakış

### Veri Seti Hakkında

"sigorta.csv" veri seti, sağlık sigortası maliyetleri hakkında bilgi içeren bir veri setidir. Bu veri seti, sigortalı bireylerin çeşitli özelliklerini ve sağlık sigortası maliyetlerini içerir. Veri seti, sigara içme alışkanlığı, yaş, cinsiyet, bölge, BMI (Vücut Kitle İndeksi), çocuk sayısı gibi faktörlerin sağlık sigortası maliyetleri üzerindeki etkilerini anlamak için kullanılabilir.

### Veri Setinin İçeriği

Veri seti genellikle aşağıdaki sütunları içerir:

- `age`: Sigortalının yaşını temsil eder.
- `sex`: Sigortalının cinsiyetini (kadın veya erkek) gösterir.
- `bmi`: Sigortalının Vücut Kitle İndeksi (BMI) değerini temsil eder.
- `children`: Sigortalının sahip olduğu çocuk sayısını ifade eder.
- `smoker`: Sigortalının sigara içip içmediğini gösterir (yes veya no).
- `region`: Sigortalının yaşadığı bölgeyi temsil eder (kuzeydoğu, kuzeybatı, güneydoğu, güneybatı).
- `charges`: Sigortalının sağlık sigortası maliyetlerini ifade eder.

### Amaç

Bu veri seti, sağlık sigortası maliyetlerini etkileyen faktörleri anlamak ve analiz etmek için kullanılabilir. Örneğin, sigara içenlerin sigorta maliyetleri sigara içmeyenlere göre nasıl değişiyor? Ya da yaş, cinsiyet, BMI gibi faktörlerin sigorta maliyetleri üzerindeki etkileri nelerdir? Bu tür soruları yanıtlamak için veri seti kullanılabilir.

### Veri Kaynağı ve Lisans

"sigorta.csv" veri setinin tam olarak hangi kaynaktan geldiğini veya hangi lisansa tabi olduğunu belirtmediğinizi gözlemledim. Eğer veri setinin kaynağı veya lisansı hakkında bilgi almak isterseniz, veri setini oluşturan kaynağa veya veri sağlayıcıya başvurmanız gerekmektedir.

Bu genel bakış, "sigorta.csv" veri seti hakkında temel bilgiler sağlamaktadır. Eğer projenizde bu veri setini kullanmayı planlıyorsanız, analizlerinizde ve sonuçlarınızda bu faktörleri dikkate alabilirsiniz.
## Authors
- Orhan Cansu
- Meryem Arslan
- Rüzgar

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
