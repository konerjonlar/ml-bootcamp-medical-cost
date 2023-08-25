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

## "insurance.csv" Kullanım Kılavuzu

Bu kılavuz, "insurance.csv" adlı veri setini kullanarak adım adım nasıl temel veri analizi yapabileceğinizi göstermektedir.

### Adım 1: Veri Setini Yüklemek

1. Öncelikle, Python programlama dilini kullanabilmek için bir Python ortamına sahip olduğunuzdan emin olun.
2. "insurance.csv" adlı veri setini indirin ve belirli bir klasöre kaydedin.
3. İlgili Python kütüphanelerini içe aktarın:

```python
import pandas as pd
```

4. Veri setini yükleyin:

```python
data = pd.read_csv("insurance.csv")
```

### Adım 2: Veri Setini İncelemek

1. Veri setinin ilk birkaç satırını görmek için:

```python
print(data.head())
```

2. Veri setinin sütunlarını ve veri tiplerini görüntülemek için:

```python
print(data.info())
```

3. Veri setinin özet istatistiklerini almak için:

```python
print(data.describe())
```

### Adım 3: Temel Veri Analizi

1. "BMI" sütununun dağılımını incelemek için:

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data['bmi'], bins=20, kde=True)
plt.title('BMI Dağılımı')
plt.xlabel('BMI')
plt.ylabel('Frekans')
plt.show()
```

2. "Sigara içen" bireylerin "suçlamalar" ile ilişkisini incelemek için:

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='charges', y='smoker', data=data)
plt.title('Sigara İçen ve Suçlamalar İlişkisi')
plt.xlabel('Suçlamalar')
plt.ylabel('Sigara İçen (0: Hayır, 1: Evet)')
plt.show()
```

3. "Sigara içen" ve "bölge" arasındaki ilişkiyi incelemek için:

```python
plt.figure(figsize=(10, 6))
sns.countplot(x='region', hue='smoker', data=data)
plt.title('Sigara İçme Durumu ve Bölgeler')
plt.xlabel('Bölge')
plt.ylabel('Birey Sayısı')
plt.legend(title='Sigara İçen')
plt.show()
```

4. "BMI" ve "cinsiyet" arasındaki ilişkiyi analiz etmek için:

```python
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='bmi', data=data)
plt.title('BMI ve Cinsiyet İlişkisi')
plt.xlabel('Cinsiyet')
plt.ylabel('BMI')
plt.show()
```

5. En çok "çocuk"un bulunduğu "bölgeyi" belirlemek için:

```python
most_children_region = data.groupby('region')['children'].sum().idxmax()
print(f'En çok çocuğun bulunduğu bölge: {most_children_region}')
```

Bu adımları takip ederek, "insurance.csv" veri setini yükleyebilir, inceleyebilir ve temel veri analizi adımlarını uygulayabilirsiniz. Bu kılavuz, veri setinizi daha iyi anlamak ve önemli özellikleri keşfetmek için başlangıç ​​noktanız olabilir.

---

## Genel Bakış

### Veri Seti Hakkında

"insurance.csv" veri seti, sağlık sigortası maliyetleri hakkında bilgi içeren bir veri setidir. Bu veri seti, sigortalı bireylerin çeşitli özelliklerini ve sağlık sigortası maliyetlerini içerir. Veri seti, sigara içme alışkanlığı, yaş, cinsiyet, bölge, BMI (Vücut Kitle İndeksi), çocuk sayısı gibi faktörlerin sağlık sigortası maliyetleri üzerindeki etkilerini anlamak için kullanılabilir.

### Veri Setinin İçeriği

Veri seti genellikle aşağıdaki sütunları içerir:

- `age`: sigortalının yaşını temsil eder.
- `sex`: sigortalının cinsiyetini (kadın veya erkek) gösterir.
- `bmi`: sigortalının Vücut Kitle İndeksi (BMI) değerini temsil eder.
- `children`: sigortalının sahip olduğu çocuk sayısını ifade eder.
- `smoker`: sigortalının sigara içip içmediğini gösterir (yes veya no).
- `region`: sigortalının yaşadığı bölgeyi temsil eder (kuzeydoğu, kuzeybatı, güneydoğu, güneybatı).
- `charges`: sigortalının sağlık sigortası maliyetlerini ifade eder.

### Amaç

Bu veri seti, sağlık sigortası maliyetlerini etkileyen faktörleri anlamak ve analiz etmek için kullanılabilir. Örneğin, sigara içenlerin sigorta maliyetleri sigara içmeyenlere göre nasıl değişiyor? Ya da yaş, cinsiyet, BMI gibi faktörlerin sigorta maliyetleri üzerindeki etkileri nelerdir? Bu tür soruları yanıtlamak için veri seti kullanılabilir. aynı zamanda kişinin sağlık sigortasının yaklaşık maliyetini tahmin etmektedir.

---

## Authors
- Orhan Cansu
- Meryem Arslan
- Murat Rüzgar Deniz

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
