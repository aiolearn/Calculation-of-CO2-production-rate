<h1 align="center">Welcome to Calculation of CO2 production rate Project 👋</h1>

# Calculation of CO2 production rate

In this project, we use machine learning to write a program that shows us the amount of CO2 produced by the car based on the data it has.

## Modules

We use many modules in this project

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

## Usage

Read data from csv file

```python
df = pd.read_csv('co2.csv')
```

Separate dependent and independent variables

```python
x = df.drop("out1", axis = 1)
y = df.out1
```

Show file
```python
df.head()
df.describe()
sns.countplot(x = 'out1', data = df)
plt.subplots(figsize = (9, 9))
sns.heatmap(df.corr(), annot=True)
```

Split data into features (x) and output (y)
```python
x = df.drop("out1", axis = 1)
x = x.drop("fuelcomb", axis = 1)
x = x.drop("cylandr", axis = 1)
y = df.out1
```
Dividing data into training and test categories
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
```

Display the training and test category

```python
x_train
x_test
```

Create a linear regression model using the LinearRegression class from the linear_model module

```python
reg_linear = linear_model.LinearRegression()
```

Training a linear regression model with x_train training data and y_train training labels
```python
reg_linear.fit(x_train, y_train)
```

Output prediction for test and display data
```python
y_tets_pred = reg_linear.predict(x_test)
y_tets_pred
```
Show the scatterplot using the test data and test labels, in blue
```python
plt.scatter(x_test, y_test, color = 'blue')
plt.show
```
A graph of prediction errors compared to reality
```python
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_test, y_tets_pred, color = 'black', linewidth = 2)
plt.show
```

Calculate the mean square error
```python
sm.mean_squared_error(y_test, y_tets_pred)
```

Prediction of test data labels

```python
y_test_pred = reg_linear.predict(X_test)
```
Applying the model to a test sample
```python
test=np.array([[3]])
```
Print the output of the model for the test sample
```python
khoroji = reg_linear.predict(test)
print(khoroji)
```
## Result
This project was written by Majid Tajanjari and the Aiolearn team, and we need your support!❤️

# محاسبه نرخ تولید CO2
در این پروژه از یادگیری ماشینی برای نوشتن برنامه ای استفاده می کنیم که میزان CO2 تولید شده توسط خودرو را بر اساس داده هایی که دارد به ما نشان می دهد.

## ماژول ها

ما در این پروژه از ماژول های زیادی استفاده می کنیم

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

## نحوه استفاده

خواندن داده ها از فایل csv

```python
df = pd.read_csv('co2.csv')
```

متغیرهای وابسته و مستقل را از هم جدا کنید

```python
x = df.drop("out1", axis = 1)
y = df.out1
```

نمایش فایل
```python
df.head()
df.describe()
sns.countplot(x = 'out1', data = df)
plt.subplots(figsize = (9, 9))
sns.heatmap(df.corr(), annot=True)
```

تقسیم داده ها به ویژگی های (x) و خروجی (y)
```python
x = df.drop("out1", axis = 1)
x = x.drop("fuelcomb", axis = 1)
x = x.drop("cylandr", axis = 1)
y = df.out1
```
تقسیم داده ها به دسته های آموزشی و آزمایشی
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
```

نمایش دسته آموزشی و آزمون

```python
x_train
x_test
```

یک مدل رگرسیون خطی با استفاده از کلاس LinearRegression از ماژول linear_model ایجاد کنید

```python
reg_linear = linear_model.LinearRegression()
```

آموزش مدل رگرسیون خطی با داده های آموزشی x_train و برچسب های آموزشی y_train
```python
reg_linear.fit(x_train, y_train)
```

پیش بینی خروجی برای داده های تست و نمایش```python
y_tets_pred = reg_linear.predict(x_test)
y_tets_pred
```
نمودار پراکندگی را با استفاده از داده های تست و برچسب های آزمایشی به رنگ آبی نشان دهید
```python
plt.scatter(x_test, y_test, color = 'blue')
plt.show
```
نموداری از خطاهای پیش بینی در مقایسه با واقعیت
```python
plt.scatter(x_test, y_test, color = 'blue')
plt.plot(x_test, y_tets_pred, color = 'black', linewidth = 2)
plt.show
```
میانگین مربعات خطا را محاسبه کنید
```python
sm.mean_squared_error(y_test, y_tets_pred)
```

پیش بینی برچسب داده های آزمایشی

```python
y_test_pred = reg_linear.predict(X_test)
```
اعمال مدل بر روی نمونه آزمایشی
```python
test=np.array([[3]])
```
خروجی مدل را برای نمونه آزمایشی چاپ کنید
```python
khoroji = reg_linear.predict(test)
print(khoroji)
```
## نتیجه
این پروژه توسط مجید تجن جاری و تیم Aiolearn نوشته شده است و ما به حمایت شما نیازمندیم!❤️