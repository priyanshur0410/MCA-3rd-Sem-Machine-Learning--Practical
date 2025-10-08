1. Install the Latest Version of Python
Go to: https://www.python.org/downloads/

Download and run the installer.

Ensure you check the box “Add Python to PATH” before installing.

2. Set Up a Virtual Environment

python -m venv ds_env
cd ds_env
pip install numpy pandas matplotlib scikit-learn jupyter

3. Install Jupyter Notebook & Run It

pip install notebook
jupyter notebook

print("Jupyter is working!")

4. Install Anaconda and Launch Jupyter

Download from: https://www.anaconda.com/products/distribution

Install it.

Open Anaconda Navigator, click Launch on Jupyter Notebook.

5. Create a Python File in PyCharm or VS Code

print("Hello, Data Science")

6. Create a Jupyter Notebook with 3 Markdown and 3 Code Cells

import numpy as np
print("NumPy imported")

import pandas as pd
print("Pandas imported")

import matplotlib.pyplot as plt
print("Matplotlib imported")

7. List and install the following libraries: NumPy, Pandas, Matplotlib, Scikit-learn.
pip install numpy pandas matplotlib scikit-learn

8. Check the installed versions of NumPy, Pandas, and Matplotlib using Python.

import numpy as np
import pandas as pd
import matplotlib

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)

9. Configure Jupyter Notebook to show plots inline using %matplotlib inline.

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])

10. Write a Python script to verify successful import of NumPy, Pandas, Matplotlib, and Scikit-
learn.

try:
    import numpy
    import pandas
    import matplotlib
    import sklearn
    print("All libraries imported successfully!")
except ImportError as e:
    print("Import failed:", e)

11. Load Iris Dataset from UCI

import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv(url, names=columns)

print(iris.head())

12. Load a CSV File

df = pd.read_csv("your_file.csv")
print(df.head())

13. Dataset Dimensions & Column Names

print("Shape:", df.shape)
print("Columns:", df.columns)

14. Data Types of Columns

print(df.dtypes)

15. Dataset Info

print(df.info())

16. Load Excel File & Describe

df_excel = pd.read_excel("your_excel_file.xlsx")
print(df_excel.describe())

17. Load Diabetes Dataset (Scikit-learn)

from sklearn.datasets import load_diabetes
import pandas as pd

diabetes = load_diabetes()
df_diabetes = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

print(df_diabetes.head())

18. Save DataFrame to CSV and Read Back

df.to_csv("saved_data.csv", index=False)
df_loaded = pd.read_csv("saved_data.csv")
print(df_loaded.head())

19. Show First and Last 10 Rows

print("Head:\n", df.head(10))
print("Tail:\n", df.tail(10))

20. Read Dataset from Online Source

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df_url = pd.read_csv(url)
print(df_url.head())

