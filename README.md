# MLR_Student_score_prediction
# 🎓 Student Performance Predictor using Machine Learning

A Machine Learning web application that predicts a student's Performance Index based on academic and lifestyle factors using Multiple Linear Regression. The project includes model training, model serialization using Pickle, and deployment using Flask.

---

# 📌 Project Overview

This project predicts a student's Performance Index using the following inputs:

- 📚 Hours Studied
- 📈 Previous Scores
- 🏆 Extracurricular Activities
- 😴 Sleep Hours
- 📝 Sample Question Papers Practiced

The trained model is saved as a `.pkl` file and integrated into a Flask web application where users can enter values and receive predictions instantly.

---

# 🚀 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Flask
- Pickle
- HTML
- CSS
- Gunicorn

---

# 📂 Project Structure

```text
Student-Performance-Predictor/
│
├── app.py
├── MLR_Exam.pkl
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

---

# 🧠 Machine Learning Model Training

The model is trained using Multiple Linear Regression.

## Google Colab Training Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

class MLR_Exam:
    def __init__(self , path):
        self.path = path
        self.df = pd.read_csv(self.path)

        if 'Extracurricular Activities' in self.df.columns:
            self.df['Extracurricular Activities'] = self.df['Extracurricular Activities'].map(
                {'Yes':1,'No':0}
            )

        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        self.X_train , self.X_test, self.y_train , self.y_test = train_test_split(
            self.X ,
            self.y ,
            test_size=0.2 ,
            random_state=42
        )

    def training(self):
        self.reg = LinearRegression()
        self.reg.fit(self.X_train, self.y_train)

        print(
            f'Train Accuracy : '
            f'{r2_score(self.y_train,self.reg.predict(self.X_train))}'
        )

        print(
            f'Train Loss : '
            f'{np.sqrt(mean_squared_error(self.y_train,self.reg.predict(self.X_train)))}'
        )

    def testing(self):
        print(
            f"Test Accuracy : "
            f"{r2_score(self.y_test,self.reg.predict(self.X_test))}"
        )

        print(
            f'Test Loss : '
            f'{np.sqrt(mean_squared_error(self.y_test,self.reg.predict(self.X_test)))}'
        )

    def sample_input(
        self,
        Hours_studies,
        Previous_Score,
        Extra_Activities,
        Sleep,
        Sample_Q
    ):
        result = self.reg.predict([
            [
                Hours_studies,
                Previous_Score,
                Extra_Activities,
                Sleep,
                Sample_Q
            ]
        ])[0]

        print(f'Sample prediction was : {result}')

    def save_model(self, filename="MLR_Exam.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.reg, f)

        print(f"Model saved as {filename}")

    def load_model(self, filename="MLR_Exam.pkl"):
        with open(filename, 'rb') as f:
            self.reg = pickle.load(f)

        print(f"Model loaded from {filename}")


if __name__ == "__main__":

    obj = MLR_Exam("Student_Performance.csv")

    obj.training()

    obj.testing()

    obj.save_model()

    obj.load_model()

    obj.sample_input(
        9,
        70,
        1,
        7,
        5
    )
```

---

# 🔍 Code Explanation

## Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
```

These libraries are used for:

- NumPy → Mathematical operations
- Pandas → Data handling
- Matplotlib → Visualization
- Scikit-Learn → Machine Learning

---

## Step 2: Load Dataset

```python
self.df = pd.read_csv(self.path)
```

Reads the CSV file into a DataFrame.

---

## Step 3: Convert Categorical Values

```python
self.df['Extracurricular Activities'] = \
self.df['Extracurricular Activities'].map(
{
'Yes':1,
'No':0
}
)
```

Machine learning models work with numbers, so:

| Original | Converted |
|-----------|-----------|
| Yes | 1 |
| No | 0 |

---

## Step 4: Separate Features and Target

```python
self.X = self.df.iloc[:, :-1]
self.y = self.df.iloc[:, -1]
```

### Features (X)

- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Papers

### Target (y)

- Performance Index

---

## Step 5: Train-Test Split

```python
train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
```

Dataset is divided into:

- 80% Training Data
- 20% Testing Data

Purpose:

- Training data teaches the model.
- Testing data evaluates performance.

---

## Step 6: Train Linear Regression Model

```python
self.reg = LinearRegression()

self.reg.fit(
self.X_train,
self.y_train
)
```

The model learns relationships between input features and target values.

---

## Step 7: Calculate Training Accuracy

```python
r2_score(
self.y_train,
self.reg.predict(self.X_train)
)
```

R² Score measures prediction quality.

### Interpretation

| Score | Meaning |
|---------|----------|
| 1.0 | Perfect Prediction |
| 0.9+ | Excellent |
| 0.8+ | Good |
| <0.5 | Poor |

---

## Step 8: Calculate Training Loss

```python
np.sqrt(
mean_squared_error(
self.y_train,
self.reg.predict(self.X_train)
)
)
```

RMSE measures average prediction error.

Lower RMSE = Better Model

---

## Step 9: Evaluate on Test Data

```python
r2_score(
self.y_test,
self.reg.predict(self.X_test)
)
```

Checks how well the model performs on unseen data.

---

## Step 10: Save Model

```python
pickle.dump(
self.reg,
f
)
```

Creates:

```text
MLR_Exam.pkl
```

This file stores the trained model.

No retraining required after saving.

---

## Step 11: Load Saved Model

```python
pickle.load(f)
```

Loads the model into memory for prediction.

---

## Step 12: Make Prediction

```python
obj.sample_input(
9,
70,
1,
7,
5
)
```

Input:

```text
Hours Studied = 9
Previous Score = 70
Extracurricular = 1
Sleep Hours = 7
Sample Papers = 5
```

Output:

```text
Predicted Performance Index = XX.XX
```

---

# 🌐 Flask Integration

The trained model is loaded inside Flask.

```python
with open("MLR_Exam.pkl","rb") as f:
    m = pickle.load(f)
```

When the user submits the form:

```python
sol = m.predict(b)[0]
```

Prediction is generated and displayed on the webpage.

---

# 🛠 Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Flask:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 🧪 Sample Input

```text
Hours Studied = 9
Previous Scores = 70
Extracurricular Activities = Yes
Sleep Hours = 7
Sample Question Papers Practiced = 5
```

## Output

```text
Predicted Performance Index = 78.5
```

(Output varies based on trained model.)

---

# 🔮 Future Enhancements

- Deep Learning Models
- Student Dashboard
- Performance Analytics
- PDF Report Generation
- Cloud Deployment
- Database Integration
- Prediction History

---

# 👩‍💻 Author

### Chandana P

Data Science & Generative AI Enthusiast

GenAI Intern @ Vihara Tech

---

# ⭐ Support

If you found this project useful:

⭐ Star the Repository

🍴 Fork the Repository

📢 Share with Others

---

# 📜 License

MIT License

Copyright (c) 2026 Chandana P

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction.

---

# 🎯 Project Status

✅ Model Trained

✅ Model Saved as Pickle File

✅ Flask Backend Integrated

✅ Frontend Developed

✅ Prediction System Working

✅ Ready for Deployment

🚀 Production Ready
