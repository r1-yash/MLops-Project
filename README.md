# Student Performance Predictor

An end-to-end machine learning web application that predicts a student's mathematics score based on demographic and academic inputs. Built with a modular ML pipeline and deployed on AWS using a full CI/CD setup.

---

## Live Demo
[StudentPerformance-env.eba-hzp4ytyp.us-east-1.elasticbeanstalk.com](http://StudentPerformance-env.eba-hzp4ytyp.us-east-1.elasticbeanstalk.com)

---

## What It Does
Takes 7 student inputs — gender, race/ethnicity, parental education level, lunch type, test preparation course, reading score, and writing score — and predicts the expected mathematics score using a trained regression model.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14 |
| Web Framework | Flask + Gunicorn |
| ML Library | scikit-learn |
| Model | Linear Regression (best performer across 7 algorithms) |
| Deployment | AWS Elastic Beanstalk |
| CI/CD | AWS CodePipeline + GitHub |
| Infrastructure | Amazon Linux 2023, EC2 |

---

## ML Pipeline

1. **Data Ingestion** — Loads raw CSV, splits into train/test sets
2. **Data Transformation** — ColumnTransformer with StandardScaler for numerical features and OneHotEncoder for categorical features
3. **Model Training** — GridSearchCV across 7 algorithms (Linear Regression, Random Forest, Decision Tree, Gradient Boosting, AdaBoost, XGBoost, CatBoost)
4. **Best Model** — Linear Regression (~88% R² score)
5. **Serialisation** — Model and preprocessor saved as `.pkl` using pickle

---

## Dataset
The [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) dataset from Kaggle contains 1,000 student records with 8 columns. It was chosen for its clean structure, mix of categorical and numerical features, and suitability for regression practice and EDA.

---

## AWS Deployment

- **Elastic Beanstalk** — Python 3.14 on Amazon Linux 2023, served via Gunicorn
- **CodePipeline** — Auto-deploys on every push to the `main` branch
- **IAM** — Service role with `AWSElasticBeanstalkWebTier` policy
- **Config** — WSGI path set via `.ebextensions/python.config`

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/r1-yash/MLops-Project.git
cd MLops-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python application.py
```

Visit `http://localhost:5000`

---

## Pages

| Route | Description |
|---|---|
| `/` | Home page with project overview |
| `/predict` | Prediction form |
| `/why-dataset` | Why this dataset was chosen |
| `/learnings` | Key learnings from the project |

---

## Author
**Yash Singhal**  
B.Tech CSE, Bennett University  
[GitHub](https://github.com/r1-yash)