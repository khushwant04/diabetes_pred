# Diabetes Prediction

Predicts whether a patient is diabetic from diagnostic measurements using a linear-kernel SVM.
Single-notebook learning project; an SVM was chosen because the dataset is small.

## Dataset

`data/diabetes.csv` - the Pima Indians Diabetes dataset: 768 rows, 8 features
(`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`)
and a binary `Outcome` (500 negative, 268 positive).

## Approach (`diabetes.ipynb`)

- Standardize all features with `StandardScaler`.
- 80/20 stratified train/test split (`random_state=42`).
- `svm.SVC(kernel='linear')`, evaluated with accuracy.
- Example prediction for a single input row, scaled with the same scaler.

Results printed in the notebook:

| Split | Accuracy |
|-------|----------|
| Train | 0.7915 |
| Test  | 0.7208 |

Note: the scaler is fit on the full dataset before the split, so test data leaks slightly into
preprocessing. Fitting it on the training split only is the cleaner setup.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook diabetes.ipynb
```

## Project structure

```
diabetes.ipynb
data/diabetes.csv
```
