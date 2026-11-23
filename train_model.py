import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)
n_samples = 1200

# Continuous internal indicators
attendance = np.random.normal(loc=72, scale=14, size=n_samples).clip(30, 100)
cia = (0.55 * attendance + np.random.normal(loc=20, scale=12, size=n_samples)).clip(10, 100)
assignments = (0.45 * attendance + 0.35 * cia + np.random.normal(loc=10, scale=10, size=n_samples)).clip(20, 100)
prior_gpa = (0.05 * attendance + 0.04 * cia + np.random.normal(loc=1.5, scale=0.8, size=n_samples)).clip(3.0, 10.0)

# Ground-truth tiers: At-Risk, Average, Good
composite = 0.35 * attendance + 0.35 * cia + 0.15 * assignments + 0.15 * (prior_gpa * 10)
labels = []
for score in composite:
    if score < 52:
        labels.append('At-Risk')
    elif score < 72:
        labels.append('Average')
    else:
        labels.append('Good')

df = pd.DataFrame({
    'attendance_pct': attendance,
    'cia_score': cia,
    'assignment_rate': assignments,
    'prior_gpa': prior_gpa,
    'risk_tier': labels
})

X = df[['attendance_pct', 'cia_score', 'assignment_rate', 'prior_gpa']]
y = df['risk_tier']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)
clf.fit(X_train, y_train)

print("Classification Performance:")
print(classification_report(y_test, clf.predict(X_test)))

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model successfully generated: model.pkl")