# 🎓 EduPulse AI — Student Academic Performance & Early-Warning Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-PythonAnywhere-4f46e5?style=for-the-badge&logo=python&logoColor=white)](https://edupulseai.pythonanywhere.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Decision%20Tree-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-Responsive%20UI-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

**EduPulse AI** is an Educational Data Mining (EDM) early-warning system built to help college faculty and academic mentors detect struggling students mid-semester—before end-semester exams make intervention too late.

Instead of relying on invasive personal or demographic attributes, EduPulse AI evaluates **four modifiable academic indicators** within direct faculty control (**Attendance %**, **CIA Internal Marks**, **Assignment Completion %**, and **Prior SGPA**) to classify risk tiers, calculate a **0–100 Composite Academic Health Index**, isolate primary root-cause bottlenecks, and prescribe concrete faculty actions.

🌐 **Live Web App:** [https://edupulseai.pythonanywhere.com](https://edupulseai.pythonanywhere.com)

---

## ✨ Key Features

- **🚦 Traffic-Light Risk Triage:** Automatically categorizes students into **Safe (Good)**, **Warning (Average)**, and **Critical (At-Risk)** tiers using a Gini-based Decision Tree Classifier (`model.pkl`) aligned with institutional threshold rules.
- **🎯 Automated Root-Cause Attribution & Prescription:** Identifies the exact academic deficit (e.g., *Missing Classes*, *Poor Midterm CIA*, *Low Assignments*, *Low Past SGPA*) and prescribes targeted faculty actions (*Mandatory Remedial Class*, *Academic Mentor Assignment*, *Structured Submission Plan*).
- **📂 Cohort Batch Ingestion & History Sidebar:** Upload entire classroom rosters via `.csv` under named exam sessions (e.g., *Midterm 1*, *Pre-University*) and switch between historical snapshots anytime.
- **📊 Interactive Visual Analytics (Chart.js):**
  - **Cohort Risk Distribution Donut Chart** & **Class Bottleneck Comparison Bar Chart** at the cohort level.
  - **Individual Factor Share Pie Chart** inside each student's diagnostic profile.
  - **Cross-Semester Student Trajectory Line Chart** tracking a student's Health Index and SGPA across multiple uploaded batches.
- **⚖️ Side-by-Side Cohort Comparison:** Compare pass/risk ratios, class averages, and health scores across two different exam milestones in one view.
- **⚡ Real-Time Search, Tier Filtering & Column Sorting:** Filter by name or roll number, isolate risk tiers, and click column headers (`Health Index`, `Attendance`, `CIA Marks`, etc.) to sort ascending or descending.
- **📑 NAAC / NBA Audit Export:** One-click CSV report export formatted for departmental mentoring records and accreditation audits.
- **📱 Mobile-Responsive Design:** Slide-over navigation drawer, tap-to-open student diagnostics, and adaptive charts for desktop, tablet, and smartphone browsers.

---

## 🛠️ Tech Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Backend & API** | Python 3, Flask, RESTful JSON Endpoints |
| **Machine Learning & Data** | Scikit-Learn (`DecisionTreeClassifier`), Pandas, NumPy |
| **Database** | SQLite3 (`academic_records.db` with relational `cohort_batches` & `predictions` tables) |
| **Frontend & Visualization** | HTML5, Tailwind CSS, Vanilla JavaScript, Chart.js |
| **Deployment** | PythonAnywhere |

---

## 📁 Project Structure

```text
EduPulse-AI/
├── app.py                   # Flask server, ML inference engine, SQLite CRUD & JSON APIs
├── model.pkl                # Pre-trained Scikit-Learn CART Decision Tree Classifier
├── academic_records.db      # Local SQLite database (auto-initialized on first run)
├── sample_batch.csv         # Sample student cohort CSV for instant testing
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── templates/
    └── dashboard.html       # Responsive faculty dashboard, modals & Chart.js scripts
