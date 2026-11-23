import os
import io
import pickle
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, 'academic_records.db')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')


# ---------------------------------------------------------
# 1. DATABASE INITIALIZATION & CONNECTION HELPER
# ---------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON;')
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cohort_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_name TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_students INTEGER DEFAULT 0,
            at_risk_count INTEGER DEFAULT 0
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL DEFAULT 1,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            attendance_pct REAL NOT NULL,
            cia_score REAL NOT NULL,
            assignment_rate REAL NOT NULL,
            prior_gpa REAL NOT NULL,
            prediction TEXT NOT NULL,
            health_score INTEGER NOT NULL,
            key_driver TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES cohort_batches(id) ON DELETE CASCADE
        )
    ''')

    # Ensure at least one default batch exists if database is brand new
    existing_batch = conn.execute('SELECT id FROM cohort_batches LIMIT 1').fetchone()
    if not existing_batch:
        conn.execute(
            'INSERT INTO cohort_batches (batch_name, total_students, at_risk_count) VALUES (?, ?, ?)',
            ('Initial Benchmark Cohort', 0, 0)
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# 2. MACHINE LEARNING MODEL & DIAGNOSTIC ATTRIBUTION ENGINE
# ---------------------------------------------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None


ml_model = load_model()


def analyze_student(attendance_pct, cia_score, assignment_rate, prior_gpa):
    # Clamp values strictly to valid domain boundaries
    att = float(np.clip(attendance_pct, 0.0, 100.0))
    cia = float(np.clip(cia_score, 0.0, 100.0))
    asg = float(np.clip(assignment_rate, 0.0, 100.0))
    gpa = float(np.clip(prior_gpa, 0.0, 10.0))

    # Weighted Composite Academic Health Score (0 - 100)
    health_score = int(round(
        (att * 0.35) +
        (cia * 0.35) +
        (asg * 0.15) +
        ((gpa * 10.0) * 0.15)
    ))

    # Model inference with deterministic threshold alignment
    if ml_model is not None:
        features = pd.DataFrame([[att, cia, asg, gpa]], columns=[
            'attendance_pct', 'cia_score', 'assignment_rate', 'prior_gpa'
        ])
        raw_pred = str(ml_model.predict(features)[0])
    else:
        raw_pred = 'Good'

    if health_score < 50 or att < 55 or cia < 35:
        prediction = 'At-Risk'
    elif health_score < 75 or att < 75 or cia < 55:
        prediction = 'Average'
    else:
        prediction = raw_pred if raw_pred in ['At-Risk', 'Average', 'Good'] else 'Good'

    # Multi-Variable Deficit Attribution & Action Prescription
    deficits = []
    if att < 75:
        deficits.append((75 - att, f"Missing Classes ({int(round(att))}%)", "Call Parents & Mandatory Remedial Class"))
    if cia < 55:
        deficits.append((55 - cia, f"Poor Midterm CIA ({int(round(cia))}/100)", "Enrolled in Remedial Lectures"))
    if asg < 70:
        deficits.append((70 - asg, f"Low Assignments ({int(round(asg))}%)", "Structured Assignment Submission Plan"))
    if gpa < 6.0:
        deficits.append(((6.0 - gpa) * 10, f"Low Past SGPA ({gpa:.2f})", "Assign Faculty Academic Mentor"))

    if deficits:
        deficits.sort(key=lambda x: x[0], reverse=True)
        _, key_driver, recommended_action = deficits[0]
    else:
        key_driver = "All Clear"
        recommended_action = "Eligible for Honors & Advanced Projects"

    return prediction, health_score, key_driver, recommended_action


def update_batch_counts(conn, batch_id):
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN prediction = 'At-Risk' THEN 1 ELSE 0 END) as critical
        FROM predictions WHERE batch_id = ?
    ''', (batch_id,)).fetchone()
    conn.execute(
        'UPDATE cohort_batches SET total_students = ?, at_risk_count = ? WHERE id = ?',
        (stats['total'] or 0, stats['critical'] or 0, batch_id)
    )


# ---------------------------------------------------------
# 3. WEB DASHBOARD & CRUD ROUTES
# ---------------------------------------------------------
@app.route('/')
def dashboard():
    conn = get_db()
    batches = conn.execute('SELECT * FROM cohort_batches ORDER BY id DESC').fetchall()

    selected_batch_id = request.args.get('batch_id', type=int)
    if not selected_batch_id and batches:
        selected_batch_id = batches[0]['id']

    active_batch = conn.execute('SELECT * FROM cohort_batches WHERE id = ?', (selected_batch_id,)).fetchone()
    records = conn.execute(
        'SELECT * FROM predictions WHERE batch_id = ? ORDER BY id DESC',
        (selected_batch_id,)
    ).fetchall()

    total_students = len(records)
    at_risk_count = sum(1 for r in records if r['prediction'] == 'At-Risk')
    average_count = sum(1 for r in records if r['prediction'] == 'Average')
    good_count = sum(1 for r in records if r['prediction'] == 'Good')

    conn.close()
    return render_template(
        'dashboard.html',
        batches=batches,
        active_batch=active_batch,
        records=records,
        total_students=total_students,
        at_risk_count=at_risk_count,
        average_count=average_count,
        good_count=good_count
    )


@app.route('/predict_single', methods=['POST'])
def predict_single():
    batch_id = request.form.get('batch_id', type=int)
    student_id = str(request.form.get('student_id', 'EN0000')).strip().upper()
    student_name = str(request.form.get('student_name', 'Unknown Student')).strip().title()
    attendance_pct = float(request.form.get('attendance_pct', 0) or 0)
    cia_score = float(request.form.get('cia_score', 0) or 0)
    assignment_rate = float(request.form.get('assignment_rate', 0) or 0)
    prior_gpa = float(request.form.get('prior_gpa', 0) or 0)

    pred, health, driver, action = analyze_student(attendance_pct, cia_score, assignment_rate, prior_gpa)

    conn = get_db()
    if not batch_id:
        latest = conn.execute('SELECT id FROM cohort_batches ORDER BY id DESC LIMIT 1').fetchone()
        batch_id = latest['id'] if latest else 1

    conn.execute('''
        INSERT INTO predictions (
            batch_id, student_id, student_name, attendance_pct, cia_score,
            assignment_rate, prior_gpa, prediction, health_score, key_driver, recommended_action
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        batch_id, student_id, student_name,
        float(np.clip(attendance_pct, 0, 100)),
        float(np.clip(cia_score, 0, 100)),
        float(np.clip(assignment_rate, 0, 100)),
        float(np.clip(prior_gpa, 0, 10)),
        pred, health, driver, action
    ))
    update_batch_counts(conn, batch_id)
    conn.commit()
    conn.close()

    return redirect(url_for('dashboard', batch_id=batch_id))


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    file = request.files.get('csv_file')
    batch_name = str(request.form.get('batch_name', '')).strip()
    if not batch_name:
        batch_name = f"Cohort Batch ({datetime.now().strftime('%d %b %Y %H:%M')})"

    if not file or not file.filename.endswith('.csv'):
        return redirect(url_for('dashboard'))

    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    conn = get_db()
    cursor = conn.execute('INSERT INTO cohort_batches (batch_name) VALUES (?)', (batch_name,))
    new_batch_id = cursor.lastrowid

    for idx, row in df.iterrows():
        s_id = str(row.get('student_id', f'STU{idx+1:03d}')).strip().upper()
        s_name = str(row.get('student_name', f'Student {idx+1}')).strip().title()

        att = pd.to_numeric(row.get('attendance_pct', 0), errors='coerce')
        cia = pd.to_numeric(row.get('cia_score', 0), errors='coerce')
        asg = pd.to_numeric(row.get('assignment_rate', 0), errors='coerce')
        gpa = pd.to_numeric(row.get('prior_gpa', 0), errors='coerce')

        att = float(np.clip(0.0 if pd.isna(att) else att, 0.0, 100.0))
        cia = float(np.clip(0.0 if pd.isna(cia) else cia, 0.0, 100.0))
        asg = float(np.clip(0.0 if pd.isna(asg) else asg, 0.0, 100.0))
        gpa = float(np.clip(0.0 if pd.isna(gpa) else gpa, 0.0, 10.0))

        pred, health, driver, action = analyze_student(att, cia, asg, gpa)

        conn.execute('''
            INSERT INTO predictions (
                batch_id, student_id, student_name, attendance_pct, cia_score,
                assignment_rate, prior_gpa, prediction, health_score, key_driver, recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (new_batch_id, s_id, s_name, att, cia, asg, gpa, pred, health, driver, action))

    update_batch_counts(conn, new_batch_id)
    conn.commit()
    conn.close()

    return redirect(url_for('dashboard', batch_id=new_batch_id))


@app.route('/delete_batch/<int:batch_id>', methods=['POST'])
def delete_batch(batch_id):
    conn = get_db()
    conn.execute('DELETE FROM cohort_batches WHERE id = ?', (batch_id,))
    remaining = conn.execute('SELECT id FROM cohort_batches LIMIT 1').fetchone()
    if not remaining:
        conn.execute(
            'INSERT INTO cohort_batches (batch_name, total_students, at_risk_count) VALUES (?, ?, ?)',
            ('Initial Benchmark Cohort', 0, 0)
        )
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))


@app.route('/export_report')
def export_report():
    batch_id = request.args.get('batch_id', type=int)
    conn = get_db()
    if batch_id:
        rows = conn.execute('SELECT * FROM predictions WHERE batch_id = ? ORDER BY id DESC', (batch_id,)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall()
    conn.close()

    export_data = [{
        'Roll No': r['student_id'],
        'Student Name': r['student_name'],
        'Status': r['prediction'],
        'Academic Health (100)': r['health_score'],
        'Primary Bottleneck': r['key_driver'],
        'Prescribed Action': r['recommended_action'],
        'Attendance %': r['attendance_pct'],
        'CIA Score': r['cia_score'],
        'Assignments %': r['assignment_rate'],
        'Past SGPA': r['prior_gpa']
    } for r in rows]

    df_export = pd.DataFrame(export_data)
    buffer = io.BytesIO()
    df_export.to_csv(buffer, index=False)
    buffer.seek(0)

    filename = f"Faculty_Report_{datetime.now().strftime('%d_%b_%Y')}.csv"
    return send_file(buffer, mimetype='text/csv', as_attachment=True, download_name=filename)


# ---------------------------------------------------------
# 4. ANALYTICAL API ENDPOINTS (COHORT COMPARISON & TRENDS)
# ---------------------------------------------------------
@app.route('/api/student_trend/<student_id>')
def student_trend(student_id):
    conn = get_db()
    rows = conn.execute('''
        SELECT cb.batch_name, p.health_score, p.prior_gpa, p.attendance_pct, p.cia_score
        FROM predictions p
        JOIN cohort_batches cb ON p.batch_id = cb.id
        WHERE UPPER(p.student_id) = UPPER(?)
        ORDER BY cb.id ASC
    ''', (student_id.strip(),)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/compare_cohorts')
def compare_cohorts():
    conn = get_db()
    batches = conn.execute('SELECT * FROM cohort_batches ORDER BY id ASC').fetchall()
    comparison = []
    for b in batches:
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN prediction = 'At-Risk' THEN 1 ELSE 0 END) as critical,
                SUM(CASE WHEN prediction = 'Average' THEN 1 ELSE 0 END) as warning,
                SUM(CASE WHEN prediction = 'Good' THEN 1 ELSE 0 END) as safe,
                ROUND(AVG(health_score), 1) as avg_health,
                ROUND(AVG(attendance_pct), 1) as avg_att,
                ROUND(AVG(cia_score), 1) as avg_cia
            FROM predictions WHERE batch_id = ?
        ''', (b['id'],)).fetchone()
        comparison.append({
            'id': b['id'],
            'batch_name': b['batch_name'],
            'total': stats['total'] or 0,
            'critical': stats['critical'] or 0,
            'warning': stats['warning'] or 0,
            'safe': stats['safe'] or 0,
            'avg_health': stats['avg_health'] or 0,
            'avg_att': stats['avg_att'] or 0,
            'avg_cia': stats['avg_cia'] or 0
        })
    conn.close()
    return jsonify(comparison)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
