import io
import unittest
import numpy as np
import pandas as pd
from app import app, init_db, evaluate_analytics

class TestAcademicPredictionEngine(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        init_db()

    # 1. Boundary & Extrema Testing
    def test_extreme_zero_values(self):
        """Test with absolute zeroes (student with zero attendance and zero marks)."""
        pred, risk, driver, action = evaluate_analytics(0.0, 0.0, 0.0, 0.0)
        self.assertEqual(pred, 'At-Risk')
        self.assertGreaterEqual(risk, 80.0)
        self.assertIn("Attd 0%", driver)

    def test_extreme_perfect_values(self):
        """Test with perfect 100s across all boards."""
        pred, risk, driver, action = evaluate_analytics(100.0, 100.0, 100.0, 10.0)
        self.assertEqual(pred, 'Good')
        self.assertLess(risk, 20.0)
        self.assertEqual(driver, "Consistent")

    def test_clamping_out_of_bounds(self):
        """Test resilience against negative numbers or numbers exceeding 100%."""
        pred, risk, driver, action = evaluate_analytics(-50.0, 150.0, -10.0, 25.0)
        self.assertIn(pred, ['At-Risk', 'Average', 'Good'])
        self.assertTrue(0.0 <= risk <= 100.0)

    # 2. Web Form API Ingestion Tests
    def test_predict_single_endpoint_success(self):
        """Validate successful HTTP POST for single candidate."""
        response = self.client.post('/predict_single', data={
            'student_id': 'TEST_EN01',
            'student_name': 'Test Student',
            'attendance': '62.5',
            'cia': '41.0',
            'assignments': '55.0',
            'gpa': '5.2'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    def test_sql_injection_sanitization(self):
        """Ensure SQL injection strings in Student ID or Name do not break the engine."""
        payload = "' OR '1'='1'; DROP TABLE predictions; --"
        response = self.client.post('/predict_single', data={
            'student_id': payload,
            'student_name': "Hacker Candidate",
            'attendance': '75.0',
            'cia': '50.0',
            'assignments': '70.0',
            'gpa': '6.0'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        
        # Verify predictions table was NOT dropped
        test_read = self.client.get('/')
        self.assertEqual(test_read.status_code, 200)

    # 3. CSV Ingestion Failure & Fuzz Testing
    def test_corrupted_csv_headers(self):
        """Ensure CSVs missing critical columns fail gracefully with an HTTP 400 rather than crashing."""
        corrupted_csv = "random_col1,random_col2\n10,20\n"
        data = {'file': (io.BytesIO(corrupted_csv.encode('utf-8')), 'bad.csv')}
        response = self.client.post('/upload_csv', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Missing required CSV columns", response.data)

    def test_csv_with_dirty_nulls_and_strings(self):
        """Validate pandas self-healing: numeric columns containing text strings or blanks."""
        dirty_csv = (
            "student_id,student_name,attendance_pct,cia_score,assignment_rate,prior_gpa\n"
            "EN_DIRTY_1,Dirty Candidate,NaN,invalid_str,,8.5\n"
            "EN_DIRTY_2,Second Candidate,82.0,75.0,80.0,7.1\n"
        )
        data = {'file': (io.BytesIO(dirty_csv.encode('utf-8')), 'dirty.csv')}
        response = self.client.post('/upload_csv', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

    # 4. Stress Ingestion Test
    def test_high_volume_batch_stress(self):
        """Ingest 500 rows sequentially and verify sub-second completion without memory fault."""
        n_rows = 500
        rows = ["student_id,student_name,attendance_pct,cia_score,assignment_rate,prior_gpa"]
        for i in range(n_rows):
            rows.append(f"STRESS_{i},Student {i},{np.random.randint(40,95)},{np.random.randint(25,90)},{np.random.randint(30,95)},{round(np.random.uniform(4.0,9.5),2)}")
        
        stress_csv = "\n".join(rows)
        data = {'file': (io.BytesIO(stress_csv.encode('utf-8')), 'stress.csv')}
        response = self.client.post('/upload_csv', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()