import unittest
from symptom_model import ROOT, prepare_dataset, load_artifact, predict_symptoms

class SymptomTests(unittest.TestCase):
    def test_dataset_has_no_duplicate_sets(self):
        frame, summary = prepare_dataset()
        self.assertEqual(summary['original_rows'], 4920)
        self.assertEqual(len(frame), 304)
        self.assertFalse(frame.symptoms.duplicated().any())

    def test_model_is_invariant_to_order_case_and_repetition(self):
        artifact = load_artifact()
        a = predict_symptoms(['itching', 'skin rash'], artifact)
        b = predict_symptoms([' SKIN_RASH ', 'itching', 'itching'], artifact)
        self.assertEqual(a, b)

    def test_invalid_symptoms_rejected(self):
        artifact = load_artifact()
        for values in [[], [''], ['made-up symptom'], [1], 'itching']:
            with self.assertRaises(ValueError):
                predict_symptoms(values, artifact)

    def test_streamlit_interaction(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertTrue(app.error)
        app.multiselect[0].set_value(['itching', 'skin rash'])
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(app.dataframe)

if __name__ == '__main__':
    unittest.main()
