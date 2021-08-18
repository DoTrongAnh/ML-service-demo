from django.test import TestCase
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.random_forest import RandomForestClassifier

class MLTests(TestCase):
	def test_registry(self):
		registry = MLRegistry()
		self.assertEqual(0, len(registry.endpoints))
		endpoint_name = "income_classifier"
		algorithm_object = RandomForestClassifier()
		algorithm_name = "random forest"
		algorithm_status = "production"
		algorithm_version = "0.0.1"
		owner = "DTA"
		algorithm_description = "Random Forest with simple pre- and postprocessing"
		algorithm_code = inspect.getsource(RandomForestClassifier)
		registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name, algorithm_status, algorithm_version, owner, algorithm_description, algorithm_code)
		self.assertEqual(1, len(registry.endpoints))


	def test_rf_algorithm(self):
		input_data = {
		"age":37,
		"workclass":"Private",
		"fnlwgt":34146,
		"education-num":9,
		"education":"HS-grad",
		"marital-status":"Married-civ-spouse",
		"occupation":"Craft-repair",
		"relationship":"Husband",
		"race":"White",
		"sex":"Male",
		"capital-gain":0,
		"capital-loss":0,
		"hours-per-week":68,
		"native-country":"United-States"
		}
		the_alg = RandomForestClassifier()
		response = the_alg.compute_prediction(input_data)
		self.assertEqual("OK", response['status'])
		self.assertTrue('label' in response)
		self.assertEqual('<=50k', response['label'])

