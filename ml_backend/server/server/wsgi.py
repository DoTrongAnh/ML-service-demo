"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

import inspect
from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.random_forest import RandomForestClassifier
from apps.ml.income_classifier.extra_trees import ExtraTreesClassifier
import apps.ml.premium_insurance.random_forest as pi_rf
import apps.ml.premium_insurance.decision_tree as pi_dt
from apps.ml.machine_translation.custom_rnn import CustomRNN

try:
	registry = MLRegistry()
	rf = RandomForestClassifier()
	et = ExtraTreesClassifier()
	rf2 = pi.rf.RandomForestClassifier()
	dt = pi_dt.DecisionTreeClassifier()
	rnn = CustomRNN()
	registry.add_algorithm(
		algorithm_object=rf,
		algorithm_name="random forest",
		algorithm_type="classifier",
		endpoint_name="income_classifier",
		algorithm_status="production",
		algorithm_version="0.0.1",
		owner="DTA",
		algorithm_description="Random Forest with simple pre- and postprocessing",
		algorithm_code=inspect.getsource(RandomForestClassifier))
	registry.add_algorithm(
		algorithm_object=et,
		algorithm_name="extra trees",
		algorithm_type="classifier",
		endpoint_name="income_classifier",
		algorithm_status="testing",
		algorithm_version="0.0.1",
		owner="DTA",
		algorithm_description="Extra Trees with simple pre- and postprocessing",
		algorithm_code=inspect.getsource(ExtraTreesClassifier))
	registry.add_algorithm(
		algorithm_object=rf2,
		algorithm_name="random forest",
		algorithm_type="classifier",
		endpoint_name="premium_insurance",
		algorithm_status="production",
		algorithm_version="0.0.1",
		owner="DTA",
		algorithm_description="Random Forest for raw insurance data",
		algorithm_code=inspect.getsource(pi_rf.RandomForestClassifier))
	registry.add_algorithm(
		algorithm_object=dt,
		algorithm_name="decision trees",
		algorithm_type="classifier",
		endpoint_name="premium_insurance",
		algorithm_status="testing",
		algorithm_version="0.0.1",
		owner="DTA",
		algorithm_description="Decision Tree for raw insurance data",
		algorithm_code=inspect.getsource(pi_dt.DecisionTreeClassifier))
	registry.add_algorithm(
		algorithm_object=rnn,
		algorithm_name="custom rnn",
		algorithm_type="nlp",
		endpoint_name="machine_translation",
		algorithm_status="production",
		algorithm_version="0.0.1",
		owner="DTA",
		algorithm_description="Recurrent neural network with embedding and bidirectional encoder-decoder",
		algorithm_code=inspect.getsource(CustomRNN))
except Exception as e:
	print("Exception while loading the algorithms to the registry: ", str(e))
