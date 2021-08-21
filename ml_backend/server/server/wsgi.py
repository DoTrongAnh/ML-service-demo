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

try:
	registry = MLRegistry()
	rf = RandomForestClassifier()
	et = ExtraTreesClassifier()
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
		algorithm_status="production",
		algorithm_version="0.0.1",
		owner="DTA",
		algorithm_description="Extra Trees with simple pre- and postprocessing",
		algorithm_code=inspect.getsource(ExtraTreesClassifier))
except Exception as e:
	print("Exception while loading the algorithms to the registry: ", str(e))
