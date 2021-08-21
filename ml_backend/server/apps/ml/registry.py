from apps.endpoints.models import Endpoint, MLAlgorithm, MLAlgorithmStatus

class MLRegistry():
	def __init__(self):
		self.endpoints = {}

	def add_algorithm(self, endpoint_name, algorithm_object, algorithm_name, algorithm_type, algorithm_status, algorithm_version, owner, algorithm_description, algorithm_code):
		endpoint,_ = Endpoint.objects.get_or_create(name=endpoint_name, owner=owner)
		database_object, algo_created = MLAlgorithm.objects.get_or_create(
			name=algorithm_name,
			description=algorithm_description,
			model_type=algorithm_type,
			code=algorithm_code,
			version=algorithm_version,
			owner=owner,
			parent_endpoint=endpoint)
		if algo_created:
			status = MLAlgorithmStatus(status=algorithm_status, created_by=owner, parent_mlalgorithm=database_object,
				active=True)
			status.save()
		self.endpoints[database_object.id] = algorithm_object


