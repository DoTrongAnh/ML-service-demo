from  rest_framework import viewsets, mixins, views, status
from rest_framework.response import Response
from apps.ml.registry import MLRegistry
from server.wsgi import registry
from apps.endpoints.models import Endpoint, MLAlgorithm, MLAlgorithmStatus, MLRequest
from apps.endpoints.serializers import EndpointSerializer, MLAlgorithmSerializer, MLAlgorithmStatusSerializer, MLRequestSerializer
import json
from numpy.random import rand


# Create your views here.
class EndpointViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
	serializer_class = EndpointSerializer
	queryset = Endpoint.objects.all()

class MLAlgorithmViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
	serializer_class = MLAlgorithmSerializer
	queryset = MLAlgorithm.objects.all()

def deactivate_other_statuses(instance):
	old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm=instance.parent_mlalgorithm,
		created_at__lt=instance.created_at,active=True)
	for status in old_statuses:
		status.active = False
	MLAlgorithmStatus.objects.bulk_update(old_statuses,['active'])

class MLAlgorithmStatusViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet, mixins.CreateModelMixin):
	serializer_class = MLAlgorithmStatusSerializer
	queryset = MLAlgorithmStatus.objects.all()
	def perform_create(self, serializer):
		try:
			with transaction.atomic():
				instance = serializer.save(active=True)
				deactivate_other_statuses(instance)
		except Exception as e:
			raise APIException(str(e))

class MLRequestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet, mixins.UpdateModelMixin):
	serializer_class = MLRequestSerializer
	queryset = MLRequest.objects.all()

class PredictView(views.APIView):
	def post(self, request, endpoint_name, format=None):
		algorithm_status = self.request.query_params.get("status","production")
		algorithm_version = self.request.query_params.get("version")
		algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name,
			status__status=algorithm_status, status__active=True)
		if algorithm_version is not None: algs = algs.filter(version=algorithm_version)
		if len(algs) == 0: return Response(
			{"status":"Error","message":"ML algorithm not available"},
			status=status.HTTP_400_BAD_REQUEST)
		if len(algs) != 1 and algorithm_status != "ab_testing":
			return Response({"status":"Error","message":"ML algorithm selection is ambiguous. Please specify algorithm version"},
				status=status.HTTP_400_BAD_REQUEST)
		alg_index = 0
		if algorithm_status == "ab_testing":
			alg_index = 0 if rand() < 0.5 else 1

		algorithm_object = registry.endpoints[algs[alg_index].id]
		prediction = algorithm_object.compute_prediction(request.data)

		label = prediction["label"] if "label" in prediction else "error"
		ml_request = MLRequest(
			input_data=json.dumps(request.data),
			full_response=prediction,
			response=label,
			feedback="",
			parent_mlalgorithm=algs[alg_index])
		ml_request.save()
		prediction["request_id"] = ml_request.id
		return Response(prediction)
