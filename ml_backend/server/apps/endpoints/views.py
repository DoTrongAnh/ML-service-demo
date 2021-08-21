from  rest_framework import viewsets, mixins, views, status
from rest_framework.response import Response
from apps.ml.registry import MLRegistry
from server.wsgi import registry
from django.db import transaction
from django.db.models import F
from apps.endpoints.models import Endpoint, MLAlgorithm, MLAlgorithmStatus, MLRequest, ABTest
from apps.endpoints.serializers import EndpointSerializer, MLAlgorithmSerializer, MLAlgorithmStatusSerializer, MLRequestSerializer, ABTestSerializer
import json
import math
import datetime
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

class ABTestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet, mixins.UpdateModelMixin, mixins.CreateModelMixin):
	serializer_class = ABTestSerializer
	queryset = ABTest.objects.all()

	def perform_create(self, serializer):
		try:
			with transaction.atomic():
				instance_type = registry.endpoints[instance.parent_mlalgorithm1].model_type
				if instance_type != registry.endpoints[instance.parent_mlalgorithm2].model_type:
					raise Exception("Two algorithms are of different types!")
				instance.model_type = instance_type
				instance = serializer.save()
				status_1 = MLAlgorithmStatus(status="ab_testing",
					created_by=instance.created_by,
					parent_mlalgorithm=instance.parent_mlalgorithm1,
					active=True)
				status_1.save()
				deactivate_other_statuses(status_1)
				status_2 = MLAlgorithmStatus(status="ab_testing",
					created_by=instance.created_by,
					parent_mlalgorithm=instance.parent_mlalgorithm2,
					active=True)
				status_2.save()
				deactivate_other_statuses(status_2)
		except Exception as e:
			raise APIException(str(e))

class StopABTestView(views.APIView):
	def post(self, request, ab_id, format=None):
		try:
			ab_test = ABTest.objects.get(pk=ab_id)
			model_type = ab_test.model_type
			if ab_test.ended_at is not None:
				return Response({"message":"A/B testing is already completed."})
			date_now = datetime.datetime.now()
			all_res_1_list = MLRequest.objects.filter(
				parent_mlalgorithm=ab_test.parent_mlalgorithm1,
				created_at__gt=ab_test.created_at,
				created_at__lt=date_now)
			
			all_res_2_list = MLRequest.objects.filter(
				parent_mlalgorithm=ab_test.parent_mlalgorithm2,
				created_at__gt=ab_test.created_at,
				created_at__lt=date_now)
			all_res_1 = all_res_1_list.count()
			all_res_2 = all_res_2_list.count()
			alg_id_1, alg_id_2 = ab_test.parent_mlalgorithm1, ab_test.parent_mlalgorithm2
			if model_type == "classifier":
				correct_res_1 = MLRequest.objects.filter(
				parent_mlalgorithm=ab_test.parent_mlalgorithm1,
				created_at__gt=ab_test.created_at,
				created_at__lt=date_now,
				response=F('feedback')).count()
				correct_res_2 = MLRequest.objects.filter(
				parent_mlalgorithm=ab_test.parent_mlalgorithm2,
				created_at__gt=ab_test.created_at,
				created_at__lt=date_now,
				response=F('feedback')).count()
				acc_1 = correct_res_1/float(all_res_1)
				acc_2 = correct_res_2/float(all_res_2)
				if acc_1 < acc_2: alg_id_1, ald_id_2 = alg_id_2, alg_id_1
				summary = f"Algorithm #1 accuracy: {acc_1}, Algorithm #2 accuracy: {acc_2}"
			else:
				rmse_1 = sum([(float(res.response) - float(res.feedback))**2 for res in all_res_1_list])
				rmse_2 = sum([(float(res.response) - float(res.feedback))**2 for res in all_res_2_list])
				rmse_1 = math.sqrt(rmse_1/float(all_res_1))
				rmse_2 = math.sqrt(rmse_2/float(all_res_2))
				if rmse_2 < rmse_1: alg_id_1, ald_id_2 = alg_id_2, alg_id_1
				summary = f"Algorithm #1 RMSE: {rmse_1}, Algorithm #2 RMSE: {rmse_2}"

			status_1 = MLAlgorithmStatus(status="production",
					created_by=ab_test.created_by,
					parent_mlalgorithm=alg_id_1,
					active=True)
			status_1.save()
			deactivate_other_statuses(status_1)
			status_2 = MLAlgorithmStatus(status="testing",
					created_by=ab_test.created_by,
					parent_mlalgorithm=alg_id_2,
					active=True)
			status_2.save()
			deactivate_other_statuses(status_2)
			
			ab_test.ended_at = date_now
			ab_test.summary = summary
			ab_test.save()
		except Exception as e:
			return Response(
				{"status":"Error","message":str(e)}, status=status.HTTP_400_BAD_REQUEST)
		return Response({"message":"A/B testing completed","summary":summary})

