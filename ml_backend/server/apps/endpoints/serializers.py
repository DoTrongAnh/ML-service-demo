from rest_framework import serializers
from apps.endpoints.models import Endpoint, MLAlgorithm, MLAlgorithmStatus, MLRequest, ABTest

class EndpointSerializer(serializers.ModelSerializer):
	class Meta:
		model = Endpoint
		read_only_fields = ("id","name","owner","created_at")
		fields = read_only_fields

class MLAlgorithmSerializer(serializers.ModelSerializer):
	current_status = serializers.SerializerMethodField(read_only=True)

	def get_current_status(self, mlalgorithm):
		return MLAlgorithmStatus.objects.filter(parent_mlalgorithm=mlalgorithm).latest('created_at').status

	class Meta:
		model = MLAlgorithm
		read_only_fields = ("id","name","model_type","description","code","version","owner","created_at",
			"parent_endpoint","current_status")
		fields = read_only_fields

class MLAlgorithmStatusSerializer(serializers.ModelSerializer):
	class Meta:
		model = MLAlgorithmStatus
		read_only_fields = ("id","active")
		fields = ("id","active","status","created_by","created_at","parent_mlalgorithm")

class MLRequestSerializer(serializers.ModelSerializer):
	class Meta:
		model = MLRequest
		read_only_fields = ("id","input_data","full_response","response","created_at","parent_mlalgorithm")
		fields = ("id","input_data","full_response","response","feedback","created_at","parent_mlalgorithm")

class ABTestSerializer(serializers.ModelSerializer):
	class Meta:
		model = ABTest
		read_only_fields = ("id","ended_at","created_at","summary","model_type")
		fields = ("id","title","model_type","created_by","created_at","ended_at","summary","parent_mlalgorithm1","parent_mlalgorithm2")