from django.db import models

# Create your models here.
class Endpoint(models.Model):
	name = models.CharField(max_length=128)
	owner = models.CharField(max_length=128)
	created_at = models.DateTimeField(auto_now_add=True, blank=True)

class MLAlgorithm(models.Model):
	name = models.CharField(max_length=128)
	model_type = models.CharField(max_length=128)
	description = models.CharField(max_length=1000)
	code = models.CharField(max_length=50000)
	version = models.CharField(max_length=128)
	owner = models.CharField(max_length=128)
	created_at = models.DateTimeField(auto_now_add=True, blank=True)
	parent_endpoint = models.ForeignKey(Endpoint, on_delete=models.CASCADE)

class MLAlgorithmStatus(models.Model):
	status = models.CharField(max_length=128)
	active = models.BooleanField()
	created_by = models.CharField(max_length=128)
	created_at = models.DateTimeField(auto_now_add=True, blank=True)
	parent_mlalgorithm = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="status")

class MLRequest(models.Model):
	input_data = models.CharField(max_length=10000)
	full_response = models.CharField(max_length=10000)
	response = models.CharField(max_length=10000)
	feedback = models.CharField(max_length=10000, blank=True, null=True)
	created_at = models.DateTimeField(auto_now_add=True, blank=True)
	parent_mlalgorithm = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE)

class ABTest(models.Model):
	title = models.CharField(max_length=10000)
	model_type = models.Charfield(max_length=128)
	created_by = models.CharField(max_length=128)
	created_at = models.DateTimeField(auto_now_add=True, blank=True)
	ended_at = models.DateTimeField(blank=True, null=True)
	summary = models.CharField(max_length=10000)
	parent_mlalgorithm1 = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="parent_mlalgorithm1")
	parent_mlalgorithm2 = models.ForeignKey(MLAlgorithm, on_delete=models.CASCADE, related_name="parent_mlalgorithm2")
