import joblib
from tensorflow.keras.models import load_model

class CustomRNN:
	def __init__(self):
		path_to_artifacts = "../../research/"
		self.processor = joblib.load(path_to_artifacts + "mtprocessor.joblib")
		self.model = load_model(path_to_artifacts + "en_fr_translator")

	def preprocessing(self, data):
		print("Preprocessing...")
		if "sentence" not in data: data["sentence"] = ""
		return self.processor.preprocess(data["sentence"])

	def predict(self, data):
		print("Predicting...")
		return self.model.predict(data)

	def postprocessing(self, data):
		print("Postprocessing...")
		label = self.processor.postprocess(data)
		return {"probability":1.0, "label":label, "status":"OK"}

	def compute_prediction(self, input_data):
		try:
			input_data = self.preprocessing(input_data)
			prediction = self.postprocessing(self.predict(input_data)[0])
		except Exception as e:
			print(str(e))
			return {"status":"Error","message":str(e)}

		return prediction