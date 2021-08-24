import joblib
import pandas as pd

class DecisionTreeClassifier:
	def __init__(self):
		path_to_artifacts = "../../research/"
		self.value_fill_missing = joblib.load(path_to_artifacts + "pi_train_mode.joblib")
		self.model = joblib.load(path_to_artifacts + "pi_decision_tree.joblib")

	def preprocessing(self, input_data):
		print("Preprocessing...")
		input_data = pd.DataFrame(input_data, index=[0])
		input_data.fillna(self.value_fill_missing, inplace=True)
		return input_data

	def predict(self, input_data):
		print("Predicting...")
		return self.model.predict(input_data)

	def postprocessing(self, input_data):
		print("Postprocessing...")
		return {"probability":1.0, "label":str(input_data), "status":"OK"}

	def compute_prediction(self, input_data):
		try:
			input_data = self.preprocessing(input_data)
			prediction = self.postprocessing(self.predict(input_data)[0])
		except Exception as e:
			print(str(e))
			return {"status":"Error","message":str(e)}

		return prediction