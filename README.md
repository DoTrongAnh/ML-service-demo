# ML-service-demo
An exercise to deploy machine learning models with Django server. This Python codebase serves as the Django REST API backend server basis for deploying machine learning algorithms and performing predictions upon HTTP requests from the front-end and responding with prediction results. A set of models is constructed to keep track of algorithms, their use cases and status, as well as machine learning request histories and A/B testing histories.
- **Endpoint**: this model represents the use cases for which algorithms are trained. In terms of machine learning, it identifies the datasets used by certain algorithms for training. This is what the front-end needs to refer to, in order to send the right data to the right destination.
- **MLAlgorithm**: self-explanatory. This model identifies the algorithms, their type (classifier|regressor), their code, the **Endpoint** they refer to (use cases) and their status (production|testing|ab_testing). New algorithms are added to the database via a registry.
- **MLAlgorithmStatus**: logs for statuses of all algorithms. Only the lastest status of each algorithm will be active.
- **MLRequest**: logs for machine learning requests **POSTed** to `<domain>:<port>/api/v1/<Endpoint_name>/predict?<conditons>`. It contains the algorithm id that performed the prediction, the contents of the response such as status (OK|Error) and the prediction results. In case of A/B testing, a feedback is also given from the front-end unit with ground truth information for the back-end to assess the model's performance.
- **ABTest**: Logs for A/B tests performed between two algorithms of the same **Endpoint**. It contains the id of the 2 algorithms being tested and the test results comparison in text format.
## Interacting with the database (through API)
### Performing predictions
POST a Http request to this URL:
```
<domain>:<port>/api/v1/<Endpoint_name>/predict?<conditons>
```
with a JSON-rendered body of a data instance you want prediction from. If the status condition is not provided, the 1st algorithm in the **Endpoint** with the `production` status. The response will look something like this:
```
{
  "status":"OK",
  "label":<result of prediction>,
  "request_id":<ID of request to send feedback to>
}
```
Or, in case of failures:
```
{
  "status":"Error",
  "message":<Error message>,
  "request_id":<ID of request to send feedback to>
}
```
### Performing A/B testing
Before posting prediction requests, an A/B Test instance needs to be created. Simply input the test title and the two algorithms that will be tested for comparison in performance. Obviously in order for the test to be valid, the two algorithms need to refer to the same **Endpoint**. One the A/B test instance is set up, the two chosen algorithms will have the `ab_tesing` status. The tester should then send the test set to the following URL:
```
<domain>:<port>/api/v1/<Endpoint_name>/predict?status=ab_testing
```
Each request POSTed here will use either of the two algorithms of `ab_testing` status randomly, and each response of these requests will prompt the tester to send a PUT request to:
```
<domain>:<port>/api/v1/mlrequests/<request_id in the response>
```
with the body:
```
{
  "feedback":<ground truth in text format>
}
```
The A/B test then collects all of these updated **MLRequest** instances, calculate accuracy scores (classification) or RMSE scores (regression), and log the assessment results into its `summary` field. The algorithm that performs better will have its status changed to `production` while the other algorithm will have the `testing` status.
### Adding new algorithms
This is done via WSGI configuration of the back-end, where the algorithms are added to the database via the registry from the moment the server starts running. It is advised to only add new algorithms this way (via registry) as the registry also handles creating new statuses for these new algorithms. Algorithms without a status will not be used for prediction.
## Future activities
- Add more use cases (endpoints) and algorithms for those endpoints
- Develop a simple React.js web application to send prediction requests with user-input data and display prediction results.
## References
- Here is the [guide](https://www.deploymachinelearning.com/) this project followed.
