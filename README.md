# Midterm Project - Pet Preference Prediction Using Machine Learning.

<p align="center"><img src="header.png" alt="Pet Preference" width="600"/></p>


1. What is the problem?
The problem is to build a machine learning model that predicts whether a person is more likely to be a cat person, dog person, both, or neither based on demographic, social, and behavioral features such as age, gender, type of settlement, education level, income, and media habits. The model aims to identify how lifestyle and background factors correlate with personal preferences toward pets.

2. Why is it important?
Understanding personality and lifestyle correlations with pet preferences can be valuable for marketing, social research, and behavioral psychology. Pet-related companies and advertisers can use such insights to better target products and services, while sociologists and psychologists may use the findings to explore cultural and demographic influences on human–animal relationships. It also demonstrates how social survey data can be turned into behavioral prediction through ML.

3. How will the solution be used?
The trained model can be integrated into survey analytics tools or marketing platforms to automatically classify respondents or customers into preference categories. It could also be used for visualization and demographic analysis—showing patterns like "younger urban males prefer dogs" or "rural, older respondents tend to like both." Researchers can apply it to new datasets to explore trends across regions or time.

4. Who will benefit from this solution?
	•	Marketing and advertising teams: to personalize campaigns for pet-related products.
	•	Sociologists and behavioral scientists: to study human-animal affinity as a social pattern.
	•	Survey organizations: to enrich responses with automated behavioral tagging.
	•	Pet industry businesses: to understand their customer base more precisely.
	•	General public: through more relevant pet content and product recommendations.

**Dataset**: 
Unfortunately there are not many publicly available datasets specifically about pet preferences. The only one I was able to find is a survey from a Russian social research organization VCIOM, which includes questions about pet ownership and preferences along with demographic and lifestyle information. The dataset can be accessed [here](https://wciom.ru/analytical-reviews/analiticheskii-obzor/publichnaja-zhizn-domashnikh-zhivotnykh).

This dataset contains a variety of features including what type of pets people own, their preferences for cats or dogs, age. Also the dataset has some infrastructure questions like whether there is a pet store nearby, and what other people are saying about pets in their social circles which we will ignore because we are interested in predicting pet preference based on demographic and lifestyle features only.

# Pet Preference Prediction

A simple machine learning project that predicts pet preferences (Cat, Dog, or Both) based on demographic data using XGBoost.

## Live API Endpoint

The model is deployed and available at: **https://zoomcamp-midterm.a7e.top**

- Interactive web interface: https://zoomcamp-midterm.a7e.top/ (served via `index.html`)
- Interactive API docs: https://zoomcamp-midterm.a7e.top/docs

The API is running on a personal Kubernetes cluster with NGINX ingress.

## Prerequisites

Install required packages:

```bash
pip install pandas pyreadstat xgboost scikit-learn fastapi uvicorn pydantic
```

## Training the Model

Run the training script to train the XGBoost model and save it:

```bash
python train.py
```

This will:
- Load data from `animals.sav`
- Split into train/validation/test sets
- Train an XGBoost classifier
- Save the model to `pet_preference_model.bin`

## Running the API Server

Start the FastAPI server:

```bash
python serve.py
```

Or with uvicorn directly:

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AGE": 35,
    "SEX": "Male",
    "EDU": "Higher education",
    "DOHOD": "Average",
    "PROF1": "Employed",
    "PROF2": "Private sector",
    "PROF3": "Services"
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "AGE": 35,
      "SEX": "Male",
      "EDU": "Higher education",
      "DOHOD": "Average",
      "PROF1": "Employed",
      "PROF2": "Private sector",
      "PROF3": "Services"
    },
    {
      "AGE": 28,
      "SEX": "Female",
      "EDU": "Secondary education",
      "DOHOD": "Below average",
      "PROF1": "Student"
    }
  ]'
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation where you can test the API interactively.

## Input Features

- **AGE**: Age in years (integer)
- **SEX**: Gender ("Male" or "Female")
- **EDU**: Education level
- **DOHOD**: Income level
- **PROF1**: Main occupation
- **PROF2**: Organization type (optional, defaults to "unknown")
- **PROF3**: Industry/field (optional, defaults to "unknown")

## Output

The API returns:
- **prediction**: The predicted class (Cat, Dog, or Both)
- **probabilities**: Probability distribution across all classes

## Docker Build & Deployment

### Building the Docker Image

The application is containerized using Docker with `uv` for fast dependency management:

```bash
sudo docker build -t golubyuri/mlzoomcamp:midterm-project .
sudo docker push golubyuri/mlzoomcamp:midterm-project
```

### Kubernetes Deployment with Helm

The service is deployed to a personal Kubernetes cluster using Helm with the following configuration:

```yaml
replicaCount: 1

image:
  repository: golubyuri/mlzoomcamp
  tag: midterm-project
  pullPolicy: IfNotPresent

nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

service:
  type: ClusterIP
  port: 9696
  targetPort: 9696

ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/backend-protocol: HTTP
  hosts:
    - host: zoomcamp-midterm.a7e.top
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: false

nodeSelector: {}

tolerations: []

affinity: {}
```

The deployment uses:
- **1 replica** for the service
- **ClusterIP** service type with internal port 9696
- **NGINX Ingress** for external access at https://zoomcamp-midterm.a7e.top
- **Resource limits**: 1 CPU core and 1Gi memory
- **Resource requests**: 500m CPU and 512Mi memory

