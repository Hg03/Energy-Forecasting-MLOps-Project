## Table of Contents:
- Course Introduction
- Course Lessons
- The Batch Architecture
- The 7-Steps Drill Down
- How Can We Adapt the Batch Architecture to an Online ML System?
- Conclusion


## Course Introduction
At the end of this 7 lessons course, you will know how to:

- Design a batch-serving architecture
- Use Hopsworks as a feature store
- Design a feature engineering pipeline that reads data from an API
- Build a training pipeline with hyper-parameter tunning
- Use W&B as an ML Platform to track your experiments, models, and metadata
- Implement a batch prediction pipeline
- Use Poetry to build your own Python packages
- Deploy your own private PyPi server
- Orchestrate everything with Airflow
- Use the predictions to code a web app using FastAPI and Streamlit
- Use Docker to containerize your code
- Use Great Expectations to ensure data validation and integrity
- Monitor the performance of the predictions over time
- Deploy everything to GCP
- Build a CI/CD pipeline using GitHub Actions
  
If that sounds like a lot, don’t worry. After you cover this course, you will understand everything I said before. Most importantly, you will know WHY I used all these tools and how they work together as a system. If you want to get the most out of this course, I suggest you access the GitHub repository containing all the lessons’ code. This course is designed to quickly read and replicate the code along the articles. By the end of the course, you will know how to implement the diagram below. Don’t worry if something doesn’t make sense to you. I will explain everything in detail.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/0b000205-6ac2-4da5-8f64-c5a422ebbee6)

This article will paint an overall picture of how to design, implement, and deploy a batch architecture.

Also, we will show you why batch architecture is a powerful way to start most of your ML projects and how to adapt it to a real-time system.

Course Lessons:

1. Batch Serving. Feature Stores. Feature Engineering Pipelines.
2. Training Pipelines. ML Platforms. Hyperparameter Tuning.
3. Batch Prediction Pipeline. Package Python Modules with Poetry.
4. Private PyPi Server. Orchestrate Everything with Airflow.
5. Data Validation for Quality and Integrity using GE. Model Performance Continuous Monitoring.
6. onsume and Visualize your Model’s Predictions using FastAPI and Streamlit. Dockerize Everything.
7. Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.
8. [Bonus] Behind the Scenes of an ‘Imperfect’ ML Project — Lessons and Insights


## 1. The Batch Architecture

**Overview**

Any batch architecture consists of 3 main components:

- **Feature Engineering Pipeline**: ingests data from one or multiple data sources and transforms it into valuable features for the model
- **Training Pipeline**: takes the features as input and trains and evaluates the model on them. At this step, before finding the best model & hyperparameters, you experiment with multiple configurations. After, you can automate the training process by leveraging the best configuration you found during the experimentation step. The output of this step is a model artifact.
Batch Prediction Pipeline: Takes as input the features and model artifacts and runs the model on batches (big chunks) of data. Usually, the inference is not real-time, as it takes a while to run. Afterward, it saves the predictions into storage, from where various apps can consume the predictions in real-time.
→ The pipelines are wrapped up in a DAG and scheduled to run every week, day, hour, etc. (what makes sense for your use case).


Batch architecture overview [Image by the Author].
Variations of the Batch Architecture
Some common variations of the batch architecture are:

Online Training: Every time new data is ingested, you train the model using the best configuration you found during your experimentation.
Offline Training: You train your model once on your static dataset and use the trained model to make predictions on new data.
What you will choose depends a lot on your context and resources. Remember that training a model is an expensive operation.

→ In the course, we used real-time energy consumption levels, which are modeled as time series. Thus, our strategy was based on the online training paradigm.


Batch architecture variations [Image by the Author].
Why Use a Batch Architecture?
Often, your initial deployment will be in batch mode. Why?

Upsides:

You don’t have to focus on latency & throughput.
You can focus on building a working solution that adds value.
It is the fastest and easiest way to serve real-time predictions. Thus providing an excellent experience to the end user.
You can serve the predictions without hosting a powerful machine running the model 24/7.
Downsides:

The predictions are lagged as you consume them from storage (cache).
Most probably, you will generate redundant and unused predictions that cost you $$$ to make.
→ That is why many solutions start using a batch architecture and slowly move to a request-response or streaming solution.

The 7-Steps Drill Down
Lesson 1: Feature Engineering Pipeline

Feature engineering pipeline [Image by the Author].
Input: one or multiple raw data sources

In the FE pipeline, you will do the standard data preprocessing, such as:

cleaning data
standardize the data
vectorize the data
aggregations → create new features
❌ One thing to avoid is doing your transformations here, as they depend on your train-test split or adding redundancy in your feature store.

Note that in bigger projects, you will move the cleaning and standardization steps into your data pipeline.

Output: A new version of your features (e.g., “Features v007”).

→ Find out more in Lesson 1: Batch Serving. Feature Stores. Feature Engineering Pipelines.


Diagram of the architecture with the Lesson 1 components highlighted in blue [Image by the Author].
Tools FE: Pandas, Spark, Polars, PyTorch, TF

Feature Stores: Hopsworks, Feast

→ In the course, we used Pandas & Hopsworks.

Lesson 2: Training Pipeline

Training pipeline [Image by the Author].
Input: a given version of the features ingested from the feature store

First of all, you will create the training & testing dataset.

When experimenting, you will:

✅ experiment with various transformations
experiment to find the best model & hyperparameters
train the model
evaluate & test the model
→ Save the best config and model artifact in the model registry.

When automatically training, you will:

train the model using the best configuration found at the experimentation step
evaluate & test the model.
→ Save the new model artifact in the model registry.

Output: a new version of your configuration (e.g., “config v010”) and model artifact (e.g., “model v020”).

In the series, we trained a LightGBM model. We leveraged W&B’s hyperparameter tuning & experiment tracking features to find the best config. Also, we used Hopsworks as a model registry.

→ Find out more in Lesson 2: Training Pipelines. ML Platforms. Hyperparameter Tuning.


Diagram of the architecture with the Lesson 2 components highlighted in blue [Image by the Author].
Tools Training: Sklearn, PyTorch, TF

ML Platforms: W&B, MLFlow, Comet ML

→ In the course, we used Sklearn + Sktime and W&B.

Lesson 3: Inference / Batch Prediction Pipeline

Inference pipeline [Image by the Author].
Input: given version of the features and model artifact.

This step is relatively straightforward:

ingest the given version of the features from the feature store based on a time horizon [datetime_min, datetime_max]
load the version of the model artifact from the model registry
run the model in batch mode on all the loaded samples (this will take a while to run)
save the predictions to storage such as S3, GCS, Redis, etc.
Output: your predictions (if necessary, you can also version the predictions).

In the course, we forecasted the energy consumption levels for multiple time series for the next 24 hours and stored them in a GCS bucket.

→ Find out more in Lesson 3: Batch Prediction Pipeline. Package Python Modules with Poetry.


Diagram of the architecture with the Lesson 3 components highlighted in blue [Image by the Author].
Tools Inference: Sklearn, PyTorch, TF

Model Registry: W&B, MLFlow, Comet ML, Hopsworks

→ In the course, we used Sklearn + Sktime and Hopsworks.

Lesson 4: Orchestration

Example of a simple DAG orchestrated by Airflow [Image by the Author].
In the orchestration step, you will wrap all your tasks (in our case, the three pipelines) into a DAG (Directed Acyclic Graph).

A DAG is a fancy term for a sequence of steps (=tasks) that cannot form cycles.

Note that a DAG supports branches (e.g., start a plain training job or do hyperparameter tuning). You will be fine as long it is unidirectional (e.g., from left to right).

A DAG will allow you to:

run the whole pipeline with a single call
easily pass the versions of your resources (features, dataset, model, etc.) between the tasks
Observation: You can quickly introduce bugs when you have to connect 10+ scripts between each other. A DAG solves this.

An Orchestration tool will allow you to:

Schedule the DAG to run hourly, daily, weekly, etc.
monitor the DAG in a unified way
To conclude, an orchestration tool is the last piece of the puzzle for every batch system you want to build.

In the course, we scheduled the pipelines to run hourly.

→ Find out more in Lesson 4: Private PyPi Server. Orchestrate Everything with Airflow.


Diagram of the architecture with the Lesson 4 components highlighted in blue [Image by the Author].
Tools: Airflow, Metaflow, Prefect, ZenML, Dagster

→ In the course, we used Airflow.

Lesson 5: Data Validation & ML Monitoring
Until now, after your system was deployed, you were blind.

Yes, you had access to the DAGs logs, but that doesn’t tell you anything about the quality of the incoming data and model.

Adding data validation and ML monitoring features adds eyes and years across your system.

Data Validation

When ingesting new data, you must define a data contract that constantly checks your data streams. For example, if your application uses tabular data, you will check for the following:

table columns names
type
continuous → date range
discrete → value set
Nulls
You can add data validation before computing your features and before ingesting the features into the feature store.

In the course, we computed data validation only before ingesting the data into the feature store.

Standard SWE Monitoring

Revolves around the system’s health.

ML Monitoring

Revolves around ensuring a constant performance of the model.

Usually, after the model is trained, its performance in the real world will degrade after X time, and it needs to be retrained.

Mainly ML monitoring is all about alerting you when you must retrain the model to ensure constant performance.

You can do that by:

Computing real-time metrics, in case you can access real-time GT (which rarely happens).
Computing data, labels, and concept drift act as proxy metrics.
Detect outliers and edge cases that might mess up your model.
In the series, as we modeled time series with an hourly granularity, we considered we have almost real-time GT. Thus, we computed ~real-time metrics as our monitoring strategy.

→ Learn more in Lesson 5: Data Validation for Quality and Integrity using GE. Model Performance Continuous Monitoring.


Diagram of the architecture with the Lesson 5 components highlighted in blue [Image by the Author].
Tools Validation: Great Expectations (GE)

Tools Monitoring: Arize, Evidently, Seldon

→ In the course, we used GE and manually computed real-time metrics.

Lesson 6: Consumer App
This section is quite straightforward. After you implement your fancy ML pipeline, you have to use it, right?

A widespread scenario is to use the predictions in a web app, such as the standard backend + frontend architecture.

That is why we built an API server using FastAPI that reads the predictions from the GCS bucket and exposes them through a RESTful API.


Diagram of the architecture with the Lesson 6 components highlighted in blue [Image by the Author].
Also, we consumed the RESTful API using Streamlit and displayed them in a simple dashboard that shows the past observations and predictions for every time series.

→ Find out more in Lesson 6: Consume and Visualize your Model’s Predictions using FastAPI and Streamlit. Dockerize Everything.


Screenshot of the dashboard built during the course [Image by the Author].
Tools BE: FastAPI, BentoML, Modal

Tools FE: Streamlit, Gradio, Dash

→ In the course, we used FastAPI and Streamlit.

Lesson 7: Deploy & CI/CD
The final step in any software program is to deploy it and build a CI/CD pipeline around it.

We deployed the system on GCP. We used:

1x VM to run the ML Pipeline orchestrated by Airflow
1x VM to host the Web App (backend + frontend)
→ Note that the ML pipeline VM needs to be more powerful than the web app VM as it needs to run intensive steps such as training the model.

Also, we leveraged GitHub Actions to build 2 CI/CD pipelines:

ML Pipeline: Builds the Python packages using Poetry and pushes them to a private PiPy server from where Airflow will load them.
Web App: Builds the BE and FE docker images and updates the old images on the VM.
We could have hosted the docker images on the Docker registry and implemented a suite of tests to run before deploying the new release, but we wanted to keep it simple.

→ Learn more in Lesson 7: Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.


Diagram of the architecture with the Lesson 7 components highlighted in blue [Image by the Author].
Cloud Vendors: GCP, AWS, Azure, Paperspace

Tools CI/CD: GitHub Actions, Jenkins, Circle CI

→ In the course, we used GCP and GitHub Actions.

How Can We Adapt the Batch Architecture to an Online ML System?
Supercharge Your ML System: Use a Model Registry
The model registry is the critical component that decouples your offline pipeline (experimental/research phase) from your production pipeline.

Usually, when training your model, you use a static data source.

Using a feature engineering pipeline, you compute the necessary features used to train the model. These features will be stored inside a features store.

After processing your data, your training pipeline creates the training & testing splits and starts training the model.

The output of your training pipeline is the trained weights, also known as the model artifact.

This artifact will be pushed into the model registry under a new version that can easily be tracked.

Since this point, the new model artifact version can be pulled by any serving strategy:

batch,
request-response,
streaming.
Your inference pipeline doesn’t care how the model artifact was generated. It just has to know what model to use and how to transform the data into features.

Using a feature store, you can even unify the offline & online FE steps to avoid the training-serving skew.

Note that this strategy is independent of the type of model & hardware you use:

classic model (Sklearn, XGboost),
distributed system (Spark),
deep learning model (PyTorch).

Example of how to use a model registry with multiple deploying scenarios [Image by the Author].
Thus, using a model registry is a simple method to detach your experimentation from your production environment regardless of what framework or hardware you use.

Note: In the course, we deployed the model only in batch mode, but we want to emphasize how to adapt the architecture for other serving techniques.

The Merge of Batch and Streaming ML Pipelines
What happens if you want to introduce a real-time/streaming data source into your system?

Let’s get some context.

Until now, you used only a static data source to train your model & compute your features. But you find out that your business wants to use real-time news feeds as features for your model.

What do you do?

Well, it is a lot easier than it sounds.

You have to implement 2 main pipelines for your new streaming input source:

One that will quickly transform the raw data into features and make them accessible into the feature store to be used by the production services.
One that will store the raw data in the static raw data source (e.g., a warehouse) so it will be used later for experimentation and research.
Before ingesting into your system, the real-time data source might need an extra processing step to standardize and adapt the data format to your interface.

A standard strategy to perform step #1. and #2. use Kafka as your streaming platform + Flink/Kafka Streams as your streaming processing units.

For step #2. most of the time, you will have access to out-of-the-box data connectors that quickly load the real-time data into your static data storage (e.g., from Kafka to an S3 bucket or Big Query data warehouse).


Example of how to integrate your batch system with a streaming data source [Image by the Author].
Thus, to add a streaming data source to your current infrastructure, you need the following:

Kafka,
Flink/Kafka Streams,
to move your streaming data source into your static one,
to quickly compute features and load them into the feature store.
Thus, it isn’t hard — just a lot of infrastructure to set up.

Note: We haven’t practically implemented any streaming data source in the course. But it is good practice to conceptually show you how to adapt the batch architecture to other practical requirements.

Conclusion
To conclude, during “The Full Stack 7-Steps MLOps Framework” FREE course, you will learn how to build an end-to-end ML batch architecture that includes hands-on explanations for the following components:

A Feature Engineering Pipeline
A Training Pipeline
A Batch Prediction Pipeline
Orchestration
Data Validation & ML Monitoring
A Web App (Backend + Frontend)
Deployment & CI/CD
