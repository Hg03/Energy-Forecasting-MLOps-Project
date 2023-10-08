# Batch Prediction Pipeline. Package Python Modules with Poetry

This tutorial represents **lesson 3 out of a 7-lesson course** that will walk you step-by-step through how to **design, implement, and deploy an ML system** using **MLOps good practices**. During the course, you will build a production-ready model to forecast energy consumption levels for the next 24 hours across multiple consumer types from Denmark.

By the end of this course, you will understand all the fundamentals of designing, coding and deploying an ML system using a batch-serving architecture.

## Table of Contents:

- Course Introduction
- Course Lessons
- Data Source
- Lesson 3: Batch Prediction Pipeline. Package Python Modules with Poetry.
- Lesson 3: Code
- Conclusion
- References

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
  
If that sounds like a lot, don’t worry. After you cover this course, you will understand everything I said before. Most importantly, you will know WHY I used all these tools and how they work together as a system. If you want to get the most out of this course, I suggest you access the [GitHub repository](https://github.com/iusztinpaul/energy-forecasting/) containing all the lessons’ code. This course is designed to quickly read and replicate the code along the articles. By the end of the course, you will know how to implement the diagram below. Don’t worry if something doesn’t make sense to you. I will explain everything in detail.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/0b000205-6ac2-4da5-8f64-c5a422ebbee6)

By the **end of Lesson 3**, you will know how to implement and integrate the **batch prediction pipeline** and **package all the Python modules using Poetry**.

**Course Lessons:**

1. [Batch Serving. Feature Stores. Feature Engineering Pipelines.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
2. [Training Pipelines. ML Platforms. Hyperparameter Tuning.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_2.md)
3. **Batch Prediction Pipeline. Package Python Modules with Poetry.**
4. [Private PyPi Server. Orchestrate Everything with Airflow.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_4.md)
5. [Data Validation for Quality and Integrity using GE. Model Performance Continuous Monitoring.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_5.md)
6. [consume and Visualize your Model’s Predictions using FastAPI and Streamlit. Dockerize Everything.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_6.md)
7. [Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_7.md)
8. [Bonus - Behind the Scenes of an ‘Imperfect’ ML Project — Lessons and Insights](https://github.com/Hg03/mlops-paul/blob/main/Bonus.md)

If you want to grasp this lesson fully, we recommend you check out our previous lesson, which talks about designing a training pipeline that uses a feature store and an ML platform:

## The Goal of Lesson 3

This lesson will teach you how to build the batch prediction pipeline. Also, it will show you how to package into Python PyPi modules, using Poetry, all the code from the pipelines we have done so far in Lessons 1, 2, and 3. 👇

**Note:** In the next lesson, we will upload these Python modules into our own private PyPi server and install them from Airflow.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/7e1c0abb-5d73-46e7-b6ed-7e62dd5a3691)

If you recall from Lesson 1, a model can be deployed in the following ways:

- batch mode
- request-response (e.g., RESTful API or gRPC)
- streaming mode
- embedded

This course will _deploy the model in batch mode_.

We will discuss strategies to transition from batch to other methods when building the web app. You will see how natural it is.

**What are the main steps of deploying a model in batch mode, aka building a batch prediction pipeline?**

**Step 1:** You will load the features from the feature store in batch mode.

**Step 2:** You will load the trained model from the model registry (in our case, we use Hopsworks as a model registry).

**Step 3:** You will forecast the energy consumption levels for the next 24 hours.

**Step 4:** You will save the predictions in a GCP bucket.

After, various consumers will read the predictions from the GCP bucket and use them accordingly. In our case, we implemented a dashboard using FastAPI and Streamlit.

_Often, your initial deployment strategy will be in batch mode_.

**Why?**

Because doing so, you don't have to focus on restrictions such as latency and throughput. By saving your predictions into some storage, you can quickly make your model online.

Thus, batch mode is the easiest and fastest way of deploying your model while preserving a good experience for the end user of the applications.

A model is online when an application can access the predictions in real-time.

Note that the predictions are not made in real-time, only accessed in real-time (e.g., read from the storage).

The _biggest downside_ of using this method is that your predictions will have a degree of lag. For example, in our use case, you make and save the predictions for the next 24 hours. Let’s assume that 2 hours pass without any new predictions. Now, you have predictions only for the next 22 hours.

Where the number of predictions that you have to store is reasonable, you can bypass this issue by making the predictions often. In our example, we will make the predictions hourly — our data has a resolution of 1 hour. Thus, we solved the lag issue by constantly making and storing new predictions.

**But here comes the second problem with the batch prediction strategy**. Suppose the set of predictions is large. For example, you want to predict the recommendations for 1 million users with a database of 100 million items. Then, computing the predictions very often will be highly costly.

Then you have to consider using other serving methods strongly.

**But here is the catch.**

Your application probably won't start with a database of 1 million users and 100 million items. That means you can safely begin using a batch mode architecture and gradually shift to other methodologies when it makes sense.

That is what most people do!

To get an intuition on how to shift to other methods, [check out this article](https://medium.com/mlearning-ai/this-is-what-you-need-to-know-to-build-an-mlops-end-to-end-architecture-c0be1deaa3ce) to learn about a _standardized ML architecture suggested by Google Cloud_.

## Theoretical Concepts & Tools

**GCS:** GCS stands for Google Cloud Storage, which is Google's storage solution within GCP. It is similar to AWS S3 if you are more familiar with it.

You can write to GCS any file. In our course, we will write Pandas DataFrames as parquet files.

**GCS vs. Redis:** We choose to write our predictions in GCS because of 4 main reasons:

* Easy to setup
* No maintenance
* Access to the free tier
* We will also use GCP to deploy the code.

Redis is a popular choice for caching your predictions to be later accessed by various clients.

**Why?**

Because you can access the data at low latency, improving the users' experience.

It would have been a good choice, but we wanted to simplify things.

Also, it is good practice to write the predictions on GCS for long-term storage and cache them in Redis for real-time access.

**Poetry:** Poetry is my favorite Python virtual environment manager. It is similar to Conda, venv, and Pipenv. In my opinion, it is superior because:

* It offers you a **.lock** file that reflects the versions of all your sub-dependencies. Thus, replicating code is extremely easy and safe.
* You can quickly build your module directly using Poetry. No other setup is required.
* You can quickly deploy your module to a PiPy server using Poetry. No other setup is required, and more…

## Lesson 3: Code

[You can access the GitHub repository here](https://github.com/iusztinpaul/energy-forecasting).

**Note:** All the installation instructions are in the READMEs of the repository. Here we will jump straight to the code.

All the code within Lesson 3 is located under the [batch-prediction-pipeline](https://github.com/iusztinpaul/energy-forecasting/tree/main/batch-prediction-pipeline) folder.

The files under the [batch-prediction-pipeline](https://github.com/iusztinpaul/energy-forecasting/tree/main/batch-prediction-pipeline) folder are structured as follows:

![image](https://github.com/Hg03/mlops-paul/assets/69637720/4a4e59db-6cd4-40d2-9762-19b7a73f2f0d)

All the code is located under the [batch_prediction_pipeline directory](https://github.com/iusztinpaul/energy-forecasting/tree/main/batch-prediction-pipeline/batch_prediction_pipeline) (note the "_" instead of "-").

Directly storing credentials in your git repository is a huge security risk. That is why you will inject sensitive information using a **.env** file.

The **.env.default** is an example of all the variables you must configure. It is also helpful to store default values for attributes that are not sensitive (e.g., project name).

![image](https://github.com/Hg03/mlops-paul/assets/69637720/ddff8423-37ac-4f61-9260-911ac8f78dc8)

## Prepare Credentials

First of all, you have to create a **.env** file where you will add all our credentials.

I already showed you in Lesson 1 how to set up your **.env** file. Also, I explained in Lesson 1 how the variables from the **.env** file are loaded from your **ML_PIPELINE_ROOT_DIR** directory into a **SETTINGS** Python dictionary to be used throughout your code.

Thus, if you want to replicate what I have done, I strongly recommend checking out Lesson 1.

If you only want a light read, you can completely skip the **"Prepare Credentials"** step.

In **Lesson 3**, you will use two services:

- [Hopsworks](https://www.hopsworks.ai/)
- [GCP — Cloud Storage](https://cloud.google.com/storage)

[Hopsworks](https://www.hopsworks.ai/) (free)

We already showed you in Lesson 1 how to set up the credentials for **Hopsworks**. Please visit the "Prepare Credentials" section from Lesson 1, where we showed you in detail how to set up the API KEY for Hopsworks.

[GCP — Cloud Storage](https://cloud.google.com/storage) (free)

While replicating this course, you will stick to the GCP — Cloud Storage free tier. You can store up to 5GB for free in GCP — Cloud Storage, which is far more than enough for our use case.

This configuration step will be longer, but I promise that it is not complicated. By the way, you will learn the basics of using a cloud vendor such as GCP.

First, go to GCP and create a project called **"energy_consumption"** (or any other name). Afterward, go to your GCP project's "Cloud Storage" section and create a **non-public bucket** called **"hourly-batch-predictions"**. Pick any region, but just be aware of it — [official docs about creating a bucket on GCP](https://cloud.google.com/storage/docs/creating-buckets).

**NOTE:** You might need to pick different names due to constant changes to the platform’s rules. That is not an issue, just call them as you wish and change them in the **.env** file: GOOGLE_CLOUD_PROJECT (ours “energy_consumption”) and GOOGLE_CLOUD_BUCKET_NAME (ours “hourly-batch-predictions”).

![image](https://github.com/Hg03/mlops-paul/assets/69637720/680b8712-dbca-44b4-8a85-1c526042ab53)

Now you finished creating all your GCP resources. The last step is to create a way to have read & write access to the GCP bucket directly from your Python code.

You can easily do this using GCP service accounts. I don't want to hijack the whole article with GCP configurations. Thus, this [GCP official doc shows you how to create a service account](https://cloud.google.com/iam/docs/service-accounts-create).

_When creating the service account, be aware of one thing!_

Service accounts have attached different roles. A role is a way to configure your service account with various permissions.

Thus, you need to configure your service account to have read & write access the your **"hourly-batch-predictions"** bucket.

You can easily do that by choosing the **"Storage Object Admin"** role when creating your service account.

The final step is to find a way to authenticate with your newly created service account in your Python code.

You can easily do that by going to your service account and creating a JSON key. Again, here are the [official GCP docs that will show you how to create a JSON key for your service account](https://cloud.google.com/iam/docs/keys-create-delete).

_Again, keep in mind one thing!_

When creating the JSON key, you will download a JSON file.

After you download your JSON file, put it in a safe place and go to your **.env** file. There, change the value of GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH with your absolute path to the JSON file.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/1d2c0669-593c-4b0f-9802-b24efd19542b)

**NOTE:** Remember to change the GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_BUCKET_NAME variables with your names.

_Congratulations! You are done configuring GCS — Cloud Storage._

Now you have created a GCP project and bucket. Also, you have read & write access using your Python code through your service account. You log in with your service account with the help of the JSON file.

## Batch Prediction Pipeline — Main Function

As you can see, the main function follows the 4 steps of a batch prediction pipeline:

1. Loads data from the Feature Store in batch mode.
2. Loads the model from the model registry.
3. Makes the predictions.
4. It saves the predictions to the GCS bucket.

Most of the function is log lines 😆

Along these 4 main steps, you must load all the parameters from the metadata generated by previous steps, such as the **feature_view_version** and **model_version**.

Also, you have to get a reference to the Hopsworks Feature store.

After, you go straight to the 4 main steps that we will detail later in the tutorial 👇

```python
def predict(
    fh: int = 24,
    feature_view_version: Optional[int] = None,
    model_version: Optional[int] = None,
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
) -> None:
    """Main function used to do batch predictions.

    Args:
        fh (int, optional): forecast horizon. Defaults to 24.
        feature_view_version (Optional[int], optional): feature store feature view version. If None is provided, it will try to load it from the cached feature_view_metadata.json file.
        model_version (Optional[int], optional): model version to load from the model registry. If None is provided, it will try to load it from the cached train_metadata.json file.
        start_datetime (Optional[datetime], optional): start datetime used for extracting features for predictions. If None is provided, it will try to load it from the cached feature_pipeline_metadata.json file.
        end_datetime (Optional[datetime], optional): end datetime used for extracting features for predictions. If None is provided, it will try to load it from the cached feature_pipeline_metadata.json file.
    """

    if feature_view_version is None:
        feature_view_metadata = utils.load_json("feature_view_metadata.json")
        feature_view_version = feature_view_metadata["feature_view_version"]
    if model_version is None:
        train_metadata = utils.load_json("train_metadata.json")
        model_version = train_metadata["model_version"]
    if start_datetime is None or end_datetime is None:
        feature_pipeline_metadata = utils.load_json("feature_pipeline_metadata.json")
        start_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_start"],
            feature_pipeline_metadata["datetime_format"],
        )
        end_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_end"],
            feature_pipeline_metadata["datetime_format"],
        )

    logger.info("Connecting to the feature store...")
    project = hopsworks.login(
        api_key_value=settings.SETTINGS["FS_API_KEY"], project="energy_consumption"
    )
    fs = project.get_feature_store()
    logger.info("Successfully connected to the feature store.")

    logger.info("Loading data from feature store...")
    logger.info(f"Loading features from {start_datetime} to {end_datetime}.")
    X, y = data.load_data_from_feature_store(
        fs,
        feature_view_version,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    logger.info("Successfully loaded data from feature store.")

    logger.info("Loading model from model registry...")
    model = load_model_from_model_registry(project, model_version)
    logger.info("Successfully loaded model from model registry.")

    logger.info("Making predictions...")
    predictions = forecast(model, X, fh=fh)
    predictions_start_datetime = predictions.index.get_level_values(
        level="datetime_utc"
    ).min()
    predictions_end_datetime = predictions.index.get_level_values(
        level="datetime_utc"
    ).max()
    logger.info(
        f"Forecasted energy consumption from {predictions_start_datetime} to {predictions_end_datetime}."
    )
    logger.info("Successfully made predictions.")

    logger.info("Saving predictions...")
    save(X, y, predictions)
    logger.info("Successfully saved predictions.")
```

### Step 1: Loading Data From the Feature Store In Batch Mode

This step is similar to what we have done in Lesson 2 when loading data for training.

But this time, instead of downloading the data from a training dataset, we directly ask for a batch of data between a datetime range, using the **get_batch_data()** method.

Doing so allows us to time travel to our desired datetime range and ask for the features we need. This method makes batch inference extremely easy.

The last step is to prepare the indexes of the DataFrame as expected by **sktime** and to split it between X and y.

**Note:** This is an autoregressive process: we learn from past values of y to predict future values of y ( y = energy consumption levels). Thus, we will use only X as input to the model. We will use y only for visualization purposes.

```python
def load_data_from_feature_store(
    fs: FeatureStore,
    feature_view_version: int,
    start_datetime: datetime,
    end_datetime: datetime,
    target: str = "energy_consumption",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data for a given time range from the feature store.

    Args:
        fs: Feature store.
        feature_view_version: Feature view version.
        start_datetime: Start datetime.
        end_datetime: End datetime.
        target: Name of the target feature.

    Returns:
        Tuple of exogenous variables and the time series to be forecasted.
    """

    feature_view = fs.get_feature_view(
        name="energy_consumption_denmark_view", version=feature_view_version
    )
    data = feature_view.get_batch_data(start_time=start_datetime, end_time=end_datetime)

    # Set the index as is required by sktime.
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=[target])
    # Prepare the time series to be forecasted.
    y = data[[target]]

    return X, y
```

### Step 2: Loading the Model From the Model Registry

Loading a model from the Hopsworks model registry is extremely easy.

The function below has as a parameter a reference to the Hopsworks project and the version of the model we want to download.

Using these two variables, you get a reference to the model registry. Afterward, you get a reference to the model itself using its name. In this case, it is **best_model**.

Finally, you download the artifact/model and load it into memory.

The trick here is that your model is versioned. Thus, you always know what model you are using.

**Note:** We uploaded the **best_model** in the model registry using the training pipeline explained in Lesson 2. The training pipeline also provides us with a metadata dictionary that contains the latest model_version.

```python
def load_model_from_model_registry(project, model_version: int):
    """
    This function loads a model from the Model Registry.
    The model is downloaded, saved locally, and loaded into memory.
    """

    mr = project.get_model_registry()
    model_registry_reference = mr.get_model(name="best_model", version=model_version)
    model_dir = model_registry_reference.download()
    model_path = Path(model_dir) / "best_model.pkl"

    model = utils.load_model(model_path)

    return model
```

### Step 3: Forecast Energy Consumption Levels for the Next 24 Hours

**Sktime** makes forecasting extremely easy. The key line from the snippet below is **"predictions = model.predict(X=X_forecast)"**, which forecasts the energy consumption values for the next 24 hours.

The forecasting horizon of 24 hours was given when the model was trained. Thus, it already knows how many data points into the future to forecast.

Also, you have to prepare the exogenous variable **X_forecast**. In time series forecasting, an exogenous variable is a feature that you already know it will happen in the future. For example, a holiday. Thus, based on your training data X which contains all the area and consumer types IDs, you can generate the **X_forecast** variable by mapping the datetime range into the forecasting range.

```python
def forecast(model, X: pd.DataFrame, fh: int = 24):
    """
    Get a forecast of the total load for the given areas and consumer types.

    Args:
        model (sklearn.base.BaseEstimator): Fitted model that implements the predict method.
        X (pd.DataFrame): Exogenous data with area, consumer_type, and datetime_utc as index.
        fh (int): Forecast horizon.

    Returns:
        pd.DataFrame: Forecast of total load for each area, consumer_type, and datetime_utc.
    """

    all_areas = X.index.get_level_values(level=0).unique()
    all_consumer_types = X.index.get_level_values(level=1).unique()
    latest_datetime = X.index.get_level_values(level=2).max()

    start = latest_datetime + 1
    end = start + fh - 1
    fh_range = pd.date_range(
        start=start.to_timestamp(), end=end.to_timestamp(), freq="H"
    )
    fh_range = pd.PeriodIndex(fh_range, freq="H")

    index = pd.MultiIndex.from_product(
        [all_areas, all_consumer_types, fh_range],
        names=["area", "consumer_type", "datetime_utc"],
    )
    X_forecast = pd.DataFrame(index=index)
    X_forecast["area_exog"] = X_forecast.index.get_level_values(0)
    X_forecast["consumer_type_exog"] = X_forecast.index.get_level_values(1)

    predictions = model.predict(X=X_forecast)

    return predictions
```

### Step 4: Save the Predictions to the Bucket

The last component is the function that saves everything to the GCP bucket.

This step is relatively straightforward, and the hard part was to configure your bucket and access credentials.

We get a reference to the bucket, iterate through X, y & predictions and write them to the bucket as a blob.

**Note:** Besides the predictions, we also save X and y to have everything in one place to quickly access everything we need and nicely render them in the web app.

```python
def save(X: pd.DataFrame, y: pd.DataFrame, predictions: pd.DataFrame):
    """Save the input data, target data, and predictions to GCS."""

    # Get the bucket object from the GCS client.
    bucket = utils.get_bucket()

    # Save the input data and target data to the bucket.
    for df, blob_name in zip(
        [X, y, predictions], ["X.parquet", "y.parquet", "predictions.parquet"]
    ):
        logger.info(f"Saving {blob_name} to bucket...")
        utils.write_blob_to(
            bucket=bucket,
            blob_name=blob_name,
            data=df,
        )
        logger.info(f"Successfully saved {blob_name} to bucket.")
```

To get a reference to the bucket, you have to access the settings you configured at the beginning of the tutorial.

As you can see, you create a GCS client with the project name and the JSON credentials file path. Afterward, you can quickly get a reference to your given bucket.

Writing a blob to a bucket is highly similar to writing a regular file.

You get a reference to the blob you want to write and open the resource with **"with blob.open("wb") as f"**.

Note that you opened the blob in binary format.

You are writing the data in parquet format, as it is an excellent trade-off between storage size and writing & reading performance.

```python
def get_bucket(
    bucket_name: str = settings.SETTINGS["GOOGLE_CLOUD_BUCKET_NAME"],
    bucket_project: str = settings.SETTINGS["GOOGLE_CLOUD_PROJECT"],
    json_credentials_path: str = settings.SETTINGS[
        "GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON_PATH"
    ],
) -> storage.Bucket:
    """Get a Google Cloud Storage bucket.

    This function returns a Google Cloud Storage bucket that can be used to upload and download
    files from Google Cloud Storage.

    Args:
        bucket_name : str
            The name of the bucket to connect to.
        bucket_project : str
            The name of the project in which the bucket resides.
        json_credentials_path : str
            Path to the JSON credentials file for your Google Cloud Project.

    Returns
        storage.Bucket
            A storage bucket that can be used to upload and download files from Google Cloud Storage.
    """

    storage_client = storage.Client.from_service_account_json(
        json_credentials_path=json_credentials_path,
        project=bucket_project,
    )
    bucket = storage_client.bucket(bucket_name=bucket_name)

    return bucket


def write_blob_to(bucket: storage.Bucket, blob_name: str, data: pd.DataFrame):
    """Write a dataframe to a GCS bucket as a parquet file.

    Args:
        bucket (google.cloud.storage.Bucket): The bucket to write to.
        blob_name (str): The name of the blob to write to. Must be a parquet file.
        data (pd.DataFrame): The dataframe to write to GCS.
    """

    blob = bucket.blob(blob_name=blob_name)
    with blob.open("wb") as f:
        data.to_parquet(f)
```

## Package Python Modules with Poetry

[Poetry](https://python-poetry.org/) makes the building process extremely easy.

The first obvious step is to use Poetry as your virtual environment manager. That means you already have the **"pyproject.toml"** and **"poetry.lock"** files — we already provided these files for you.

Now, all you have to do is to go to your project at the same level as your Poetry files (the ones mentioned above — for example, go to your [batch-prediction-pipeline directory](https://github.com/iusztinpaul/energy-forecasting/tree/main/batch-prediction-pipeline) ) and run:

```python
poetry build
```

This will create a **dist** folder containing your package as a **wheel**. Now you can directly install your package using the wheel file or deploy it to a PyPi server.

To deploy it, configure your PyPi server credentials with the following:

```python
poetry config repositories.<my-pypi-server> <pypi server URL>
poetry config http-basic.<my-pypi-server> <username> <password>
```

Finally, deploy it using the following:

```python
poetry publish -r <my-pypi-server>
```

And that was it. I was amazed at how easy Poetry can make this process.

Otherwise, building and deploying your Python package is a tedious and lengthy process.

In Lesson 4, you will deploy your private PyPi server and deploy all the code you have written until this point using the commands I showed you above.

## Conclusion

Congratulations! You finished the **third lesson** from the **Full Stack 7-Steps MLOps Framework course**.

If you have reached this far, you know how to:

- choose the right architecture
- access data from the feature store in batch mode
- download your model from the model registry
- build an inference pipeline
- save your predictions to GCS

Now that you understand the power of using and implementing a batch prediction architecture, you can quickly serve models in real-time while paving your way for other fancier serving methods.

Check out Lesson 4 to learn about hosting your own private PyPi server and orchestrating all the pipelines using Airflow.
