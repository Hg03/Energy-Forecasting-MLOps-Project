This tutorial represents **lesson 2 out of a 7-lesson course** that will walk you step-by-step through how to **design, implement, and deploy an ML system** using **MLOps good practices**. During the course, you will build a production-ready model to forecast energy consumption levels for the next 24 hours across multiple consumer types from Denmark.

By the end of this course, you will understand all the fundamentals of designing, coding and deploying an ML system using a batch-serving architecture.

This course targets mid/advanced machine learning engineers who want to level up their skills by building their own end-to-end projects.

## Table of Contents:

- Course Introduction (As usual from lesson 1)
- Course Lessons (As usual from lesson 1)
- Data Source
- Lesson 2: Training Pipelines. ML Platforms. Hyperparameter Tuning.
- Lesson 2: Code
- Conclusion
- References

### Data Source

We used a free & open API that provides hourly energy consumption values for all the energy consumer types within Denmark.

They provide an intuitive interface where you can easily query and visualize the data. You can access the data [here](https://www.energidataservice.dk/tso-electricity/ConsumptionDE35Hour).

**The data has 4 main attributes:**

- **Hour UTC:** the UTC datetime when the data point was observed.
- **Price Area:** Denmark is divided into two price areas: DK1 and DK2 — divided by the Great Belt. DK1 is west of the Great Belt, and DK2 is east of the Great Belt.
- **Consumer Type:** The consumer type is the Industry Code DE35, owned and maintained by Danish Energy.
- **Total Consumption:** Total electricity consumption in kWh

**Note:** The observations have a lag of 15 days! But for our demo use case, that is not a problem, as we can simulate the same steps as it would be in real-time.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/edf9d03a-fa16-4b00-ada1-ec6370d02e8b)

The data points have an hourly resolution. For example: “2023–04–15 21:00Z”, “2023–04–15 20:00Z”, “2023–04–15 19:00Z”, etc.

We will model the data as multiple time series. Each unique **price area** and **consumer type tuple** represents its unique time series.

Thus, we will build a model that independently forecasts the energy consumption for the next 24 hours for every time series.

## The Goal of Lesson 2

This lesson will teach you how to build the training pipeline and use an ML platform, as shown in the diagram below 👇

![image](https://github.com/Hg03/mlops-paul/assets/69637720/1bf4927b-6d5d-4c54-9622-5bfc6b9e81ec)

More concretely, we will show you how to use the data from the Hopsworks feature store to train your model.

Also, we will show you how to build a forecasting model using LightGBM and Sktime that will predict the energy consumption levels for the next 24 hours between multiple consumer types across Denmark.

Another critical step we will cover is how to use W&B as an ML platform that will track your experiments, register your models & configurations as artifacts, and perform hyperparameter tuning to find the best configuration for your model.

Finally, based on the best config found in the hyperparameter tuning step, we will train the final model on the whole dataset and load it into the Hopsworks model registry to be used further by the batch prediction pipeline.

**NOTE:** This course is not about time series forecasting or hyperparameter tuning. This is an ML engineering course where I want to show you how multiple pieces come together into a single system. Thus, I will keep things straight to the point for the DS part of the code without going into too much detail.

## Theoretical Concepts & Tools

**Sktime:** Sktime is a Python package that provides tons of functionality for time series. It follows the same interface as Sklearn, hence its name. Using Sktime, we can quickly wrap LightGBM and perform forecasting for 24 hours in the future, cross-validation, and more. Sktime official documentation [3]

**LightGBM:** LightGBM is a boosting tree-based model. It is built on top of Gradient Boosting and XGBoost, offering performance and speed improvements. Starting with XGBoost or LightGBM is a common practice. LightGBM official documentation [4]

If you want to learn more about LightGBM, check out my article, where I explain in 15 minutes everything you need to know, from decision trees to LightGBM.

**ML Platform:** An ML platform is a tool that allows you to easily track your experiments, log metadata about your training, upload and version artifacts, data lineage and more. An ML platform is a must in any training pipeline. You can intuitively see an ML platform as your central research & experimentation hub.

**Weights & Biases:** W&B is a popular serverless ML platform. We choose them as our ML platform because of 3 main reasons:

1. their tool is fantastic & very intuitive to use
2. they provide a generous freemium version for personal research and projects
3. it is serverless — no pain in deploying & maintaining your tools

**Training Pipeline:** The training pipeline is a logical construct (a single script, an application, or more) that takes curated and validated data as input (a result from the data and feature engineering pipelines) and outputs a working model as an artifact. Usually, the model is uploaded into a model registry that can later be accessed by various inference pipelines (the batch prediction pipeline from our series is an example of a concrete implementation of an inference pipeline).

## Lesson 2: Code

[You can access the GitHub repository here](https://github.com/iusztinpaul/energy-forecasting).

**Note:** All the installation instructions are in the READMEs of the repository. Here we will jump straight to the code.

All the code within Lesson 2 is located under the [training-pipeline folder](https://github.com/iusztinpaul/energy-forecasting/tree/main/training-pipeline).

The files under the [training-pipeline](https://github.com/iusztinpaul/energy-forecasting/tree/main/training-pipeline) folder are structured as follows:

![image](https://github.com/Hg03/mlops-paul/assets/69637720/1650cac5-6b3f-4ba3-9f10-6cfb25e63804)

All the code is located under the training_pipeline directory (note the "_" instead of "-").

Directly storing credentials in your git repository is a huge security risk. That is why you will inject sensitive information using a **.env** file.

The **.env.default**is an example of all the variables you must configure. It is also helpful to store default values for attributes that are not sensitive (e.g., project name).

![image](https://github.com/Hg03/mlops-paul/assets/69637720/f959ff2c-0717-4f2b-9fed-8b622504a8d1)

## Prepare Credentials

First of all, we have to create a **.env** file where we will add all our credentials. I already showed you in Lesson 1 how to set up your **.env** file. Also, I explained in Lesson 1 how the variables from the **.env** file are loaded from your **ML_PIPELINE_ROOT_DIR** directory into a **SETTINGS** Python dictionary to be used throughout your code.

Thus, if you want to replicate what I have done, I strongly recommend checking out Lesson 1.

If you only want a light read, you can completely skip the **“Prepare Credentials”** step.

In Lesson 2, we will use two services:

- [Hopsworks](https://www.hopsworks.ai/)
- [Weights & Biases](https://wandb.ai)

**Hopsworks (free)**

We already showed you in Lesson 1 how to set up the credentials for **Hopsworks**. Please visit the "Prepare Credentials" section from Lesson 1, where we showed you in detail how to set up the API KEY for Hopsworks.

**Weights & Biases (free)**

To keep the lessons compact, we assume that you already read and applied the steps for preparing the credentials for **Hopsworks** from Lesson 1.

The good news is that 90% of the steps are similar to the ones for configuring **Hopsworks**, except for how you can get your API key from W&B.

First, create an account on W&B. After, create a team (aka entity) and a project (or use your default ones, if you have any).

**Then, check the image below to see how to get your own W&B API KEY 👇**

![image](https://github.com/Hg03/mlops-paul/assets/69637720/7e591a74-e6cf-45e5-ac4c-ca0813bdf2de)

Once you have all your W&B credentials, go to your **.env** file and replace them as follows:

- **WANDB_ENTITY:** your entity/team name (ours: “teaching-mlops”)
- **WANDB_PROJECT:** your project name (ours: “energy_consumption”)
- **WANDB_API_KEY:** your API key

## Loading the Data From the Feature Store

As always, the first step is to access the data used to train and test the model. We already have all the data in the Hopsworks feature store. Thus, downloading it becomes a piece of cake.

The code snippet below has the **load_dataset_from_feature_store() IO** function under the **training_pipeline/data.py** file. You will use this function to download the data for a given **feature_view_version** and **training_dataset_version**.

**NOTE:** By giving a specific data version, you will always know with what data you trained and evaluated the model. Thus, you can consistently reproduce your results.

Using the function below, we perform the following steps:

1. We access the Hopsworks feature store.
2. We get a reference to the given version of the feature view.
3. We get a reference to the given version of the training data.
4. We log to W&B all the metadata that relates to the used dataset.
5.Now that we downloaded the dataset, we run it through the **prepare_data()** function. We will detail it a bit later. For now, notice that we split the data between train and test.
6 .We log to W&B all the metadata related to how we split the dataset, plus some basic statistics for every split, such as split size and features.
   
Important observation:** Using W&B, you log all the metadata that describes how you extracted and prepared the data. By doing so, you can easily understand for every experiment the origin of its data.

By using **run.use_artifact("<artifact_name>")**, you can link different artifacts between them. In our example, by calling **run.use_artifact(“energy_consumption_denmark_feature_view:latest”)** we linked this W&B run with an artifact created in a different W&B run.

```python
def load_dataset_from_feature_store(
    feature_view_version: int, training_dataset_version: int, fh: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features from feature store.
    Args:
        feature_view_version (int): feature store feature view version to load data from
        training_dataset_version (int): feature store training dataset version to load data from
        fh (int, optional): Forecast horizon. Defaults to 24.
    Returns:
        Train and test splits loaded from the feature store as pandas dataframes.
    """

    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project="energy_consumption"
    )
    fs = project.get_feature_store()

    with init_wandb_run(
        name="load_training_data", job_type="load_feature_view", group="dataset"
    ) as run:
        feature_view = fs.get_feature_view(
            name="energy_consumption_denmark_view", version=feature_view_version
        )
        data, _ = feature_view.get_training_data(
            training_dataset_version=training_dataset_version
        )

        fv_metadata = feature_view.to_dict()
        fv_metadata["query"] = fv_metadata["query"].to_string()
        fv_metadata["features"] = [f.name for f in fv_metadata["features"]]
        fv_metadata["link"] = feature_view._feature_view_engine._get_feature_view_url(
            feature_view
        )
        fv_metadata["feature_view_version"] = feature_view_version
        fv_metadata["training_dataset_version"] = training_dataset_version

        raw_data_at = wandb.Artifact(
            name="energy_consumption_denmark_feature_view",
            type="feature_view",
            metadata=fv_metadata,
        )
        run.log_artifact(raw_data_at)

        run.finish()

    with init_wandb_run(
        name="train_test_split", job_type="prepare_dataset", group="dataset"
    ) as run:
        run.use_artifact("energy_consumption_denmark_feature_view:latest")

        y_train, y_test, X_train, X_test = prepare_data(data, fh=fh)

        for split in ["train", "test"]:
            split_X = locals()[f"X_{split}"]
            split_y = locals()[f"y_{split}"]

            split_metadata = {
                "timespan": [
                    split_X.index.get_level_values(-2).min(),
                    split_X.index.get_level_values(-2).max(),
                ],
                "dataset_size": len(split_X),
                "num_areas": len(split_X.index.get_level_values(0).unique()),
                "num_consumer_types": len(split_X.index.get_level_values(1).unique()),
                "y_features": split_y.columns.tolist(),
                "X_features": split_X.columns.tolist(),
            }
            artifact = wandb.Artifact(
                name=f"split_{split}",
                type="split",
                metadata=split_metadata,
            )
            run.log_artifact(artifact)

        run.finish()

    return y_train, y_test, X_train, X_test
```
Check out the video below to see how the W&B runs & artifacts look like in the W&B interface 👇

[Video](https://youtu.be/reBcVmAu1ss)

Now, let's dig into the **prepare_data()** function.

I want to highlight that in the **prepare_data()** function, we won't perform any feature engineering steps.

As you can see below, in this function, you will restructure the data to be compatible with the **sktime** interface, pick the target, and split the data.

The data is modeled for hierarchical time series, translating to multiple independent observations of the same variable in different contexts. In our example, we observe the energy consumption for various areas and energy consumption types.

Sktime, for hierarchical time series, expects the data to be modeled using multi-indexes, where the datetime index is the last. To learn more about hierarchical forecasting, check out [Sktime's official tutorial](https://github.com/sktime/sktime/blob/main/examples/01c_forecasting_hierarchical_global.ipynb).

Also, we can safely split the data using **sktime's temporal_train_test_split()** function. The test split has the length of the given **fh (=forecast horizon)**.

One key observation is that the test split isn't sampled randomly but based on the latest observation. For example, if you have data from the 1st of May 2023 until the 7th of May 2023 with a frequency of 1 hour, then the test split with a length of 24 hours will contain all the values from the last day of the data, which is 7th of May 2023.

```python
def prepare_data(
    data: pd.DataFrame, target: str = "energy_consumption", fh: int = 24
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Structure the data for training:
    - Set the index as is required by sktime.
    - Prepare exogenous variables.
    - Prepare the time series to be forecasted.
    - Split the data into train and test sets.
    """

    # Set the index as is required by sktime.
    data["datetime_utc"] = pd.PeriodIndex(data["datetime_utc"], freq="H")
    data = data.set_index(["area", "consumer_type", "datetime_utc"]).sort_index()

    # Prepare exogenous variables.
    X = data.drop(columns=[target])
    # Prepare the time series to be forecasted.
    y = data[[target]]

    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=fh)

    return y_train, y_test, X_train, X_test
```

## Building the Forecasting Model

### Baseline model

Firstly, you will create a naive baseline model to use as a reference. This model predicts the last value based on a given seasonal periodicity.

For example, if **seasonal_periodicity = 24 hours**, it will return the value from **"present - 24 hours"**.

Using a baseline is a healthy practice that helps you compare your fancy ML model to something simpler. The ML model is useless if you can't beat the baseline model with your fancy model.

```python
def build_baseline_model(seasonal_periodicity: int):
    """Builds a naive forecaster baseline model using Sktime that predicts the last value given a seasonal periodicity."""

    return NaiveForecaster(sp=seasonal_periodicity)
```

Fancy ML model

We will build the model using Sktime and LightGBM.

Check out [Sktime documentation](https://www.sktime.net/en/latest/index.html) and [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html) here.

If you are into time series, check out this [Forecasting with Sktime tutorial](https://www.sktime.net/en/latest/examples/01_forecasting.html#1.-Basic-forecasting-workflows). If you only want to understand the system's big picture, you can continue.

LightGBM will be your regressor that learns patterns within the data and forecasts future values.

Using the **WindowSummarizer** class from **Sktime**, you can quickly compute lags and mean & standard deviation for various windows.

For example, for the lag, we provide a default value of **list(range(1, 72 + 1))**, which translates to "compute the lag for the last 72 hours".

Also, as an example of the mean lag, we have the default value of **[[1, 24], [1, 48], [1, 72]]**. For example, **[1, 24]** translates to a lag of 1 and a window size of 24, meaning it will compute the mean in the last 24 days. Thus, in the end, for **[[1, 24], [1, 48], [1, 72]]**, you will have the mean for the last 24, 46, and 72 days.

The same principle applies to the standard deviation values. [Check out this doc to learn more](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.transformations.series.summarize.WindowSummarizer.html?highlight=windowsummarizer).

You wrap the LightGBM model using the **make_reduction()** function from **Sktime**. By doing so, you can easily attach the **WindowSummarizer** you initialized earlier. Also, by specifying **strategy = "recursive"**, you can easily forecast multiple values into the future using a recursive paradigm. For example, if you want to predict 3 hours into the future, the model will first forecast the value for T + 1. Afterward, it will use as input the value it forecasted at T + 1 to forecast the value at T + 2, and so on…

Finally, we will build the **ForecastingPipeline** where we will attach two transformers:

- **transformers.AttachAreaConsumerType()**: a custom transformer that takes the area and consumer type from the index and adds it as an exogenous variable. We will show you how we defined it.
- **DateTimeFeatures()**: a transformer from Sktime that computes different datetime-related exogenous features. In our case, we used only the day of the week and the hour of the day as additional features.

**Note** that these transformers are similar to the ones from Sklearn, as Sktime kept the same interface and design. Using transformers is a critical step in designing modular models. To learn more about Sklearn transformers and pipelines, check out my article about [How to Quickly Design Advanced Sklearn Pipelines](https://towardsdatascience.com/how-to-quickly-design-advanced-sklearn-pipelines-3cc97b59ce16).

Finally, we initialized the hyperparameters of the pipeline and model with the given configuration.

```python
def build_model(config: dict):
    """
    Build an Sktime model using the given config.
    It supports defaults for windowing the following parameters:
    - lag: list(range(1, 72 + 1))
    - mean: [[1, 24], [1, 48], [1, 72]]
    - std: [[1, 24], [1, 48], [1, 72]]
    """

    lag = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__lag",
        list(range(1, 72 + 1)),
    )
    mean = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__mean",
        [[1, 24], [1, 48], [1, 72]],
    )
    std = config.pop(
        "forecaster_transformers__window_summarizer__lag_feature__std",
        [[1, 24], [1, 48], [1, 72]],
    )
    n_jobs = config.pop("forecaster_transformers__window_summarizer__n_jobs", 1)
    window_summarizer = WindowSummarizer(
        **{"lag_feature": {"lag": lag, "mean": mean, "std": std}},
        n_jobs=n_jobs,
    )

    regressor = lgb.LGBMRegressor()
    forecaster = make_reduction(
        regressor,
        transformers=[window_summarizer],
        strategy="recursive",
        pooling="global",
        window_length=None,
    )

    pipe = ForecastingPipeline(
        steps=[
            ("attach_area_and_consumer_type", transformers.AttachAreaConsumerType()),
            (
                "daily_season",
                DateTimeFeatures(
                    manual_selection=["day_of_week", "hour_of_day"],
                    keep_original_columns=True,
                ),
            ),
            ("forecaster", forecaster),
        ]
    )
    pipe = pipe.set_params(**config)

    return pipe
```
The **AttachAreaConsumerType** transformer is quite easy to comprehend. We implemented it as an example to show what is possible.

Long story short, it just copies the values from the index into its own column.

```python
class AttachAreaConsumerType(BaseTransformer):
    """Transformer used to extract the area and consumer type from the index to the input data."""

    _tags = {
        "capability:inverse_transform": True,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": CORE_MTYPES,  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
        "handles-missing-data": True,  # can estimator handle missing data?
    }

    def _transform(self, X, y=None):
        X["area_exog"] = X.index.get_level_values(0)
        X["consumer_type_exog"] = X.index.get_level_values(1)

        return X

    def _inverse_transform(self, X, y=None):
        X = X.drop(columns=["area_exog", "consumer_type_exog"])

        return X
```

### IMPORTANT OBSERVATION — DESIGN DECISION

As you can see, all the feature engineering steps are built-in into the forecasting pipeline object.

You might ask: "But why? By doing so, don't we keep the feature engineering logic in the training pipeline?"

Well, yes… and no…

We indeed defined the forecasting pipeline in the training script, but the key idea is that we will save the whole forecasting pipeline to the model registry.

Thus, when we load the model, we will also load all the preprocessing and postprocessing steps included in the forecasting pipeline.

This means all the feature engineering is encapsulated in the forecasting pipeline, and we can safely treat it as a black box.

This is one way to store the transformation + the raw data in the feature store, as discussed in Lesson 1.

We could have also stored the transformation functions independently in the feature store, but composing a single pipeline object is cleaner.

## Hyperparameter Tuning

**How to use W&B sweeps**

You will use W&B to perform hyperparameter tuning. They provide all the methods you need. Starting from a regular Grid Search until a Bayesian Search.

W&B uses sweeps to do hyperparameter tuning. A sweep is a fancy word for a single experiment within multiple experiments based on your hyperparameter search space.

We will use the MAPE (mean absolute percentage error) metric to compare experiments to find the best hyperparameter configuration. We chose MAPE over MAE or RMSE because the values are normalized between [0, 1], thus making it easier to analyze.

Check out the video below to see how the sweeps board looks in W&B 👇
[Video](https://youtu.be/ioYCcLQmBrU)

Now that we understand our goal let's look at the code under the **training_pipeline/hyperparamter_tuning.py** file.

As you can see in the function below, we load the dataset from the feature store for a specific feature_view_version and a training_dataset_version.

Using solely the training data, we start the hyperparameter optimization.

**Note:** It is essential that you don't use your test data for your hyperparameter optimization search. Otherwise, you risk overfitting your test split, and your model will not generalize. Your test split should be used only for the final decision.

Finally, we save the metadata of the run, which contains the **sweep_id** of the search.

```python
def run(
    fh: int = 24,
    feature_view_version: Optional[int] = None,
    training_dataset_version: Optional[int] = None,
) -> dict:
    """Run hyperparameter optimization search.
    Args:
        fh (int, optional): Forecasting horizon. Defaults to 24.
        feature_view_version (Optional[int], optional): feature store - feature view version.
             If none, it will try to load the version from the cached feature_view_metadata.json file. Defaults to None.
        training_dataset_version (Optional[int], optional): feature store - feature view - training dataset version.
            If none, it will try to load the version from the cached feature_view_metadata.json file. Defaults to None.
    Returns:
        dict: Dictionary containing metadata about the hyperparameter optimization run.
    """

    feature_view_metadata = utils.load_json("feature_view_metadata.json")
    if feature_view_version is None:
        feature_view_version = feature_view_metadata["feature_view_version"]
    if training_dataset_version is None:
        training_dataset_version = feature_view_metadata["training_dataset_version"]

    y_train, _, X_train, _ = load_dataset_from_feature_store(
        feature_view_version=feature_view_version,
        training_dataset_version=training_dataset_version,
        fh=fh,
    )

    sweep_id = run_hyperparameter_optimization(y_train, X_train, fh=fh)

    metadata = {"sweep_id": sweep_id}
    utils.save_json(metadata, file_name="last_sweep_metadata.json")

    return metadata
```

Now, let's look at the **run_hyperparameter_optimization()** function, which takes the training data, creates a new sweep and starts a W&B agent.

Within a single sweep run, we build the model and train the model using cross-validation.

As you can see, the config is provided by W&B based on the given hyperparameter search space (we will explain this in a bit). Also, we log the config as an artifact to access it later.

```python
def run_hyperparameter_optimization(
    y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int
):
    """Runs hyperparameter optimization search using W&B sweeps."""

    sweep_id = wandb.sweep(
        sweep=gridsearch_configs.sweep_configs, project=SETTINGS["WANDB_PROJECT"]
    )

    wandb.agent(
        project=SETTINGS["WANDB_PROJECT"],
        sweep_id=sweep_id,
        function=partial(run_sweep, y_train=y_train, X_train=X_train, fh=fh),
    )

    return sweep_id


def run_sweep(y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int):
    """Runs a single hyperparameter optimization step (train + CV eval) using W&B sweeps."""

    with init_wandb_run(
        name="experiment", job_type="hpo", group="train", add_timestamp_to_name=True
    ) as run:
        run.use_artifact("split_train:latest")

        config = wandb.config
        config = dict(config)
        model = build_model(config)

        model, results = train_model_cv(model, y_train, X_train, fh=fh)
        wandb.log(results)

        metadata = {
            "experiment": {"name": run.name, "fh": fh},
            "results": results,
            "config": config,
        }
        artifact = wandb.Artifact(
            name=f"config",
            type="model",
            metadata=metadata,
        )
        run.log_artifact(artifact)

        run.finish()
```

In our example, we used a simple grid search to perform hyperparameter tuning.

As you can see below, we created a Python dictionary called **sweep_config **with the method, the metric to minimize, and the parameters to search for.

[Check out W&B official docs to learn more about sweeps](https://docs.wandb.ai/guides/sweeps).

```python
sweep_configs = {
    "method": "grid",
    "metric": {"name": "validation.MAPE", "goal": "minimize"},
    "parameters": {
        "forecaster__estimator__n_jobs": {"values": [-1]},
        "forecaster__estimator__n_estimators": {"values": [1000, 2000, 2500]},
        "forecaster__estimator__learning_rate": {"values": [0.1, 0.15]},
        "forecaster__estimator__max_depth": {"values": [-1, 5]},
        "forecaster__estimator__reg_lambda": {"values": [0, 0.01, 0.015]},
        "daily_season__manual_selection": {"values": [["day_of_week", "hour_of_day"]]},
        "forecaster_transformers__window_summarizer__lag_feature__lag": {
            "values": [list(range(1, 73))]
        },
        "forecaster_transformers__window_summarizer__lag_feature__mean": {
            "values": [[[1, 24], [1, 48], [1, 72]]]
        },
        "forecaster_transformers__window_summarizer__lag_feature__std": {
            "values": [[[1, 24], [1, 48]]]
        },
        "forecaster_transformers__window_summarizer__n_jobs": {"values": [1]},
    },
}
```

Note: With a few tweaks, you can quickly run multiple W&B agents in parallel within a single sweep. Thus, speeding up the hyperparameter tuning drastically. [Check out their docs if you want to learn more](https://docs.wandb.ai/guides/sweeps/parallelize-agents).

**How to do cross-validation with time series data**

So, I highlighted that it is critical to do hyperparameter-tuning only using the training dataset.

But then, on what split should you compute your metrics?

Well, you will be using cross-validation adapted to time series.

As shown in the image below, we used a 3-fold cross-validation technique. The key idea is that because you are using time series data, you can't pick the whole dataset for every fold. It makes sense, as you can't learn from the future to predict the past.

Thus, using the same principles as when we split the data between train and test, we sample 1/3 from the beginning of the dataset, where the **forecasting horizon (the orange segment)** is used to compute the validation metric. The next fold takes 2/3, and the last one 3/3 of the dataset.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/9088872c-7f06-4c1b-ae17-cfb1cdc1decc)

Once again, **Sktime** makes our lives easier. Using the **ExpandingWindowSplitter** class and **cv_evaluate() **function, you can quickly train and evaluate the model using the specified cross-validation strategy — official docs here [8]

In the end, we restructured the **results** DataFrame, which the **cv_evaluate()** function returned to fit our interface.

```python
def train_model_cv(
    model, y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int, k: int = 3
):
    """Train and evaluate the given model using cross-validation."""

    data_length = len(y_train.index.get_level_values(-1).unique())
    assert data_length >= fh * 10, "Not enough data to perform a 3 fold CV."

    cv_step_length = data_length // k
    initial_window = max(fh * 3, cv_step_length - fh)
    cv = ExpandingWindowSplitter(
        step_length=cv_step_length, fh=np.arange(fh) + 1, initial_window=initial_window
    )
    render_cv_scheme(cv, y_train)

    results = cv_evaluate(
        forecaster=model,
        y=y_train,
        X=X_train,
        cv=cv,
        strategy="refit",
        scoring=MeanAbsolutePercentageError(symmetric=False),
        error_score="raise",
        return_data=False,
    )

    results = results.rename(
        columns={
            "test_MeanAbsolutePercentageError": "MAPE",
            "fit_time": "fit_time",
            "pred_time": "prediction_time",
        }
    )
    mean_results = results[["MAPE", "fit_time", "prediction_time"]].mean(axis=0)
    mean_results = mean_results.to_dict()
    results = {"validation": mean_results}

    logger.info(f"Validation MAPE: {results['validation']['MAPE']:.2f}")
    logger.info(f"Mean fit time: {results['validation']['fit_time']:.2f} s")
    logger.info(f"Mean predict time: {results['validation']['prediction_time']:.2f} s")

    return model, results
```

Excellent, now you finished running your hyperparameter tuning step using W&B sweeps.

At the end of this step, we have a **sweep_id** that has attached multiple experiments, where each experiment has a **config artifact**.

Now we have to parse this information and create a **best_config artifact**.

## Upload the Best Configuration from the Hyperparameter Tuning Search

Using the [training_pipeline/best_config.py](https://github.com/iusztinpaul/energy-forecasting/blob/main/training-pipeline/training_pipeline/best_config.py) script, we will parse all the experiments for the given **sweep_id** and find the best experiment with the lowest MAPE validation score.

Fortunately, this is done automatically by W&B when we call the **best_run()** function. After, you resume the **best_run** and rename the run to **best_experiment**.

Also, you upload the config attached to the best configuration into its artifact called best_config.

Later, we will use this artifact to train models from scratch as often as we want.

```python
def upload(sweep_id: Optional[str] = None):
    """Upload the best config from the given sweep to the "best_experiment" wandb Artifact.
    Args:
        sweep_id (Optional[str], optional): Sweep ID to look for the best config. If None, it will look for the last sweep in the cached last_sweep_metadata.json file. Defaults to None.
    """

    if sweep_id is None:
        last_sweep_metadata = utils.load_json("last_sweep_metadata.json")
        sweep_id = last_sweep_metadata["sweep_id"]

        logger.info(f"Loading sweep_id from last_sweep_metadata.json with {sweep_id=}")

    api = wandb.Api()
    sweep = api.sweep(
        f"{SETTINGS['WANDB_ENTITY']}/{SETTINGS['WANDB_PROJECT']}/{sweep_id}"
    )
    best_run = sweep.best_run()

    with utils.init_wandb_run(
        name="best_experiment",
        job_type="hpo",
        group="train",
        run_id=best_run.id,
        resume="must",
    ) as run:
        run.use_artifact("config:latest")

        best_config = dict(run.config)

        logger.info(f"Best run {best_run.name}")
        logger.info("Best run config:")
        logger.info(best_config)
        logger.info(
            f"Best run = {best_run.name} with results {dict(run.summary['validation'])}"
        )

        config_path = OUTPUT_DIR / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(best_config, f, indent=4)

        artifact = wandb.Artifact(
            name="best_config",
            type="model",
            metadata={"results": {"validation": dict(run.summary["validation"])}},
        )
        artifact.add_file(str(config_path))
        run.log_artifact(artifact)

        run.finish()
```

Now you have the **best_config** artifact that tells you precisely what hyperparameters you should use to train your final model on.

## Train the Final Model Using the Best Configuration

Finally, training and loading the final model to the model registry is the last piece of the puzzle.

Within the **from_best_config()** function from the [training_pipeline/train.py](https://github.com/iusztinpaul/energy-forecasting/blob/main/training-pipeline/training_pipeline/train.py) file, we perform the following steps:

1. Load the data from Hopsworks.
2. Initialize a W&B run.
3. Load the best_config artifact.
4. Build the baseline model.
5. Train and evaluate the baseline model on the test split.
6. Build the fancy model using the latest best configuration.
7. Train and evaluate the fancy model on the test split.
8. Render the results to see how they perform visually.
9. Retrain the model on the whole dataset. This is critical for time series models as you must retrain them until the present moment to forecast the future.
10. Forecast future values.
11. Render the forecasted values.
12. Save the best model as an Artifact in W&B
13. Save the best model in the Hopsworks' model registry

**Note:** You can either use W&B Artifacts as a model registry or directly use the Hopsworks model registry feature. We will show you how to do it both ways.

Notice how we used **wandb.log()** to upload to W&B all the variables of interest.

```python
def from_best_config(
    fh: int = 24,
    feature_view_version: Optional[int] = None,
    training_dataset_version: Optional[int] = None,
) -> dict:
    """Train and evaluate on the test set the best model found in the hyperparameter optimization run.
    After training and evaluating it uploads the artifacts to wandb & hopsworks model registries.
    Args:
        fh (int, optional): Forecasting horizon. Defaults to 24.
        feature_view_version (Optional[int], optional): feature store - feature view version.
             If none, it will try to load the version from the cached feature_view_metadata.json file. Defaults to None.
        training_dataset_version (Optional[int], optional): feature store - feature view - training dataset version.
            If none, it will try to load the version from the cached feature_view_metadata.json file. Defaults to None.
    Returns:
        dict: Dictionary containing metadata about the training experiment.
    """

    feature_view_metadata = utils.load_json("feature_view_metadata.json")
    if feature_view_version is None:
        feature_view_version = feature_view_metadata["feature_view_version"]
    if training_dataset_version is None:
        training_dataset_version = feature_view_metadata["training_dataset_version"]

    y_train, y_test, X_train, X_test = load_dataset_from_feature_store(
        feature_view_version=feature_view_version,
        training_dataset_version=training_dataset_version,
        fh=fh,
    )

    training_start_datetime = y_train.index.get_level_values("datetime_utc").min()
    training_end_datetime = y_train.index.get_level_values("datetime_utc").max()
    testing_start_datetime = y_test.index.get_level_values("datetime_utc").min()
    testing_end_datetime = y_test.index.get_level_values("datetime_utc").max()
    logger.info(
        f"Training model on data from {training_start_datetime} to {training_end_datetime}."
    )
    logger.info(
        f"Testing model on data from {testing_start_datetime} to {testing_end_datetime}."
    )
    # Loading predictions from 2023-04-06 22:00:00 to 2023-04-07 21:00:00.

    with utils.init_wandb_run(
        name="best_model",
        job_type="train_best_model",
        group="train",
        reinit=True,
        add_timestamp_to_name=True,
    ) as run:
        run.use_artifact("split_train:latest")
        run.use_artifact("split_test:latest")
        # Load the best config from sweep.
        best_config_artifact = run.use_artifact(
            "best_config:latest",
            type="model",
        )
        download_dir = best_config_artifact.download()
        config_path = Path(download_dir) / "best_config.json"
        with open(config_path) as f:
            config = json.load(f)
        # Log the config to the experiment.
        run.config.update(config)

        # # Baseline model
        baseline_forecaster = build_baseline_model(seasonal_periodicity=fh)
        baseline_forecaster = train_model(baseline_forecaster, y_train, X_train, fh=fh)
        _, metrics_baseline = evaluate(baseline_forecaster, y_test, X_test)
        slices = metrics_baseline.pop("slices")
        for k, v in metrics_baseline.items():
            logger.info(f"Baseline test {k}: {v}")
        wandb.log({"test": {"baseline": metrics_baseline}})
        wandb.log({"test.baseline.slices": wandb.Table(dataframe=slices)})

        # Build & train best model.
        best_model = build_model(config)
        best_forecaster = train_model(best_model, y_train, X_train, fh=fh)

        # Evaluate best model
        y_pred, metrics = evaluate(best_forecaster, y_test, X_test)
        slices = metrics.pop("slices")
        for k, v in metrics.items():
            logger.info(f"Model test {k}: {v}")
        wandb.log({"test": {"model": metrics}})
        wandb.log({"test.model.slices": wandb.Table(dataframe=slices)})

        # Render best model on the test set.
        results = OrderedDict({"y_train": y_train, "y_test": y_test, "y_pred": y_pred})
        render(results, prefix="images_test")

        # Update best model with the test set.
        # NOTE: Method update() is not supported by LightGBM + Sktime. Instead we will retrain the model on the entire dataset.
        # best_forecaster = best_forecaster.update(y_test, X=X_test)
        best_forecaster = train_model(
            model=best_forecaster,
            y_train=pd.concat([y_train, y_test]).sort_index(),
            X_train=pd.concat([X_train, X_test]).sort_index(),
            fh=fh,
        )
        X_forecast = compute_forecast_exogenous_variables(X_test, fh)
        y_forecast = forecast(best_forecaster, X_forecast)
        logger.info(
            f"Forecasted future values for renderin between {y_test.index.get_level_values('datetime_utc').min()} and {y_test.index.get_level_values('datetime_utc').max()}."
        )
        results = OrderedDict(
            {
                "y_train": y_train,
                "y_test": y_test,
                "y_forecast": y_forecast,
            }
        )
        # Render best model future forecasts.
        render(results, prefix="images_forecast")

        # Save best model.
        save_model_path = OUTPUT_DIR / "best_model.pkl"
        utils.save_model(best_forecaster, save_model_path)
        metadata = {
            "experiment": {
                "fh": fh,
                "feature_view_version": feature_view_version,
                "training_dataset_version": training_dataset_version,
                "training_start_datetime": training_start_datetime.to_timestamp().isoformat(),
                "training_end_datetime": training_end_datetime.to_timestamp().isoformat(),
                "testing_start_datetime": testing_start_datetime.to_timestamp().isoformat(),
                "testing_end_datetime": testing_end_datetime.to_timestamp().isoformat(),
            },
            "results": {"test": metrics},
        }
        artifact = wandb.Artifact(name="best_model", type="model", metadata=metadata)
        artifact.add_file(str(save_model_path))
        run.log_artifact(artifact)

        run.finish()
        artifact.wait()

    model_version = attach_best_model_to_feature_store(
        feature_view_version, training_dataset_version, artifact
    )

    metadata = {"model_version": model_version}
    utils.save_json(metadata, file_name="train_metadata.json")

    return metadata
```

Check out this video to visually see how we use W&B as an experiment tracker 👇

[Video](https://youtu.be/cbvTPsLRiCM)

## Train & evaluate the model.

To train any **Sktime** model, we implemented this general function that takes in any model, the data, and the forecast horizon.

```python
def train_model(model, y_train: pd.DataFrame, X_train: pd.DataFrame, fh: int):
    """Train the forecaster on the given training set and forecast horizon."""

    fh = np.arange(fh) + 1
    model.fit(y_train, X=X_train, fh=fh)

    return model
```

Using the method below, we evaluated the model on the test split using both aggregated metrics and slices over all the unique combinations of areas and consumer types.

By evaluating the model on slices, you can quickly investigate for fairness and bias.

As you can see, most of the heavy lifting, such as the implementation of MAPE and RMSPE, is directly accessible from **Sktime**.

```python
def evaluate(
    forecaster, y_test: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, dict]:
    """Evaluate the forecaster on the test set by computing the following metrics:
        - RMSPE
        - MAPE
        - Slices: RMSPE, MAPE
    Args:
        forecaster: model following the sklearn API
        y_test (pd.DataFrame): time series to forecast
        X_test (pd.DataFrame): exogenous variables
    Returns:
        The predictions as a pd.DataFrame and a dict of metrics.
    """

    y_pred = forecaster.predict(X=X_test)

    # Compute aggregated metrics.
    results = dict()
    rmspe = mean_squared_percentage_error(y_test, y_pred, squared=False)
    results["RMSPE"] = rmspe
    mape = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    results["MAPE"] = mape

    # Compute metrics per slice.
    y_test_slices = y_test.groupby(["area", "consumer_type"])
    y_pred_slices = y_pred.groupby(["area", "consumer_type"])
    slices = pd.DataFrame(columns=["area", "consumer_type", "RMSPE", "MAPE"])
    for y_test_slice, y_pred_slice in zip(y_test_slices, y_pred_slices):
        (area_y_test, consumer_type_y_test), y_test_slice_data = y_test_slice
        (area_y_pred, consumer_type_y_pred), y_pred_slice_data = y_pred_slice

        assert (
            area_y_test == area_y_pred and consumer_type_y_test == consumer_type_y_pred
        ), "Slices are not aligned."

        rmspe_slice = mean_squared_percentage_error(
            y_test_slice_data, y_pred_slice_data, squared=False
        )
        mape_slice = mean_absolute_percentage_error(
            y_test_slice_data, y_pred_slice_data, symmetric=False
        )

        slice_results = pd.DataFrame(
            {
                "area": [area_y_test],
                "consumer_type": [consumer_type_y_test],
                "RMSPE": [rmspe_slice],
                "MAPE": [mape_slice],
            }
        )
        slices = pd.concat([slices, slice_results], ignore_index=True)

    results["slices"] = slices

    return y_pred, results
```

## Render the results

Using **Sktime**, you can quickly render various time series into a single plot.

As shown in the video above, we rendered the results for every (area, consumer_type) combination in the W&B experiment tracker.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/7d15c642-93ef-4dee-8762-d7ba98372520)

![image](https://github.com/Hg03/mlops-paul/assets/69637720/c299b291-8808-4b84-96d1-3d87f21bb677)

## Upload the model to the model registry

The last step is to upload the model to a model registry. After the model is uploaded, it will be downloaded and used by our batch prediction pipeline.

During the experiment, we already uploaded the model as a W&B Artifact. If you plan to have dependencies with W&B in your applications, using it directly from there is perfectly fine.

But we wanted to keep the batch prediction pipeline dependent only on Hopsworks.

Thus, we used Hopswork's model registry feature.

In the following code, based on the given **best_model_artifact**, we added a tag to the Hopsworks feature view to link the two. This is helpful for debugging.

Finally, we downloaded the best model weights and loaded them to the Hopsworks model registry using the **mr.python.create_model()** method.

```python
def attach_best_model_to_feature_store(
    feature_view_version: int,
    training_dataset_version: int,
    best_model_artifact: wandb.Artifact,
) -> int:
    """Adds the best model artifact to the model registry."""

    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project="energy_consumption"
    )
    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(
        name="energy_consumption_denmark_view", version=feature_view_version
    )

    # Attach links to the best model artifact in the feature view and the training dataset of the feature store.
    fs_tag = {
        "name": "best_model",
        "version": best_model_artifact.version,
        "type": best_model_artifact.type,
        "url": f"https://wandb.ai/{SETTINGS['WANDB_ENTITY']}/{SETTINGS['WANDB_PROJECT']}/artifacts/{best_model_artifact.type}/{best_model_artifact._name}/{best_model_artifact.version}/overview",
        "artifact_name": f"{SETTINGS['WANDB_ENTITY']}/{SETTINGS['WANDB_PROJECT']}/{best_model_artifact.name}",
    }
    feature_view.add_tag(name="wandb", value=fs_tag)
    feature_view.add_training_dataset_tag(
        training_dataset_version=training_dataset_version, name="wandb", value=fs_tag
    )

    # Upload the model to the Hopsworks model registry.
    best_model_dir = best_model_artifact.download()
    best_model_path = Path(best_model_dir) / "best_model.pkl"
    best_model_metrics = best_model_artifact.metadata["results"]["test"]

    mr = project.get_model_registry()
    py_model = mr.python.create_model("best_model", metrics=best_model_metrics)
    py_model.save(best_model_path)

    return py_model.version
```

Now with a few lines of code, you can download and run inference on your model without carrying any more about all the complicated steps we showed you in this lesson.

Check out Lesson 3 to see how we will build a batch prediction pipeline using the model from the Hopsworks model registry.

## Conclusion

Congratulations! You finished the **second lesson** from the **Full Stack 7-Steps MLOps Framework course**.

If you have reached this far, you know how to:

- use an ML platform for experiment & metadata tracking
- use an ML platform for hyperparameter tuning
- read data from the feature store based on a given version
- build an encapsulated ML model and pipeline
- upload your model to a model registry

Now that you understand the power of using an ML platform, you can finally take control over your experiments and quickly export your model as an artifact to be easily used in your inference pipelines.

Check out Lesson 3 to learn about implementing a batch prediction pipeline and packaging your Python modules using Poetry.





