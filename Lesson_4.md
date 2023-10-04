This tutorial represents **lesson 4 out of a 7-lesson course** that will walk you step-by-step through how to **design, implement, and deploy an ML system** using **MLOps good practices**. During the course, you will build a production-ready model to forecast energy consumption levels for the next 24 hours across multiple consumer types from Denmark.

By the end of this course, you will understand all the fundamentals of designing, coding and deploying an ML system using a batch-serving architecture.

## Table of Contents:

- Course Introduction
- Course Lessons
- Data Source
- Lesson 4: Private PyPi Server. Orchestrate Everything with Airflow.
- Lesson 4: Code
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

By the end of **Lesson 4**, you will know how to host your PyPi repository and orchestrate the three pipelines using Airflow. You will learn how to schedule the pipelines to create hourly forecasts..

**Course Lessons:**

1. [Batch Serving. Feature Stores. Feature Engineering Pipelines.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
2. [Training Pipelines. ML Platforms. Hyperparameter Tuning.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_2.md)
3. [Batch Prediction Pipeline. Package Python Modules with Poetry.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_3.md)
4. **Private PyPi Server. Orchestrate Everything with Airflow.**
5. [Data Validation for Quality and Integrity using GE. Model Performance Continuous Monitoring.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_5.md)
6. [consume and Visualize your Model’s Predictions using FastAPI and Streamlit. Dockerize Everything.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_6.md)
7. [Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_7.md)
8. [Bonus - Behind the Scenes of an ‘Imperfect’ ML Project — Lessons and Insights](https://github.com/Hg03/mlops-paul/blob/main/Bonus.md)

If you want to grasp this lesson fully, we recommend you check out our previous lesson, which talks about designing a training pipeline that uses a feature store and an ML platform:

## The goal of lesson 4

This lesson will teach you how to use Airflow to orchestrate the three pipelines you have implemented so far.

Also, to run the code inside Airflow, you will learn to host your PiPy repository and deploy the pipelines as 3 different Python modules. Later you will install your modules inside Airflow directly from your PiPy repository.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/561b43a0-7575-4be2-b816-89015f785875)

By orchestrating everything using Airflow, you will automate your entire process. Instead of running manually 10 different scripts, you will hit once a "Run" button to run the whole code.

Also, connecting all the steps together in a programmatic way is less prone to bugs.

**Why ?**

Because every script needs its configurations. For example, the batch prediction pipeline needs the feature view version (data version) and the model version as input.

This information is generated as metadata from previous scripts. When you run everything manually, you can easily copy the wrong version. But when you wrap up everything inside a single DAG, you have to build it once, and afterward, it will work all the time.

Also, by using Airflow, you can:

- schedule the pipeline to run periodically (you will run it hourly);
- configure your entire process using Airflow Variables;
- monitor the logs of every task.

Here is an overview of what you will build in Airflow 👇

[Video](https://youtu.be/c6WI4yJEyoA)

## Theoretical Concepts & Tools

**Airflow:** Airflow is one of the most popular orchestration tools out there. The project was developed at Airbnb but is now open source under the Apache License. That means that you can modify and host it yourself for free. Airflow lets you build, schedule and monitor DAGs.

**DAG (Directed Acyclic Graph):** A DAG is a graph with no loops, meaning the logic flow can go only one way.

**PyPi Registry:** A PiPy registry is a server where you can host various Python modules. When you run "pip install <your_package>", pip knows how to look at the official PyPi repository for your package and install it. Hosting your own PyPi registry will behave precisely the same, but you must configure pip to know how to access it. Only people with access to your PyPi server can install packages from it.

## Lesson 4: Code

[You can access the GitHub repository here](https://github.com/iusztinpaul/energy-forecasting).

**Note:** All the installation instructions are in the READMEs of the repository. Here you will jump straight to the code.

All the code within Lesson 4 is located under the [airflow](https://github.com/iusztinpaul/energy-forecasting/tree/main/airflow) folder.

The files under the airflow folder are structured as follows:

![image](https://github.com/Hg03/mlops-paul/assets/69637720/ffed5039-a7c3-4cc7-99e0-4a92202f70e6)

All the code is located under the [dags](https://github.com/iusztinpaul/energy-forecasting/tree/main/airflow/dags) directory. Every DAG will have its own Python file.

The Docker files will help you quickly host Airflow and the PiPy repository. I will explain them in detail later.

Directly storing credentials in your git repository is a huge security risk. That is why you will inject sensitive information using a **.env**file.

The **.env.default** is an example of all the variables you must configure. It is also helpful to store default values for attributes that are not sensitive (e.g., project name).

![image](https://github.com/Hg03/mlops-paul/assets/69637720/3a8aae5c-406f-4429-bf06-b0a895f9e444)

## Prepare Credentials

As Lesson 4 talks about orchestrating the code from all the other lessons, if you want to reproduce the code, you need to check how to set up the 3 pipelines from Lesson 1, Lesson 2, and Lesson 3.

These three lessons will show you how to set up all the necessary tools and services. Also, it will show you how to create and complete the required .env file that contains all the credentials.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/c7b9f756-8d63-48b9-8219-869e713af83a)

The only thing to be careful of is 👇

This time you have to place the **.env** that contains your credentials under the [airflow/dags](https://github.com/iusztinpaul/energy-forecasting/tree/main/airflow/dags) folder.

We have set up a default value of **/opt/airflow/dags** for the **ML_PIPELINE_ROOT_DIR** environment variable inside the docker-compose.yaml file. Thus, when running the pipelines inside Airflow, it will know to load the **.env** file from **/opt/airflow/dags** by default.

Also, note that there is another .env file under the /airflow folder. This one doesn't contain your custom credentials, but Airflow needs some custom configurations. This is what it looks like 👇

![image](https://github.com/Hg03/mlops-paul/assets/69637720/862378ab-5391-4ae5-8b54-9f39b62474dc)

I explained how to complete this **.env** file in the [README.md](https://github.com/iusztinpaul/energy-forecasting/tree/main#usage) of the repository. But as a side note, **AIRFLOW_UID** represents your computer's USER ID, and you know what **ML_PIPELINE_ROOT_DIR** is.

I just wanted to show you that you can override the default value for **ML_PIPELINE_ROOT_DIR** here. Note that this path will be used inside the Docker container, hence the path that starts with **/opt/**.

```
# Move to the airflow directory.
cd airflow

# Make expected directories and environment variables
mkdir -p ./logs ./plugins
sudo chmod 777 ./logs ./plugins

# It will be used by Airflow to identify your user.
echo -e "AIRFLOW_UID=$(id -u)" > .env
# This shows where the project root directory is located.
echo "ML_PIPELINE_ROOT_DIR=/opt/airflow/dags" >> .env
```

## Setup Private PyPi Server

You can easily host a PiPy server using this repository. But let me explain how we did it in our setup.

The first step is to create a set of credentials that you will need to connect to the PyPi server.

```
# Install dependencies.
sudo apt install -y apache2-utils
pip install passlib

# Create the credentials under the energy-forecasting name.
mkdir ~/.htpasswd
htpasswd -sc ~/.htpasswd/htpasswd.txt energy-forecasting
```

The PyPi repository will know to load the credentials from the **~/.htpasswd/htpasswd.txt** file.

Now, you will add the new private PyPi repository to Poetry. To configure Poetry, you need to specify the URL of the server, the name of the server and the username & password to use to authenticate (which are the ones you configured one step before):

```
poetry config repositories.my-pypi http://localhost
poetry config http-basic.my-pypi energy-forecasting <password>
```

In our example:

- **name of the server:** my-pypy
- **URL:** http://localhost
- **username:** energy-forecasting
- **password:** <password>

Check if your credentials are set correctly in your Poetry **auth.toml** file:

```
cat ~/.config/pypoetry/auth.toml
```

So, you finished preparing the username and password that will be loaded by your PyPi repository to authenticate. Also, you configured Poetry to be aware of your PyPi server.

Now, let's see how to run the PyPi server.

The [pyserver code](https://github.com/pypiserver/pypiserver) you will be using is already dockerized.

To simplify things, we added the PyPi server as an additional service to the docker-compose.yaml file that runs the Airflow application.

To better understand the docker-compose.yaml file check [Airflow's official documentation](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html) and our [README.md](https://github.com/iusztinpaul/energy-forecasting/tree/main#the-pipeline). But be careful to use the docker-compose.yaml file from our repository as we modified the original one, as you will see below.

Scroll at the bottom of the [airflow/docker-compose.yaml](https://github.com/iusztinpaul/energy-forecasting/blob/main/airflow/docker-compose.yaml) file, and you will see:

```
  my-private-pypi:
    image: pypiserver/pypiserver:latest
    restart: always
    ports:
      - "80:8080"
    volumes:
      - ~/.htpasswd:/data/.htpasswd
    command:
      - run
      - -P
      - .htpasswd/htpasswd.txt
      - --overwrite
```

This code uses the PyPi server's latest image, exposes the server under the **80 port**, loads the **~/.htpasswd** folder that contains your credentials as a volume and runs the server with the following command:

`run -P .htpasswd/htpasswd.txt --overwrite`

* "-P .htpasswd/htpasswd.txt" explicitly tells the server what credentials to use.
* "— overwrite" states that if a new module with the same version is deployed, it will overwrite the last one.

Thats it! When you run the Airflow application, you automatically start the PyPi server.
**Note:** In a production environment, you will likely host the PyPi server on a different server than Airflow. The steps are identical except for adding everything in a single docker-compose.yaml file. In this tutorial, we wanted to make everything easy to run.

## Customize Airflow Docker File

Because you have to run all the code in Python 3.9, you have to inherit the default apache/airflow:2.5.2 Airflow Docker image and add some extra dependencies.

This is what is going on in the Docker file below:

- inherit apache/airflow:2.5.2
- switch to the root user to install system dependencies
- install Python 3.9 dependencies needed to install packages from the private PyPi server
- switch back to the default user

```
FROM apache/airflow:2.5.2

ARG CURRENT_USER=$USER

USER root
# Install Python dependencies to be able to process the wheels from the private PyPI server.
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y python3.9-distutils python3.9-dev build-essential
USER ${CURRENT_USER}
```

Because we switched:

```
x-airflow-common:
  &airflow-common
  image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:2.5.2}
```

To:

```
version: '3.8'
x-airflow-common:
  &airflow-common
#  image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:2.5.2}
  build: .
```

Docker will know to use your custom image instead of **apache/airflow:2.5.2 when running docker-compose**.

## Run Airflow

Now that you understand how to prepare the credentials and how the Docker files work, go to the [./airflow directory](https://github.com/iusztinpaul/energy-forecasting/tree/main/airflow) and run:

```
# Go to the ./airflow directory.
cd ./airflow

# Initialize the Airflow database
docker compose up airflow-init

# Start up all services
# Note: You should set up the private PyPi server credentials before running this command.
docker compose --env-file .env up --build -d
```

[Check out the Usage section of the GitHub repository for more info](https://github.com/iusztinpaul/energy-forecasting/tree/main#-usage-).

After you finish your Airflow setup, you can access Airflow at **127.0.0.1:8080** with the default credentials:

- username: airflow
- password: airflow

![image](https://github.com/Hg03/mlops-paul/assets/69637720/07f90ceb-b4eb-48f5-a0d2-f3a5bdca9455)

## Deploy Modules to Private PyPi Server

Remember that you added to Poetry your new PyPi server using the following commands:

```
poetry config repositories.my-pypi http://localhost
poetry config http-basic.my-pypi energy-forecasting <password>
```

Now, using **my-pypi** as an identifier, you can quickly push new packages to your PyPi repository.

Using the [deploy/ml-pipeline.sh](https://github.com/iusztinpaul/energy-forecasting/blob/main/deploy/ml-pipeline.sh) shell script, you can build & deploy all the 3 pipelines using solely Poetry:

```
!/bin/bash

# Build and publish the feature-pipeline, training-pipeline, and batch-prediction-pipeline packages.
# This is done so that the pipelines can be run from the CLI.
# The pipelines are executed in the feature-pipeline, training-pipeline, and batch-prediction-pipeline
# directories, so we must change directories before building and publishing the packages.
# The my-pypi repository must be defined in the project's poetry.toml file.

cd feature-pipeline
poetry build
poetry publish -r my-pypi

cd ../training-pipeline
poetry build
poetry publish -r my-pypi

cd ../batch-prediction-pipeline
poetry build
poetry publish -r my-pypi
```

As you can see, we iteratively go to the folders of the 3 pipelines and run:

```
poetry build
poetry publish -r my-pypi
```

Poetry uses these two commands to look for the **pyproject.toml** and **poetry.lock** files inside the folders and knows how to build the package.

Afterward, based on the generated **wheel** file, running **"poetry publish -r my-pypi"**, you push it to your** my-pipy** repository.

Remember that you labeled your PyPi server as **my-pipy**.

You are done. You have your own PyPi repository.

In future sections, I will show you how to install packages from your private PyPi repository.

**Note:** You used Poetry just to build & deploy the modules. Airflow will use pip to install them from your PiPy repository.

## Define the DAG Object

Your dag is defined under the [airflow/dags/ml_pipeline_dag.py](https://github.com/iusztinpaul/energy-forecasting/blob/main/airflow/dags/ml_pipeline_dag.py) file.

Using the **API of Airflow 2.0**, you can define a DAG using the **dag()** Python decorator.

Your dag will be defined inside the **ml_pipeline()** function, which is called at the end of the file. Also, Airflow knows to load all the DAGs defined under the airflow/dags directory.

The DAG has the following properties:

- **dag_id:** the ID of the DAG
- **schedule:** it defines how often the DAG runs
- **start_date:** when should the DAG start running based on the given schedule
- **catchup:** automatically backfill between [start_date, present]
- **tags:** tags 😄
- **max_active_runs:** how many instances of this DAG can run in parallel

```python
@dag(
    dag_id="ml_pipeline",
    schedule="@hourly",
    start_date=datetime(2023, 4, 14),
    catchup=False,
    tags=["feature-engineering", "model-training", "batch-prediction"],
    max_active_runs=1,
)
def ml_pipeline():
  ...
 

ml_pipeline()

```

## Define the Tasks

The code below might look long, but you can easily read it once you understand the main ideas.

Inside a DAG, you have defined multiple tasks. A task is a single logic unit/step that performs a specific operation.

The tasks are defined similarly to the DAG: a function + a decorator. Every task has its function and decorator.

_**Note:** This a simple reminder that we used the API for Airflow 2.0, not 1.0._

In our case, a task will represent a main pipeline script. For example, the feature engineering pipeline will run inside a single task.

You will use a DAG to glue all your scripts under a single "program", where every script has a 1:1 representation with a task.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/d8eb5652-c20c-4bd1-b0b0-f01e83cf1d58)

As you can see inside every task, you just import and call the function from its own module… and maybe add some additional logs.

The key step in defining a task is in the arguments of the **task.virtualenv()** Python decorator.

For every task, this specific decorator will create a different Python virtual environment inside which it will install all the given requirements.

**Note:** **172.17.0.1** is the IP address of your private PyPi repository. Remember that you host your PyPi repository using docker-compose under the same network as Airflow. **172.17.0.1** is the bridge IP address accessible by every Docker container inside the **default Docker** network. Thus, the Airflow container can access the PyPi server container using the bridge IP address.

As you can see in the **requirements** argument, we defined the following:

- **"— trusted-host 172.17.0.1":** As the PyPi server is not secured with HTTPS, you must explicitly say that you trust this source.
- **"— extra-index-url http://172.17.0.1":** Tell Pip to also look at this PyPi repository when searching for a Python package. Note that Pip will still look in the official PyPi repository in addition to yours.
- **"<your_python_packages>":** After the two lines described above, you can add any Python package. But note that you installed **feature_pipeline**, **training_pipeline**, and **batch_prediction_pipeline** as Python packages you built and deployed using Poetry.

The other arguments aren't that interesting, but let me explain them:

- **task_id=" <task_id>":** The unique ID of a task.
- **python_version=" 3.9":** When I was writing this course, Hopsworks worked only with Python 3.9, so we had to enforce this version of Python.
- **multiple_outputs=True:** The task returns a Python dictionary.
- **system_site_packages=True:** Install default system packages.

**Important**

Note that almost every task returns a dictionary of metadata that contains information such as:

- the date range when the data was extracted,
- the version of the feature group, feature view, etc.
- the version of the sweep,
- the version of the model, etc.

This information is essential to be passed between tasks. For example, the **create_feature_view** task needs to know what version of the **feature_group** to use to create the next feature view. Also, when running **batch_predict**, you have to know the version of the feature view and model to use to generate the predictions.

```python
@dag(
    dag_id="ml_pipeline",
    schedule="@hourly",
    start_date=datetime(2023, 4, 14),
    catchup=False,
    tags=["feature-engineering", "model-training", "batch-prediction"],
    max_active_runs=1,
)
def ml_pipeline():
    @task.virtualenv(
        task_id="run_feature_pipeline",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "feature_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=True,
    )
    def run_feature_pipeline(
        export_end_reference_datetime: str,
        days_delay: int,
        days_export: int,
        url: str,
        feature_group_version: int,
    ) -> dict:
        """
        Run the feature pipeline.
        Args:
            export_end_reference_datetime: The end reference datetime of the export window. If None, the current time is used.
                Because the data is always delayed with "days_delay" days, this date is used only as a reference point.
                The real extracted window will be computed as [export_end_reference_datetime - days_delay - days_export, export_end_reference_datetime - days_delay].
            days_delay : int
                Data has a delay of N days. Thus, we have to shift our window with N days.
            days_export : int
                The number of days to export.
            url : str
                URL to the raw data.
            feature_group_version : int
                Version of the feature store feature group to use.
        Returns:
            Metadata of the feature pipeline run.
        """

        from datetime import datetime

        from feature_pipeline import utils, pipeline

        logger = utils.get_logger(__name__)

        try:
            export_end_reference_datetime = datetime.strptime(
                export_end_reference_datetime, "%Y-%m-%d %H:%M:%S.%f%z"
            )
        except ValueError:
            export_end_reference_datetime = datetime.strptime(
                export_end_reference_datetime, "%Y-%m-%d %H:%M:%S%z"
            )
        export_end_reference_datetime = export_end_reference_datetime.replace(
            microsecond=0, tzinfo=None
        )

        logger.info(f"export_end_datetime = {export_end_reference_datetime}")
        logger.info(f"days_delay = {days_delay}")
        logger.info(f"days_export = {days_export}")
        logger.info(f"url = {url}")
        logger.info(f"feature_group_version = {feature_group_version}")

        return pipeline.run(
            export_end_reference_datetime=export_end_reference_datetime,
            days_delay=days_delay,
            days_export=days_export,
            url=url,
            feature_group_version=feature_group_version,
        )

    @task.virtualenv(
        task_id="create_feature_view",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "feature_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def create_feature_view(feature_pipeline_metadata: dict) -> dict:
        """
        This function creates a feature view based on the feature pipeline computations. The feature view
        is created using the feature group version from the feature pipeline metadata.
        """

        from feature_pipeline import feature_view

        return feature_view.create(
            feature_group_version=feature_pipeline_metadata["feature_group_version"]
        )

    @task.virtualenv(
        task_id="run_hyperparameter_tuning",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "training_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False,
    )
    def run_hyperparameter_tuning(feature_view_metadata: dict) -> dict:
        """
        This function runs hyperparameter tuning for the training pipeline.
        The feature store feature view version and training dataset version are passed
        based on the results from the create_feature_view task.
        """

        from training_pipeline import hyperparameter_tuning

        return hyperparameter_tuning.run(
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"],
        )

    @task.virtualenv(
        task_id="upload_best_config",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "training_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=False,
        system_site_packages=False,
    )
    def upload_best_config(last_sweep_metadata: dict):
        """
        Upload the best config to W&B ML platform found in the hyperparameter tuning step
        based on the given sweep id.
        """

        from training_pipeline import best_config

        best_config.upload(sweep_id=last_sweep_metadata["sweep_id"])

    @task.virtualenv(
        task_id="train_from_best_config",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "training_pipeline",
        ],
        python_version="3.9",
        multiple_outputs=True,
        system_site_packages=False,
        trigger_rule=TriggerRule.ALL_DONE,
    )
    def train_from_best_config(feature_view_metadata: dict) -> dict:
        """Trains model from the best config found in hyperparameter tuning.
        Args:
            feature_view_metadata (dict): Contains feature store feature view and training dataset version.
        Returns:
            metadata from the training run
        """

        from training_pipeline import utils, train

        has_best_config = utils.check_if_artifact_exists("best_config")
        if has_best_config is False:
            raise RuntimeError(
                "No best config found. Please run hyperparameter tuning first."
            )

        return train.from_best_config(
            feature_view_version=feature_view_metadata["feature_view_version"],
            training_dataset_version=feature_view_metadata["training_dataset_version"],
        )

    @task.virtualenv(
        task_id="compute_monitoring",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "batch_prediction_pipeline",
        ],
        python_version="3.9",
        system_site_packages=False,
    )
    def compute_monitoring(feature_view_metadata: dict):
        """Compute monitoring metrics for newly obbserved data.
        Args:
            feature_view_metadata: metadata containing the version of the feature store feature view version.
        """

        from batch_prediction_pipeline import monitoring

        monitoring.compute(
            feature_view_version=feature_view_metadata["feature_view_version"],
        )

    @task.virtualenv(
        task_id="batch_predict",
        requirements=[
            "--trusted-host 172.17.0.1",
            "--extra-index-url http://172.17.0.1",
            "batch_prediction_pipeline",
        ],
        python_version="3.9",
        system_site_packages=False,
    )
    def batch_predict(
        feature_view_metadata: dict,
        train_metadata: dict,
        feature_pipeline_metadata: dict,
        fh: int = 24,
    ):
        """
        This is the function that runs the batch prediction pipeline
        Args:
            feature_view_metadata (dict):  the metadata from the create feature view task
            train_metadata (dict): the metadata from the training pipeline task
            feature_pipeline_metadata (dict): the metadata from the feature pipeline task
            fh (int, optional): forecast horizon. Defaults to 24.
        """

        from datetime import datetime
        from batch_prediction_pipeline import batch

        start_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_start"],
            feature_pipeline_metadata["datetime_format"],
        )
        end_datetime = datetime.strptime(
            feature_pipeline_metadata["export_datetime_utc_end"],
            feature_pipeline_metadata["datetime_format"],
        )

        batch.predict(
            fh=fh,
            feature_view_version=feature_view_metadata["feature_view_version"],
            model_version=train_metadata["model_version"],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
```

One interesting task is **task.branch(task_id=" if_run_hyperparameter_tuning_branching")**, which defines an if-else logic between whether to run the hyperparameter tuning logic or not.

This special type of task returns a list of **task_ids** that will be executed next. For example, if it returns **["branch_run_hyperparameter_tuning"]**, it will run only the task with the **task_id = branch_run_hyperparameter_tuning**.

As you can see below, two empty operators (tasks) are defined with the task_ids used inside the **task.branch()** logic. It is a common pattern suggested by Airflow to use a set of empty operators (no operation) when choosing between multiple branches.

```python
@dag(
    dag_id="ml_pipeline",
    schedule="@hourly",
    start_date=datetime(2023, 4, 14),
    catchup=False,
    tags=["feature-engineering", "model-training", "batch-prediction"],
    max_active_runs=1,
)
def ml_pipeline():
    
  ...
  
  @task.branch(task_id="if_run_hyperparameter_tuning_branching")
      def if_run_hyperparameter_tuning_branching(run_hyperparameter_tuning: bool) -> bool:
          """Task used to branch between hyperparameter tuning and skipping it."""
          if run_hyperparameter_tuning is True:
              return ["branch_run_hyperparameter_tuning"]
          else:
              return ["branch_skip_hyperparameter_tuning"]

      # Define empty operators used for branching between hyperparameter tuning and skipping it.
      branch_run_hyperparameter_tuning_operator = EmptyOperator(
          task_id="branch_run_hyperparameter_tuning"
      )
      branch_skip_hyperparameter_tuning_operator = EmptyOperator(
          task_id="branch_skip_hyperparameter_tuning"
      )
```

## Connect the Tasks into a DAG

Now that you defined all the tasks, the final step is to connect them into a DAG. You have to perform this step so Airflow knows in what order to run every task.

Basically, here you will define the logic graph.

**#1.** The first step is to _**determine the set of variables**_ that you will use to configure the DAG, such as **days_delay, days_export, feature_group_version, etc.** You can access these variables from the “Admin -> Variables” panel of Airflow.

Note that you have to add them using the blue plus button explicitly.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/d8b96931-7dbe-4437-890d-b076ae255034)

**#2.** The second step is to **call the tasks with the right parameters**. As you can see, because of the Airflow 2.0 API, this step is just like calling a bunch of Python functions in a specific order.

**Note:** A dependency in the graph is automatically created if the output of a function is added as an input to another function.

> It is essential to highlight how we passed the metadata of every pipeline element to the next ones. In doing so, we enforce the following scripts to use the correct data and model versions.

I also want to emphasize the following piece of code:

```python
feature_pipeline_metadata = run_feature_pipeline(
        export_end_reference_datetime="{{ dag_run.logical_date }}",
    )
```

**“{{ dag_run.logical_date }}"** is a template variable injected by Airflow that reflects the logical date when the DAG is run. Not the current date. By doing so, using the Airflow backfill features, you can easily use this as a datetime reference to backfill in a given time window. Now you can easily manipulate your extraction window's starting and ending points.

For example, if you want to run the DAG to backfill the energy consumption predictions between 10 and 11 May 2023, you will run the Airflow backfill logic with the "10 May 2023 00:00 am date".

**#3.** The last step is to enforce a specific _**DAG structure**_ using the ">>" operator.

**"A >> B >> C"** means run A, then B, then C.

The only trickier piece of code is this one:

```
>> if_run_hyperparameter_tuning_branch
>> [
    if_run_hyperparameter_tuning_branch
    >> Label("Run HPO")
    >> branch_run_hyperparameter_tuning_operator
    >> last_sweep_metadata
    >> upload_best_model_step,
    if_run_hyperparameter_tuning_branch
    >> Label("Skip HPO")
    >> branch_skip_hyperparameter_tuning_operator,
]
```

,where based on the **branch** operator, the DAG will either run the **branch_run_hyperparameter_tuning_operator** or **branch_skip_hyperparameter_tuning_operator** branches of the DAG.

Read more about branching in Airflow [here](https://docs.astronomer.io/learn/airflow-branch-operator).

In English, it will run hyper optimization tunning or skip it, as shown in the image below — I know the image is quite small. Check the video for a better view. 👇

![image](https://github.com/Hg03/mlops-paul/assets/69637720/6106860b-9e42-4e2f-98a0-5e9a9cc311e7)

That is it! You orchestrated all the 3 pipelines using Airflow. Congrats!

```python
@dag(
    dag_id="ml_pipeline",
    schedule="@hourly",
    start_date=datetime(2023, 4, 14),
    catchup=False,
    tags=["feature-engineering", "model-training", "batch-prediction"],
    max_active_runs=1,
)
def ml_pipeline():
  
  # Define tasks.
  ...
  
  # Define Airflow variables.
  days_delay = int(Variable.get("ml_pipeline_days_delay", default_var=15))
  days_export = int(Variable.get("ml_pipeline_days_export", default_var=30))
  url = Variable.get(
      "ml_pipeline_url",
      default_var="https://api.energidataservice.dk/dataset/ConsumptionDE35Hour",
  )
  feature_group_version = int(
      Variable.get("ml_pipeline_feature_group_version", default_var=1)
  )
  should_run_hyperparameter_tuning = (
      Variable.get(
          "ml_pipeline_should_run_hyperparameter_tuning", default_var="False"
      )
      == "True"
  )

  # Feature pipeline
  feature_pipeline_metadata = run_feature_pipeline(
      export_end_reference_datetime="{{ dag_run.logical_date }}",
      days_delay=days_delay,
      days_export=days_export,
      url=url,
      feature_group_version=feature_group_version,
  )
  feature_view_metadata = create_feature_view(feature_pipeline_metadata)

  # Training pipeline
  if_run_hyperparameter_tuning_branch = if_run_hyperparameter_tuning_branching(
      should_run_hyperparameter_tuning
  )
  last_sweep_metadata = run_hyperparameter_tuning(feature_view_metadata)
  upload_best_model_step = upload_best_config(last_sweep_metadata)
  train_metadata = train_from_best_config(feature_view_metadata)

  # Batch prediction pipeline
  compute_monitoring_step = compute_monitoring(feature_view_metadata)
  batch_predict_step = batch_predict(
      feature_view_metadata, train_metadata, feature_pipeline_metadata
  )

  # Define DAG structure.
  (
      feature_view_metadata
      >> if_run_hyperparameter_tuning_branch
      >> [
          if_run_hyperparameter_tuning_branch
          >> Label("Run HPO")
          >> branch_run_hyperparameter_tuning_operator
          >> last_sweep_metadata
          >> upload_best_model_step,
          if_run_hyperparameter_tuning_branch
          >> Label("Skip HPO")
          >> branch_skip_hyperparameter_tuning_operator,
      ]
      >> train_metadata
      >> compute_monitoring_step
      >> batch_predict_step
  )

```

## Run the ML Pipeline DAG

This step is easy.

Just go to your ml_pipeline DAG and click the play button.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/3495fbaa-f844-46d3-a864-e09048826896)

## Backfill using Airflow

Find your airflow-webserver docker container ID:

`docker ps`

Start a shell inside the airflow-webserver container and run airflow dags backfill as follows:

```
docker exec -it <container-id-of-airflow-webserver> sh
# In this example, you did a backfill between 2023/04/11 00:00:00 and 2023/04/13 23:59:59.
airflow dags backfill --start-date "2023/04/11 00:00:00" --end-date "2023/04/13 23:59:59" ml_pipeline
```

If you want to clear the tasks and rerun them, run these commands:

```
docker exec -it <container-id-of-airflow-airflow-webserver> sh
airflow tasks clear --start-date "2023/04/11 00:00:00" --end-date "2023/04/13 23:59:59" ml_pipeline
```

## Conclusion

Congratulations! You finished the fourth lesson from the Full Stack 7-Steps MLOps Framework course.

If you have reached this far, you know how to:

- Host your own PyPi server
- Build & deploy your Python modules using Poetry
- Orchestrate multiple pipelines using Airflow

Now that you understand the power of using an orchestrator such as Airflow, you can build robust production-ready pipelines that you can quickly schedule, configure, and monitor.

Check out **Lesson 5** to learn how to use Great Expectations to validate the integrity and quality of your data. Also, you will understand how to implement a monitoring component on top of your ML system.

