# FastAPI and Streamlit: The Python Duo You Must Know About

This tutorial represents **lesson 6 out of a 7-lesson course** that will walk you step-by-step through how to **design, implement, and deploy an ML system** using **MLOps good practices**. During the course, you will build a production-ready model to forecast energy consumption levels for the next 24 hours across multiple consumer types from Denmark.

_By the end of this course, you will understand all the fundamentals of designing, coding and deploying an ML system using a batch-serving architecture._

This course targets mid/advanced machine learning engineers who want to level up their skills by building their own end-to-end projects.

## Table of Contents:

- Course Introduction
- Course Lessons
- Data Source
- Lesson 6: Consume and Visualize your Model’s Predictions using FastAPI and Streamlit. Dockerize Everything.
- Lesson 6: Code
- Conclusion
- References

## Course Introduction

**At the end of this 7 lessons course, you will know how to:**__

- design a batch-serving architecture
- use Hopsworks as a feature store
- design a feature engineering pipeline that reads data from an API
- build a training pipeline with hyper-parameter tunning
- use W&B as an ML Platform to track your experiments, models, and metadata
- implement a batch prediction pipeline
- use Poetry to build your own Python packages
- deploy your own private PyPi server
- orchestrate everything with Airflow
- use the predictions to code a web app using FastAPI and Streamlit
- use Docker to containerize your code
- use Great Expectations to ensure data validation and integrity
- monitor the performance of the predictions over time
- deploy everything to GCP
- build a CI/CD pipeline using GitHub Actions

If that sounds like a lot, don't worry. After you cover this course, you will understand everything I said before. Most importantly, you will know WHY I used all these tools and how they work together as a system.

By the end of the course, you will know how to implement the diagram below. Don't worry if something doesn't make sense to you. I will explain everything in detail.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/77cc228d-c501-452a-8c4d-3c4e13e0b300)

By the end of Lesson 6, you will know how to consume the predictions and the monitoring metrics from the GCP bucket within a web app using FastAPI and Streamlit.

## Course Lessons:

1. [Batch Serving. Feature Stores. Feature Engineering Pipelines.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
2. [Training Pipelines. ML Platforms. Hyperparameter Tuning.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_2.md)
3. [Batch Prediction Pipeline. Package Python Modules with Poetry.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_3.md)
4. [Private PyPi Server. Orchestrate Everything with Airflow.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_4.md)
5. [Data Validation for Quality and Integrity using GE. Model Performance Continuous Monitoring.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_5.md)
6. **Consume and Visualize your Model’s Predictions using FastAPI and Streamlit. Dockerize Everything.**
7. [Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_7.md)
8. [Bonus - Behind the Scenes of an ‘Imperfect’ ML Project — Lessons and Insights](https://github.com/Hg03/mlops-paul/blob/main/Bonus.md)

Check out **Lesson 3** to learn how we computed and stored the predictions in a GCP bucket.

Also, in **Lesson 5**, you can see how we calculated the monitoring metrics, which are also stored in a GCP bucket.

You will consume the predictions and the monitoring metrics from the GCP bucket and display them in a friendly dashboard using FastAPI and Streamlit.

## The goal of Lesson 6

In Lesson 6, you will build a FastAPI backend that will consume the predictions and monitoring metrics from GCS and expose them through a RESTful API. More concretely, through a set of endpoints that will expose the data through HTTP(S).

Also, you will implement 2 different frontend applications using solely Streamlit:

1. a dashboard showing the forecasts (aka your application),
2. a dashboard showing the monitoring metrics (aka your monitoring dashboard).

Both frontend applications will request data from the FastAPI RESTful API through HTTP(s) and use Streamlit to render the data into some beautiful plots.

I want to highlight that you can use both frameworks (FastAPI & Streamlit) in Python. This is extremely useful for a DS or MLE, as Python is their holy grail.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/65a8d58e-65af-4c6a-9cc2-d174bd5c80c1)

**Note** that consuming the predictions from the bucket is completely decoupled from the 3 pipeline design. For example, running the 3 pipelines: feature engineer, training, and inference takes ~10 minutes. But to read the predictions or the monitor metrics from the bucket is almost instant.

Thus, by caching the predictions into GCP, you served the ML model online from the client's point of view: the predictions are served in real time.

_This is the magic of the batch architecture_

The next natural steps are to move your architecture from a batch architecture to a request-response or streaming one.

The good news is that the FE and training pipelines would be almost the same, and you would have to move only the batch prediction pipeline (aka the inference step) into your web infrastructure. [Read this article to learn the basics of deploying your model in a request-response fashion using Docker](https://medium.com/faun/key-concepts-for-model-serving-38ccbb2de372).

**Why?**

Because the training pipeline uploads the weights of the trained model into the model registry. From there, you can use the weights as it fits best for your use case.

## Theoretical Concepts & Tools

**FastAPI**: One of the latest and most famous Python API web frameworks. I have tried all of the top Python API web frameworks: Django, Flask, and FastAPI, and my heart goes to FastAPI.

Why?

First, it is natively async, which can boost performance with fewer computing resources.

Secondly, it is easy and intuitive to use, which makes it suitable for applications of all sizes. Even though, for behemoth monoliths, I would still choose Django. But this is a topic for another time.

**Streamlit**: Streamlit makes coding simple UI components, mostly dashboards, extremely accessible using solely Python.

The scope of Streamlit is to let Data Scientists and ML engineers use what they know best, aka Python, to build a beautiful frontend for their models quickly.

**Which is precisely what we did ✌️**

Thus, you will use FastAPI as your backend and Streamlit as your frontend to build a web app solely in Python.

## Lesson 6: Code

**Note**: All the installation instructions are in the READMEs of the repository. Here you will jump straight to the code.

The code within Lesson 6 is located under the following:

- `app-api` folder — FastAPI backend
- `app-frontend` folder — Predictions Dashboard
- `app-monitoring` folder — Monitoring Dashboard

Using Docker, you can quickly spin up all 3 components at once:

`docker compose -f deploy/app-docker-compose.yml --project-directory . up --build`

Directly storing credentials in your git repository is a huge security risk. That is why you will inject sensitive information using a **.env** file.

The **.env.default** is an example of all the variables you must configure. It is also helpful to store default values for attributes that are not sensitive (e.g., project name).

![image](https://github.com/Hg03/mlops-paul/assets/69637720/f3109d32-8d38-462c-8aff-6d98df491aba)

## Prepare credentials

For this lesson, the only service you need access to is GCS. In the **Prepare Credentials section** of Lesson 3, we already explained in detail how to do this. Also, you have more information in the **GitHub README**.

To keep things concise, in this lesson, I want to highlight that the web app GCP service account should have read access only for security reasons.

Why?

Because the FastAPI API will only read data from the GCP buckets & keeping the permissions to the bare minimum is good practice.

Thus, if your web app is hacked, the attacker can only read the data using the stolen service account credentials. He can't delete or overwrite the data, which is much more dangerous in this case.

Thus, repeat the same steps as in the **Prepare Credentials** section of **Lesson 3**, but instead of choosing the _Store Object Admin role_, choose the _Storage Object Viewer role_.

Remember that now you have to download a different JSON file containing your GCP service account key with read-only access.

Check out the README to learn how to complete the **.env** file. I want to highlight that only the FastAPI backend will have to load the **.env** file. Thus, you must place the **.env** file only in the **app-api** folder.


## FastAPI Backend

[Video](https://youtu.be/dWm9oR_aMyo)

As a reminder, the FastAPI code can be found under **app-api/api**.

**Step 1:** Create the FastAPI application, where we configured the docs, the CORS middleware and the endpoints root API router.

```python
from typing import Any

import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.views import api_router
from api.config import get_settings


def get_app() -> FastAPI:
    """Create FastAPI app."""
    
    app = FastAPI(
        title=get_settings().PROJECT_NAME,
        docs_url=f"/api/{get_settings().VERSION}/docs",
        redoc_url=f"/api/{get_settings().VERSION}/redoc",
        openapi_url=f"/api/{get_settings().VERSION}/openapi.json",
    )
    # For demo purposes, allow all origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=f"/api/{get_settings().VERSION}")

    return app
```

**Step 2:** Define the Settings class. The scope of this class is to hold all the constants and configurations you need across your API code, such as:

- **generic configurations:** the port, log level or version,
- **GCP credentials:** bucket name or path to the JSON service account keys.

You will use the Settings object across the project using the **get_settings()** function.

```python
import enum
from functools import lru_cache
import logging
import sys
from types import FrameType
from typing import List, Optional, cast

from pydantic import AnyHttpUrl, BaseSettings


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    # General configurations.
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: LogLevel = LogLevel.INFO
    # - Current version of the API.
    VERSION: str = "v1"
    # - Quantity of workers for uvicorn.
    WORKERS_COUNT: int = 1
    # - Enable uvicorn reloading.
    RELOAD: bool = False

    PROJECT_NAME: str = "Energy Consumption API"

    # Google Cloud Platform credentials
    GCP_PROJECT: Optional[str] = None
    GCP_BUCKET: Optional[str] = None
    GCP_SERVICE_ACCOUNT_JSON_PATH: Optional[str] = None

    class Config:
        env_file = ".env"
        env_prefix = "APP_API_"
        case_sensitive = False
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings():
    return Settings()
```

Also, inside the **Config** class, we programmed FastAPI to look for a **.env** file in the current directory and load all the variables prefixed with **APP_API_**.

As you can see in the **.env.default** file, all the variables start with **APP_API_**.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/b0f06cec-2545-4734-ae0e-d71218c8cb50)

**Step 3:** Define the schemas of the API data using Pydantic. These schemas encode or decode data from JSON to a Python object or vice versa. Also, they validate the type and structure of your JSON object based on your defined data model.

When defining a Pydantic BaseModel, it is essential to add a type to every variable, which will be used at the validation step.

```python
from typing import Any, List

from pydantic import BaseModel


class UniqueArea(BaseModel):
    values: List[int]
     

class UniqueConsumerType(BaseModel):
    values: List[int]
      
class PredictionResults(BaseModel):
    preds_energy_consumption: List[float]


class MonitoringMetrics(BaseModel):
    datetime_utc: List[int]
    mape: List[float]


class MonitoringValues(BaseModel):
    y_monitoring_datetime_utc: List[int]
    y_monitoring_energy_consumption: List[float]
    predictions_monitoring_datetime_utc: List[int]
    predictions_monitoring_energy_consumptionc: List[float]

```
**Step 4:** Define your endpoints, in web lingo, known as views. Usually, a view has access to some data storage and based on a query, it returns a subset of the data source to the requester.

**Thus, a standard flow for retrieving (aka GET request) data looks like this:**

“client → request data → endpoint → access data storage → encode to a Pydantic schema → decode to JSON → respond with requested data”

Let's see how we defined an endpoint to GET all the consumer types:

```python
import gcsfs
from typing import Any, List

import pandas as pd
from fastapi import APIRouter, HTTPException

from api import schemas
from api.config import get_settings


fs = gcsfs.GCSFileSystem(
    project=get_settings().GCP_PROJECT,
    token=get_settings().GCP_SERVICE_ACCOUNT_JSON_PATH,
)

api_router = APIRouter()


@api_router.get(
    "/consumer_type_values", response_model=schemas.UniqueConsumerType, status_code=200
)
def consumer_type_values() -> List:
    """
    Retrieve unique consumer types.
    """

    # Download the data from GCS.
    X = pd.read_parquet(f"{get_settings().GCP_BUCKET}/X.parquet", filesystem=fs)

    unique_consumer_type = list(X.index.unique(level="consumer_type"))

    return {"values": unique_consumer_type}
```

We used **"gcsfs.GCSFileSystem"** to access the GCS bucket as a standard filesystem.

We attached the endpoint to the **api_router**.

Using the **api_router.get()** Python decorator, we attached a basic function to the **/consumer_type_values** endpoint.

In the example above, when calling **"https://<some_ip>:8001/api/v1/consumer_type_values"** the **consumer_type_values()** function will be triggered, and the response of the endpoint will be strictly based on what the function return.

Another important thing is to highlight that by defining the **response_model (aka the schema) in the Python decorator**, you don't have to create the Pydantic schema explicitly.

If you return a dictionary that is 1:1, respecting the schema structure, FastAPI will automatically create the Pydantic object for you.

That's it. Now we will repeat the same logic to define the rest of the endpoints. FastAPI makes everything so easy and intuitive for you.

Now, let's take a look at the whole **views.py** file, where we defined endpoints for the following:

- **/health** → health check
- **/consumer_type_values** → GET all possible consumer types
- **/area_values** → GET all possible area types
- **/predictions/{area}/{consumer_type}** → GET the predictions for a given area and consumer type. Note that using the {<some_variable>} syntax, you can add parameters to your endpoint — [FastAPI docs](https://fastapi.tiangolo.com/tutorial/path-params/).
- **/monitoring/metrics** → GET the aggregated monitoring metrics
- **/monitoring/values/{area}/{consumer_type}** → GET the monitoring values for a given area and consumer type

```python
import gcsfs
from typing import Any, List

import pandas as pd
from fastapi import APIRouter, HTTPException

from api import schemas
from api.config import get_settings


fs = gcsfs.GCSFileSystem(
    project=get_settings().GCP_PROJECT,
    token=get_settings().GCP_SERVICE_ACCOUNT_JSON_PATH,
)

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Health check endpoint.
    """

    health_data = schemas.Health(
        name=get_settings().PROJECT_NAME, api_version=get_settings().VERSION
    )

    return health_data.dict()


@api_router.get(
    "/consumer_type_values", response_model=schemas.UniqueConsumerType, status_code=200
)
def consumer_type_values() -> List:
    """
    Retrieve unique consumer types.
    """

    # Download the data from GCS.
    X = pd.read_parquet(f"{get_settings().GCP_BUCKET}/X.parquet", filesystem=fs)

    unique_consumer_type = list(X.index.unique(level="consumer_type"))

    return {"values": unique_consumer_type}


@api_router.get("/area_values", response_model=schemas.UniqueArea, status_code=200)
def area_values() -> List:
    """
    Retrieve unique areas.
    """

    # Download the data from GCS.
    X = pd.read_parquet(f"{get_settings().GCP_BUCKET}/X.parquet", filesystem=fs)

    unique_area = list(X.index.unique(level="area"))

    return {"values": unique_area}


@api_router.get(
    "/predictions/{area}/{consumer_type}",
    response_model=schemas.PredictionResults,
    status_code=200,
)
async def get_predictions(area: int, consumer_type: int) -> Any:
    """
    Get forecasted predictions based on the given area and consumer type.
    """

    # Download the data from GCS.
    train_df = pd.read_parquet(f"{get_settings().GCP_BUCKET}/y.parquet", filesystem=fs)
    preds_df = pd.read_parquet(
        f"{get_settings().GCP_BUCKET}/predictions.parquet", filesystem=fs
    )

    # Query the data for the given area and consumer type.
    try:
        train_df = train_df.xs((area, consumer_type), level=["area", "consumer_type"])
        preds_df = preds_df.xs((area, consumer_type), level=["area", "consumer_type"])
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for the given area and consumer type: {area}, {consumer_type}",
        )

    if len(train_df) == 0 or len(preds_df) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for the given area and consumer type: {area}, {consumer_type}",
        )

    # Return only the latest week of observations.
    train_df = train_df.sort_index().tail(24 * 7)

    # Prepare data to be returned.
    datetime_utc = train_df.index.get_level_values("datetime_utc").to_list()
    energy_consumption = train_df["energy_consumption"].to_list()

    preds_datetime_utc = preds_df.index.get_level_values("datetime_utc").to_list()
    preds_energy_consumption = preds_df["energy_consumption"].to_list()

    results = {
        "datetime_utc": datetime_utc,
        "energy_consumption": energy_consumption,
        "preds_datetime_utc": preds_datetime_utc,
        "preds_energy_consumption": preds_energy_consumption,
    }

    return results


@api_router.get(
    "/monitoring/metrics",
    response_model=schemas.MonitoringMetrics,
    status_code=200,
)
async def get_metrics() -> Any:
    """
    Get monitoring metrics.
    """

    # Download the data from GCS.
    metrics = pd.read_parquet(
        f"{get_settings().GCP_BUCKET}/metrics_monitoring.parquet", filesystem=fs
    )

    datetime_utc = metrics.index.to_list()
    mape = metrics["MAPE"].to_list()

    return {
        "datetime_utc": datetime_utc,
        "mape": mape,
    }


@api_router.get(
    "/monitoring/values/{area}/{consumer_type}",
    response_model=schemas.MonitoringValues,
    status_code=200,
)
async def get_predictions(area: int, consumer_type: int) -> Any:
    """
    Get forecasted predictions based on the given area and consumer type.
    """

    # Download the data from GCS.
    y_monitoring = pd.read_parquet(
        f"{get_settings().GCP_BUCKET}/y_monitoring.parquet", filesystem=fs
    )
    predictions_monitoring = pd.read_parquet(
        f"{get_settings().GCP_BUCKET}/predictions_monitoring.parquet", filesystem=fs
    )

    # Query the data for the given area and consumer type.
    try:
        y_monitoring = y_monitoring.xs(
            (area, consumer_type), level=["area", "consumer_type"]
        )
        predictions_monitoring = predictions_monitoring.xs(
            (area, consumer_type), level=["area", "consumer_type"]
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for the given area and consumer typefrontend: {area}, {consumer_type}",
        )

    if len(y_monitoring) == 0 or len(predictions_monitoring) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for the given area and consumer type: {area}, {consumer_type}",
        )

    print(predictions_monitoring)

    # Prepare data to be returned.
    y_monitoring_datetime_utc = y_monitoring.index.get_level_values(
        "datetime_utc"
    ).to_list()
    y_monitoring_energy_consumption = y_monitoring["energy_consumption"].to_list()

    predictions_monitoring_datetime_utc = predictions_monitoring.index.get_level_values(
        "datetime_utc"
    ).to_list()
    predictions_monitoring_energy_consumptionc = predictions_monitoring[
        "energy_consumption"
    ].to_list()

    results = {
        "y_monitoring_datetime_utc": y_monitoring_datetime_utc,
        "y_monitoring_energy_consumption": y_monitoring_energy_consumption,
        "predictions_monitoring_datetime_utc": predictions_monitoring_datetime_utc,
        "predictions_monitoring_energy_consumptionc": predictions_monitoring_energy_consumptionc,
    }

    return results
```

I want to highlight again that the FastAPI backend only reads the GCS bucket's predictions. The inference step is done solely in the batch prediction pipeline.

You can also go to ["http://<your-ip>:8001/api/v1/docs"](http://35.207.134.188:8001/api/v1/docs) to access the Swagger docs of the API, where you can easily see and test all your endpoints:

![image](https://github.com/Hg03/mlops-paul/assets/69637720/2dc92092-1053-47ab-bcda-3bc65af14c67)

Thats it! Now you know how to build a FastAPI backend. Things might get more complicated when adding a database layer and user sessions, but you learned all the main concepts that will get you started!

## Streamlit predictions dashboard

[Video](https://youtu.be/nVjD0ukG7es)

Access the code under **app-frontend/frontend**.

Using Streamlit is quite simple. The whole UI is defined using the code below that does the following

- it defines the title,
- it makes a request to the backend for all possible area types & creates a dropdown based on it,
- it makes a request to the backend for all possible consumer types & creates a dropdown based on it,
- based on the current chosen area and consumer types, it builds and renders a plotly chart.

```python
import requests

import streamlit as st

from settings import API_URL, TITLE
from components import build_data_plot


st.set_page_config(page_title=TITLE)
st.title(TITLE)


# Create dropdown for area selection.
area_response = requests.get(API_URL / "area_values")

area = st.selectbox(
    label="Denmark is divided in two price areas, or bidding zones,\
        divided by the Great Belt. DK1 (shown as 1) is west of the Great Belt \
            and DK2 (shown as 2) is east of the Great Belt.",
    options=area_response.json().get("values", []),
)

# Create dropdown for consumer type selection.
consumer_type_response = requests.get(API_URL / "consumer_type_values")

consumer_type = st.selectbox(
    label="The consumer type is the Industry Code DE35 which is owned \
          and maintained by Danish Energy, a non-commercial lobby \
              organization for Danish energy companies. \
                The code is used by Danish energy companies.",
    options=consumer_type_response.json().get("values", []),
)


# Check if both area and consumer type have values listed, then create plot for data.
if area and consumer_type:
    st.plotly_chart(build_data_plot(area, consumer_type))
```
Straight forward, right?

Note that we could have made additional checks for the status code of the HTTP requests. For example, if the request status code differs from 200, display a text with "The server is down." But we wanted to keep things simple and emphasize only the Streamlit code ✌️

We moved all the constants to a different file to be easily accessible all over the code. As a next step, you could make them configurable through a **.env** file, similar to the FastAPI setup.

```python
from yarl import URL

TITLE = "Energy Consumption Forecasting"
API_URL = URL("http://172.17.0.1:8001/api/v1")
```

Now, let's see how we built the chart 🔥

This part contains no Streamlit code, only some Pandas and Plotly code.

The **build_data_plot()** function performs 3 main steps:

    It requests the prediction data for an area and consumer type from the FastAPI backend.
    If the response is valid (status_code == 200), it extracts the data from the response and builds a DataFrame from it. Otherwise, it creates an empty DataFrame to pass the same structure further.
    It builds a line plot — plotly graph using the DataFrame computed above.

The role of the **build_dataframe()** function is to take 2 lists:

    a list of datetimes which will be used as the X axis of the line plot;
    a list of values that be used as the Y-axis of the line plot;

…and to convert them into a DataFrame. If some data points are missing, we resample the datetimes to a frequency of 1H to have the data continuous and highlight the missing data points.

```python
from typing import List
import requests

import pandas as pd
import plotly.graph_objects as go

from settings import API_URL


def build_data_plot(area: int, consumer_type: int):
    """
    Build plotly graph for data.
    """

    # Get predictions from API.
    response = requests.get(
        API_URL / "predictions" / f"{area}" / f"{consumer_type}", verify=False
    )
    if response.status_code != 200:
        # If the response is invalid, build empty dataframes in the proper format.
        train_df = build_dataframe([], [])
        preds_df = build_dataframe([], [])

        title = "NO DATA AVAILABLE FOR THE GIVEN AREA AND CONSUMER TYPE"
    else:
        json_response = response.json()

        # Build DataFrames for plotting.
        datetime_utc = json_response.get("datetime_utc")
        energy_consumption = json_response.get("energy_consumption")
        pred_datetime_utc = json_response.get("preds_datetime_utc")
        pred_energy_consumption = json_response.get("preds_energy_consumption")
        
        train_df = build_dataframe(datetime_utc, energy_consumption)
        preds_df = build_dataframe(pred_datetime_utc, pred_energy_consumption)

        title = "Energy Consumption per DE35 Industry Code per Hour"

    # Create plot.
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Arial", size=16),
        ),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Datetime UTC")
    fig.update_yaxes(title_text="Total Consumption")
    fig.add_scatter(
        x=train_df["datetime_utc"],
        y=train_df["energy_consumption"],
        name="Observations",
        line=dict(color="#C4B6B6"),
        hovertemplate="<br>".join(["Datetime: %{x}", "Energy Consumption: %{y} kWh"]),
    )
    fig.add_scatter(
        x=preds_df["datetime_utc"],
        y=preds_df["energy_consumption"],
        name="Predictions",
        line=dict(color="#FFC703"),
        hovertemplate="<br>".join(["Datetime: %{x}", "Total Consumption: %{y} kWh"]),
    )

    return fig


def build_dataframe(datetime_utc: List[int], energy_consumption_values: List[float]):
    """
    Build DataFrame for plotting from timestamps and energy consumption values.
    Args:
        datetime_utc (List[int]): list of timestamp values in UTC 
        values (List[float]): list of energy consumption values
    """

    df = pd.DataFrame(
        list(zip(datetime_utc, energy_consumption_values)),
        columns=["datetime_utc", "energy_consumption"],
    )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], unit="h")

    # Resample to hourly frequency to make the data continuous.
    df = df.set_index("datetime_utc")
    df = df.resample("H").asfreq()
    df = df.reset_index()

    return df
```

Quite simple, right? That is why people love Streamlit.

## Streamlit Monitoring Dashboard

[Video](https://youtu.be/MoiWfQqgWlE)

The monitoring code can be accessed under **app-monitoring/monitoring**.

You will see that the code is almost identical to the predictions dashboard.

When defining the Streamlit UI structure, we additionally implemented a plot containing the aggregated metrics and a divider.

The nice thing about decoupling the definition of the UI components with the data access is that you can inject any data in the UI without modifying it as long as you respect the interface of the expected data.

```python
import requests

import streamlit as st

from settings import API_URL, TITLE
from components import build_metrics_plot, build_data_plot


st.set_page_config(page_title=TITLE)
st.title(TITLE)

# Create plot for metrics over time.
st.plotly_chart(build_metrics_plot())

st.divider()


# Create dropdown for area selection.
area_response = requests.get(API_URL / "area_values")

area = st.selectbox(
    label="Denmark is divided in two price areas, or bidding zones,\
        divided by the Great Belt. DK1 (shown as 1) is west of the Great Belt \
            and DK2 (shown as 2) is east of the Great Belt.",
    options=area_response.json().get("values", []),
)

# Create dropdown for consumer type selection.
consumer_type_response = requests.get(API_URL / "consumer_type_values")

consumer_type = st.selectbox(
    label="The consumer type is the Industry Code DE35 which is owned \
          and maintained by Danish Energy, a non-commercial lobby \
              organization for Danish energy companies. \
                The code is used by Danish energy companies.",
    options=consumer_type_response.json().get("values", []),
)


# Check if both area and consumer type have values listed, then create plot for data.
if area and consumer_type:
    st.plotly_chart(build_data_plot(area, consumer_type))
```

The **build_metrics_plot()** function is almost identical to the **build_data_plot()** function from the predictions dashboard, except for the data we request from the API.

```python
def build_metrics_plot():
    """
    Build plotly graph for metrics.
    """

    response = requests.get(API_URL / "monitoring" / "metrics", verify=False)
    if response.status_code != 200:
        # If the response is invalid, build empty dataframes in the proper format.
        metrics_df = build_dataframe([], [], values_column_name="mape")

        title = "No metrics available."
    else:
        json_response = response.json()

        # Build DataFrame for plotting.
        datetime_utc = json_response.get("datetime_utc", [])
        mape = json_response.get("mape", [])
        metrics_df = build_dataframe(datetime_utc, mape, values_column_name="mape")

        title = "Predictions vs. Observations | Aggregated Metrics"

    # Create plot.
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Arial", size=16),
        ),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Datetime UTC")
    fig.update_yaxes(title_text="MAPE")
    fig.add_scatter(
        x=metrics_df["datetime_utc"],
        y=metrics_df["mape"],
        name="MAPE",
        line=dict(color="#C4B6B6"),
        hovertemplate="<br>".join(["Datetime UTC: %{x}", "MAPE: %{y} kWh"]),
    )

    return fig

```
The same story goes for the **build_data_plot()** function from the monitoring dashboard:

```python
def build_data_plot(area: int, consumer_type: int):
    """
    Build plotly graph for data.
    """

    # Get predictions from API.
    response = requests.get(
        API_URL / "monitoring" / "values" / f"{area}" / f"{consumer_type}", verify=False
    )
    if response.status_code != 200:
        # If the response is invalid, build empty dataframes in the proper format.
        train_df = build_dataframe([], [])
        preds_df = build_dataframe([], [])

        title = "NO DATA AVAILABLE FOR THE GIVEN AREA AND CONSUMER TYPE"
    else:
        json_response = response.json()

        # Build DataFrames for plotting.
        y_monitoring_datetime_utc = json_response.get("y_monitoring_datetime_utc", [])
        y_monitoring_energy_consumption = json_response.get(
            "y_monitoring_energy_consumption", []
        )
        predictions_monitoring_datetime_utc = json_response.get(
            "predictions_monitoring_datetime_utc", []
        )
        predictions_monitoring_energy_consumptionc = json_response.get(
            "predictions_monitoring_energy_consumptionc", []
        )

        train_df = build_dataframe(y_monitoring_datetime_utc, y_monitoring_energy_consumption)
        preds_df = build_dataframe(predictions_monitoring_datetime_utc, predictions_monitoring_energy_consumptionc)
        
        title = "Predictions vs. Observations | Energy Consumption"

    # Create plot.
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Arial", size=16),
        ),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Datetime UTC")
    fig.update_yaxes(title_text="Total Consumption")
    fig.add_scatter(
        x=train_df["datetime_utc"],
        y=train_df["energy_consumption"],
        name="Observations",
        line=dict(color="#C4B6B6"),
        hovertemplate="<br>".join(
            ["Datetime UTC: %{x}", "Energy Consumption: %{y} kWh"]
        ),
    )
    fig.add_scatter(
        x=preds_df["datetime_utc"],
        y=preds_df["energy_consumption"],
        name="Predictions",
        line=dict(color="#FFC703"),
        hovertemplate="<br>".join(
            ["Datetime UTC: %{x}", "Total Consumption: %{y} kWh"]
        ),
    )

    return fig

def build_dataframe(datetime_utc: List[int], energy_consumption_values: List[float], values_column_name: str = "energy_consumption"):
    """
    Build DataFrame for plotting from timestamps and energy consumption values.

    Args:
        datetime_utc (List[int]): list of timestamp values in UTC 
        values (List[float]): list of energy consumption values
        values_column_name (str): name of the column containing the values
    """

    df = pd.DataFrame(
        list(zip(datetime_utc, energy_consumption_values)),
        columns=["datetime_utc", values_column_name],
    )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], unit="h")

    # Resample to hourly frequency to make the data continuous.
    df = df.set_index("datetime_utc")
    df = df.resample("H").asfreq()
    df = df.reset_index()

    return df
```

As you can see, all the data access and manipulation are handled on the FastAPI backend. The Streamlit UI's job is to request and display the data.

It is nice that we just reused 90% of the predictions dashboard code to build a friendly monitoring dashboard.

## Wrap everything with Docker

The final step is to Dockerize the 3 web applications and wrap them up in a docker-compose file.

Thus, we can start the whole web application with a single command:

`docker compose -f deploy/app-docker-compose.yml --project-directory . up --build`

Here is the **FastAPI Docker File**

```
FROM python:3.9.4

WORKDIR /app/src

RUN apt-get update && apt-get upgrade -y
RUN pip install --no-cache -U pip
RUN pip install --no-cache poetry==1.4.2

# Configuring poetry.
RUN poetry config virtualenvs.create false

# First copy & install requirements to speed up the build process in case only the code changes.
COPY ./app-api/pyproject.toml /app/src/
COPY ./app-api/poetry.lock /app/src/

RUN poetry install --no-interaction --no-root -vvv

# Copy the rest of the files.
ADD ./app-api /app/src

# Give access to run the run.sh script.
RUN chmod +x run.sh

CMD ["bash", "./run.sh"]
```
One interesting thing to highlight is that we initially copied & installed only the Poetry dependencies. Thus, when you modify the code, the Docker image will be rebuilt only starting from line 19, aka copying your code.

This is a common strategy to leverage the Docker caching features when building an image to speed up your development process, as you rarely add new dependencies and installing them is the most time-consuming step.

Also, inside **run.sh** we call:

`/usr/local/bin/python -m api`

But wait, there is no Python file in the command 😟

Well, you can actually define a __main__.py file inside a module, making your module executable.

Thus, when calling the api module, you call the __main__.py file:

```python
import uvicorn

from api.config import get_settings


def main() -> None:
    """Entrypoint of the application."""
    uvicorn.run(
        "api.application:get_app",
        workers=get_settings().WORKERS_COUNT,
        host=get_settings().HOST,
        port=get_settings().PORT,
        reload=get_settings().RELOAD,
        log_level=get_settings().LOG_LEVEL.value.lower(),
        factory=True,
    )


if __name__ == "__main__":
    main()
```

In our case, in the __main__.py file, we use the uvicorn web server to start the FastAPI backend and configure it with the right IP, port, log_level, etc.

Here is the **Streamlit predictions dashboard Dockerfile**:

```
FROM python:3.9.8

WORKDIR /app/src

RUN apt-get update && apt-get upgrade -y
RUN pip install --no-cache -U pip
RUN pip install --no-cache poetry==1.4.2

# Configuring poetry.
RUN poetry config virtualenvs.create false

# First copy & install requirements to speed up the build process in case only the code changes.
COPY ./app-frontend/pyproject.toml /app/src/
COPY ./app-frontend/poetry.lock /app/src/

RUN poetry install --no-interaction --no-root -vvv

# Copy the rest of the files.
ADD ./app-frontend /app/src

CMD ["streamlit", "run", "frontend/main.py", "--server.port", "8501"]
```

As you can see, this Dockerfile is almost identical to the one used for the FastAPI backend, except for the last **CMD** command, which is a standard CLI command for starting your Streamlit application.

The _Streamlit monitoring dashboard Dockerfile_ is identical to the predictions dashboard Dockerfile. So it is redundant to copy-paste it here.

The good news is that you can leverage the Dockerfile template I showed you above to Dockerize most of your Python applications ✌️

Finally, let's see how to wrap up everything with docker-compose. You can access the file under **deploy/app-docker-compose.yml:**

```
version: '3.9'

services:
  frontend:
    build:
        dockerfile: app-frontend/Dockerfile
    image: app-frontend:${APP_FRONTEND_VERSION:-latest}
    restart: always
    ports:
      - 8501:8501
    depends_on:
      - api

  monitoring:
    build:
        dockerfile: app-monitoring/Dockerfile
    image: app-monitoring:${APP_MONITORING_VERSION:-latest}
    restart: always
    ports:
      - 8502:8502
    depends_on:
      - api

  api:
    build:
        dockerfile: app-api/Dockerfile
    image: app-api:${APP_API_VERSION:-latest}
    restart: always
    volumes:
      - ./credentials:/app/src/credentials
    env_file:
      - app-api/.env    
    ports:
      - 8001:8001
```

As you can see, the frontend and monitoring services must wait for the API to turn on before starting.

Also, only the API needs to load the credentials from a **.env** file.

Now, you can run the entire web application using only the following command, and Docker will take care of building the images and running the containers:

`docker compose -f deploy/app-docker-compose.yml --project-directory . up --build`

## Conclusion

Congratulations! You finished the **sixth lesson from the Full Stack 7-Steps MLOps Framework course**. It means that now you understand how to consume the predictions of your ML system to build your awesome application.

In this lesson, you learned how to:

- consume the predictions & monitoring metrics from GCS,
- build a FastAPI backend to load and serve the data from GCS,
- implement a dashboard in Streamlit to show the predictions,
- create a monitoring dashboard in Streamlit to visualize the performance of the model.

Now that you understand the flexibility of building an application on top of an ML system that uses a batch prediction architecture, you can easily design full-stack machine learning applications.

Check out Lesson 7 for the final step of the **Full Stack 7-Steps MLOps Framework**, which is to deploy everything to GCP and build a CI/CD pipeline using GitHub Actions.
