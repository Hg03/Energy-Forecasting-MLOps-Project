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

By the **end of Lesson 4**, you will know how to implement and integrate the **batch prediction pipeline** and **package all the Python modules using Poetry**.

**Course Lessons:**

1. [Batch Serving. Feature Stores. Feature Engineering Pipelines.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
2. [Training Pipelines. ML Platforms. Hyperparameter Tuning.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_2.md)
3. [Batch Prediction Pipeline. Package Python Modules with Poetry.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
4. [Private PyPi Server. Orchestrate Everything with Airflow.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
5. [Data Validation for Quality and Integrity using GE. Model Performance Continuous Monitoring.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
6. [consume and Visualize your Model’s Predictions using FastAPI and Streamlit. Dockerize Everything.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
7. [Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
8. [Bonus] Behind the Scenes of an ‘Imperfect’ ML Project — Lessons and Insights

If you want to grasp this lesson fully, we recommend you check out our previous lesson, which talks about designing a training pipeline that uses a feature store and an ML platform:


