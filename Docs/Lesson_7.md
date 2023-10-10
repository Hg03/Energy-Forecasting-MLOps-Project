# Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.

This tutorial represents **lesson 7 out of a 7-lesson course** that will walk you step-by-step through how to **design, implement, and deploy an ML system** using **MLOps good practices**. During the course, you will build a production-ready model to forecast energy consumption levels for the next 24 hours across multiple consumer types from Denmark.

_By the end of this course, you will understand all the fundamentals of designing, coding and deploying an ML system using a batch-serving architecture._

## Table of Contents:

- Course Introduction
- Course Lessons
- Data Source
- Lesson 7: Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.
- Lesson 7: Code
- Conclusion
- References

## Course Introduction

At the end of this 7 lessons course, you will know how to:

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

![image](https://github.com/Hg03/mlops-paul/assets/69637720/16a88309-7213-4172-baf0-72fe6fbababb)

By **the end of Lesson 7**, you will know how to manually deploy the 3 ML pipelines and the web app to GCP. Also, you will build a CI/CD pipeline that will automate the deployment process using GitHub Actions.

## Course Lessons:

1. [Batch Serving. Feature Stores. Feature Engineering Pipelines.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_1.md)
2. [Training Pipelines. ML Platforms. Hyperparameter Tuning.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_2.md)
3. [Batch Prediction Pipeline. Package Python Modules with Poetry.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_3.md)
4. [Private PyPi Server. Orchestrate Everything with Airflow.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_4.md)
5. [Data Validation for Quality and Integrity using GE. Model Performance Continuous Monitoring.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_5.md)
6. [consume and Visualize your Model’s Predictions using FastAPI and Streamlit. Dockerize Everything.](https://github.com/Hg03/mlops-paul/blob/main/Lesson_6.md)
7. **Deploy All the ML Components to GCP. Build a CI/CD Pipeline Using Github Actions.**
8. [Bonus - Behind the Scenes of an ‘Imperfect’ ML Project — Lessons and Insights](https://github.com/Hg03/mlops-paul/blob/main/Bonus.md)

As Lesson 7 focuses on teaching you how to deploy all the components to GCP and build a CI/CD pipeline around it, for the full experience, we recommend you watch the other lessons of the course.

Check out **Lesson 4** to learn how to orchestrate the 3 ML pipelines using Airflow and **Lesson 6** to see how to consume the model's predictions using FastAPI and **Streamlit**.

## The goal of Lesson 7

Within Lesson 7, I will teach you 2 things:

- How to manually deploy the 3 ML pipelines and the web app to GCP.
- How to automate the deployment process with a CI/CD pipeline using GitHub Actions.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/33617899-a01b-4651-8c94-a7a8f064ae19)

In other words, you will take everything you have done so far and show it to the world.

As long your work sits on your computer, it can be the best ML solution in the world, but unfortunately, it won't add any value.

Knowing how to deploy your code is critical to any project.

So remember…

We will use GCP as the cloud provider and GitHub Actions as the CI/CD tool.

## Theoretical Concepts & Tools

**CI/CD:** CI/CD stands for continuous integration and continuous delivery.

The CI step mainly consists of building and testing your code every time you push code to git.

The CD step automatically deploys your code to multiple environments: dev, staging, and production.

Depending on your specific software requirements, you need or do not need all the specifications of a standard CI/CD pipeline.

For example, you might work on a proof of concept. Then a staging environment might be overkill. But having a dev and production CD pipeline will drastically improve your productivity.

**GitHub Actions:** GitHub Actions is one of the most popular CI/CD tools out there in the wild. It is directly integrated into your GitHub repository. The sweet part is that you don't need any VMs to run your CI/CD pipeline. Everything is running on GitHub's computers.

You need to specify a set of rules within a YAML file, and GitHub Actions will take care of the rest. I will show you how it works in this article.

**GitHub Actions** is entirely **free** for public repositories. How awesome is that?

As a side note. Using GitHub Actions, you can trigger any job based on various repository events, but using it as a CI/CD tool is the most common use case.

## Lesson 7: Code

You can access the GitHub repository here.

**Note**: All the installation instructions are in the READMEs of the repository. Here you will jump straight to the code.

_The code and instructions for Lesson 7 are under the following:_

- **deploy/** — Docker and shell deploying files
- **.github/workflows** — GitHub Actions CI/CD workflows
- **README_DEPLOY** — README dedicated to deploying the code to GCP
- **README_CICD** — README dedicated to setting up the CI/CD pipeline

## Prepare Credentials

Directly storing credentials in your git repository is a huge security risk. That is why you will inject sensitive information using a **.env** file.

The **.env.default** is an example of all the variables you must configure. It is also helpful to store default values for attributes that are not sensitive (e.g., project name).

![image](https://github.com/Hg03/mlops-paul/assets/69637720/2db0a0e4-0bb7-4fc5-8e4c-030c594a1f09)

_2 main components can be deployed separately._

**#1. The 3 ML pipelines:**

* Feature Pipeline
* Training Pipeline
* Batch Prediction Pipeline

For **#1.**, you have to set up the following:

- [Hopsworks](https://www.hopsworks.ai/) (free) — Feature Store: Lesson 1
- [W&B](https://wandb.ai/site) (free)— ML Platform: Lesson 2
- [GCS buckets](https://cloud.google.com/storage) (free) — Storage on GCP: Lesson 3
- [Airflow](https://www.airflow.apache.org/) (free)— Open source orchestration tool: Lesson 4

**#2. Web App:**

- [FastAPI](https://fastapi.tiangolo.com) backend (free): Lesson 6
- [Streamlit](https://streamlit.io/) Predictions Dashboard (free): Lesson 6
- [Streamlit](https://streamlit.io/) Monitoring Dashboard (free): Lesson 6

Fortunately, for **#2.**, you have to set up only the [GCP GCS buckets](https://cloud.google.com/storage) used as storage.

But note that if you do only section **#2**., you won't have any data to consume within your web app.


**NOTE:** The only service that doesn’t have a freemium plan is within this lesson. When I wrote this course, deploying and testing the infrastructure on GCP cost me ~20$. But I had a brand new GCP account that offered me 300$ in GCP credits, thus indirectly making it free of charge. Just remember to delete all the GCP resources when you are done, and you will be OK.

## Manually Deploy to GCP

So, let's manually deploy the 2 main components to GCP:

- ML Pipeline
- Web App

But, as a first step, let's set up all the GCP resources we need for the deployment. After, you will SSH to your machines and deploy your code.

For more info, access the GitHub deployment README.

### Set Up Resources

Let’s go to your GCP energy_consumption project and create the following resources:

1. Admin VM Service Account with IAP Access
2. Expose Ports Firewall Rule
3. IAP for TCP Tunneling Firewall Rule
4. VM for the Pipeline
5. VM for the Web App
6. External Static IP

Don’t get discouraged by the fancy names. You will have access to step-by-step guides using this article + the GCP documentation I will provide.

**Note:** If you don’t plan to replicate the infrastructure on your GCP infrastructure, skip the **“Set Up Resources”** section and go directly to **“Deploy the ML Pipeline”**.

**#1. Admin VM Service Account with IAP Access**

We need a new GCP service account with admin rights & IAP access to the GCP VMs.

You have to create a new service account and assign to it the following roles:

* Compute Instance Admin (v1)
* IAP-secured Tunnel User
* Service Account Token Creator
* Service Account User

IAP stands for Identity-Aware Proxy. It is a way to create tunnels that route TCP traffic inside your private network. For your knowledge, you can read more about this topic using the following docs (you don't have to understand it to proceed to the next steps):

- [Using IAP for TCP forwarding](https://cloud.google.com/iap/docs/using-tcp-forwarding)
- [Overview of TCP forwarding](https://cloud.google.com/iap/docs/tcp-forwarding-overview)

**#2. Expose Ports Firewall Rule**

Create a firewall rule that exposes the following TCP ports: 8501, 8502, and 8001.

Also, add a target tag called energy-forecasting-expose-ports.

Here are 2 docs that helped us create and configure the ports for the firewall rule:

- [How to open a specific port such as 9090 in Google Compute Engine](https://stackoverflow.com/questions/21065922/how-to-open-a-specific-port-such-as-9090-in-google-compute-engine)
- [How to Open Firewall Ports on a GCP Compute Engine Instance](https://www.howtogeek.com/devops/how-to-open-firewall-ports-on-a-gcp-compute-engine-instance/)

Here is what our firewall rule looks like 👇

![image](https://github.com/Hg03/mlops-paul/assets/69637720/8dea4688-859b-410f-bfa1-76b503a9368a)


**#3. IAP for TCP Tunneling Firewall Rule**

Now we will create a firewall rule allowing IAP for TCP Tunneling on all the VMs connected to the default network.

[Step-by-step guide on how to create the IAP for TCP tunneling Firewall rule](https://cloud.google.com/iap/docs/using-tcp-forwarding#preparing_your_project_for_tcp_forwarding).

Here is what our firewall rule looks like 👇

![image](https://github.com/Hg03/mlops-paul/assets/69637720/9f740320-c699-4530-9109-f5880622fb67)


**#4. VM for the Pipeline**

Go to your GCP energy_consumption project -> VM Instances -> Create Instance.

Choose e2-standard-2: 2 vCPU cores — 8 GB RAM as your VM instance type.

Call it: ml-pipeline

Change the disk to 20 GB Storage.

Pick region europe-west3 (Frankfurt)` and zone europe-west3-c. Here, you can pick any other region & zone, but if it is your first time doing this, we suggest you do it like us.

Network: default

Also, check the HTTP and HTTPS boxes and add the energy-forecasting-expose-ports custom firewall rule we did a few steps back.

Here are 2 docs that helped me create and configure the ports for the firewall rule:

- [How to open a specific port such as 9090 in Google Compute Engine](https://stackoverflow.com/questions/21065922/how-to-open-a-specific-port-such-as-9090-in-google-compute-engine)
- [How to Open Firewall Ports on a GCP Compute Engine Instance](https://www.howtogeek.com/devops/how-to-open-firewall-ports-on-a-gcp-compute-engine-instance/)

**#5. VM for the Web App**

Now let's repeat a similar process for the Web App VM, but with slightly different settings.

This time choose e2-micro: 0.25 2 vCPU — 1 GB memory as your VM instance type.

Call it: app

Change the disk to 15 GB standard persisted disk

Pick region europe-west3 (Frankfurt) and zone europe-west3-c.

Network: default

Also, check the HTTP and HTTPS boxes and add the energy-forecasting-expose-ports custom firewall rule we created a few steps back.

**#6. External Static IP**

This is the last piece of the puzzle.

If we want the external IP for our web app to be static (aka not to change), we have to attach a static address to our web app VM.

We suggest adding it only to the app VM we created a few steps ahead.

Also, adding a static external IP to the ml-pipeline VM is perfectly fine.

[Docs on reserving a static external IP address](https://cloud.google.com/compute/docs/ip-addresses/reserve-static-external-ip-address).

Now that the boring part is finished let's start deploying the code 👇

## Deploy the ML Pipeline

As a first step, we must install the gcloud [GCP CLI tool](https://cloud.google.com/sdk/docs/install) to talk between our computer and the GCP VMs.

To authenticate, we will use the service account configured with admin rights for VMs and IAP access to SSH.

Now, we must tell the gcloud GCP CLI to use that service account.

To do so, you must create a key for your service account and download it as a JSON file. Same as you did for the buckets service accounts — here are some [docs to refresh your mind](https://cloud.google.com/iam/docs/keys-create-delete).

After you download the file, you have to run the following gcloud command in your terminal:

`gcloud auth activate-service-account SERVICE_ACCOUNT@DOMAIN.COM - key-file=/path/key.json - project=PROJECT_ID`

[Check out this doc for more details about the gcloud auth command](https://cloud.google.com/sdk/gcloud/reference/auth/activate-service-account).

Now whenever you run commands with gcloud, it will use this service account to authenticate.

Now let's connect through SSH to the ml-pipeline GCP VM you created a few steps ahead:

`gcloud compute ssh ml-pipeline - zone europe-west3-c - quiet - tunnel-through-iap - project <your-project-id>`

**NOTE 1:** Change the zone if you haven't created a VM within the same zone as us.
**NOTE 2:** Your project-id is NOT your project name. Go to your GCP projects list and find the project id.

Starting this point, if you configured the firewalls and service account correctly, as everything is Dockerized, all the steps will be 99% similar to those from the rest of the articles.

Check out the Github README — Set Up Additional Tools and Usage sections for step-by-step instructions.

You can follow the same steps while you are connected with SSH to the ml-pipeline GCP machine.

Note that the GCP machine is using Linux as its OS. Thus, you can directly copy & paste the commands from the README regardless of the OS you use on your local device.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/3f8ac73b-de9e-4133-9fdf-e71ea050b712)


You can safely repeat all the steps you've done setting The Pipeline locally using this SSH connection, but you have to keep in mind the following 3 edge cases:

**#1. Clone the code in the home directory of the VM**

Just SHH to the VM and run:

```
git clone https://github.com/iusztinpaul/energy-forecasting.git
cd energy-forecasting
```

**#2. Install Docker using following Commands**

Install Docker

```
sudo apt update
sudo apt install --yes apt-transport-https ca-certificates curl gnupg2 software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt update
sudo apt install --yes docker-ce
```

Add sudo access to Docker

```
sudo usermod -aG docker $USER
logout 
```

Log again to your machine

`gcloud compute ssh ml-pipeline --zone europe-west3-c --quiet --tunnel-through-iap --project <your-project-id>`

[Check out these docs for full instructions](https://tomroth.com.au/gcp-docker/)

**#3. Replace all cp commands with gcloud compute scp:**

This command will help you to copy files from your local machine to the VM.

For example, instead of running:

`cp -r /path/to/admin/gcs/credentials/admin-buckets.json credentials/gcp/energy_consumption`

Run in a different terminal (not the one connected with SSH to your VM):

`gcloud compute scp --recurse --zone europe-west3-c --quiet --tunnel-through-iap --project <your-project-id> /local/path/to/admin-buckets.json ml-pipeline:~/energy-forecasting/airflow/dags/credentials/gcp/energy_consumption/`

This command will copy your local admin-buckets.json file to the ml-pipeline VM.

After setting up your code on the ml-pipeline GCP VM, go to your VM view from GCP and the Network tags section. There you will find the External IP address column, as shown in the image below. Copy that IP and attach port 8080 to it.

For example, based on the External IP address from the image below, I accessed Airflow using this address: 35.207.134.188:8080.

Congrats! You connected to your own self-hosted Airflow application.

**Note:** If it doesn't connect, give it a few seconds to load properly.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/da09a786-7dd6-4c6d-a9dc-c93a05ce05d1)

## Deploy the web app

Let's connect using SSH to the “app” GCP VM you created a few steps ahead:

`gcloud compute ssh app --zone europe-west3-c --quiet --tunnel-through-iap --project <your-project-id>`

**NOTE 1**: Change the zone if you haven't created a VM within the same zone as us.
**NOTE 2**: Your project-id is NOT your project name. Go to your GCP projects list and find the project id.

Here the process is similar to the one described in the **“Deploy the ML Pipeline”** section.

You can deploy the web app following the steps described in Lesson 6 or in the GitHub repository's Set Up Additional Tools & Usage sections.

> But don’t forget to keep in mind the 3 edge cases described in the **“Deploy the ML Pipeline” section**.

Please excuse me for referring you to so much external documentation on how to set up this stuff. The article is too long, and I didn't want to replicate the GCP Google documentation here.

## CI/CD Pipeline using Github Actions (free)

The GitHub Actions YAML files are under the **.github/workflows** directory.

Firstly, let me explain the main components you have to know about a GitHub Actions file 👇

Using the **"on -> push -> branches:"** section, you specify which branch to listen to for events. In this case, the GitHub Action is triggered when new code is committed to the **"main"** branch.

In the **"env: "** section, you can declare the environment variables you need inside the script.

In the **"jobs -> ci_cd -> steps:"** section, you will declare the CI/CD pipeline steps, which will run sequentially.

In the **"jobs -> ci_cd -> runs-on:"** section, you specify the image of the VM you want the steps to run on.

template.yml 

```
name: <some_description>

on:
  push:
    paths-ignore:
      - /any/path/from/repo
    branches: [ "main" ]
    
env:
  ENV_VAR_1: '${{ vars.ENV_VAR_1 }}'

jobs:
  ci_cd:
    runs-on: ubuntu-latest
    steps:
      - <step_1>
      - <step_2>
      - <step_3>
      - ...
```
Now, let's take a look at some actual GitHub Action files 🔥

## ML Pipelines Github Actions YAML file

The action will be triggered when new code is committed to the **"main"** branch, except for the web app directories and the YAML and Markdown files.

We added environment variables that contain information about the GCP project and VM.

As for the CI/CD steps, we mainly do 2 things:

1. configure the credentials & authenticate to GCP,
2. connect with SSH on the given GCP VM and run a command that: goes to the code directory, pulls the latest changes, builds the Python packages, and deploys them to the PyPi registry. Now Airflow will use the new Python packages the next time it runs.

Basically, it does what you would have done manually, but now, everything is nicely automated using GitHub Actions.

Note that you don't have to remember or know how to write a GitHub Actions file from scratch, as you can find already written templates for most of the use cases. For example, here is the [google-github-actions/ssh-compute](https://github.com/google-github-actions/ssh-compute) repository we used to write the YAML file below.

You will find similar templates for almost any use case you have in mind.

```
name: CD/CD for the ml-pipeline that builds all the pipeline modules and pushes them to the private PyPI registry. From where Airflow will install the latest versions and use them in the next run.

on:
  push:
    paths-ignore:
      - 'app-api/'
      - 'app-frontend/'
      - 'app-monitoring/'
      - '**/*.yml'
      - '**/*.md'
    branches: [ "main" ]
    
env:
  CLOUDSDK_CORE_PROJECT: '${{ vars.CLOUDSDK_CORE_PROJECT }}'
  USER: '${{ vars.USER }}'
  INSTANCE_NAME: '${{ vars.ML_PIPELINE_INSTANCE_NAME }}'
  ZONE: '${{ vars.ZONE }}'

jobs:
  ci_cd:
    runs-on: ubuntu-latest
    steps:
      - uses: 'actions/checkout@v3'

      - id: 'auth'
        uses: 'google-github-actions/auth@v0'
        with:
          credentials_json: '${{ secrets.GCP_CREDENTIALS }}'
      - id: 'compute-ssh'
        uses: 'google-github-actions/ssh-compute@v0'
        with:
          project_id: '${{ env.CLOUDSDK_CORE_PROJECT }}'
          user: '${{ env.USER }}'
          instance_name: '${{ env.INSTANCE_NAME }}'
          zone: '${{ env.ZONE }}'
          ssh_private_key: '${{ secrets.GCP_SSH_PRIVATE_KEY }}'
          command: >
            cd ~/energy-forecasting && 
            git pull && 
            sh deploy/ml-pipeline.sh
```

## Web App Github Actions YAML file

The Web App actions file is 90% the same as the one used for the ML pipeline, except for the following:

* we ignore the ML pipeline files;
* we run a docker command that builds and runs the web app.

But where does the **"${{ vars… }}"** weird syntax come from? I will explain in just a sec, but what you have to know now is the following:

- **“${{ vars.<name> }}”**:variables set inside GitHub;

![image](https://github.com/Hg03/mlops-paul/assets/69637720/0b8cf8de-4aea-4ffc-bc51-1a51a5f67b3a)

- **“${{ secrets.<name> }}"**: secrets set inside GitHub. Once a secret is set, you can't see it anymore (the variables you can);

![image](https://github.com/Hg03/mlops-paul/assets/69637720/4bb2a6cd-b8bf-4ef1-839b-4c9335bc0f11)


- **"${{ env.<name> }}"**: environment variables set in the "env:" section.

ci_cd_web_app.yml
```
name: CI/CD for the web app (API + frontend)

on:
  push:
    paths-ignore:
      - 'batch-prediction-pipeline/'
      - 'feature-pipeline/'
      - 'training-pipeline'
      - '**/*.yml'
      - '**/*.md'
    branches: [ "main" ]
    
env:
  CLOUDSDK_CORE_PROJECT: '${{ vars.CLOUDSDK_CORE_PROJECT }}'
  USER: '${{ vars.USER }}'
  INSTANCE_NAME: '${{ vars.APP_INSTANCE_NAME }}'
  ZONE: '${{ vars.ZONE }}'

jobs:
  ci_cd:
    runs-on: ubuntu-latest
    steps:
      - uses: 'actions/checkout@v3'

      - id: 'auth'
        uses: 'google-github-actions/auth@v0'
        with:
          credentials_json: '${{ secrets.GCP_CREDENTIALS }}'
      - id: 'compute-ssh'
        uses: 'google-github-actions/ssh-compute@v0'
        with:
          project_id: '${{ env.CLOUDSDK_CORE_PROJECT }}'
          user: '${{ env.USER }}'
          instance_name: '${{ env.INSTANCE_NAME }}'
          zone: '${{ env.ZONE }}'
          ssh_private_key: '${{ secrets.GCP_SSH_PRIVATE_KEY }}'
          command: >
            cd ~/energy-forecasting && 
            git pull && 
            docker compose -f deploy/app-docker-compose.yml --project-directory . up --build -d
```

**Important Observation**

The YAML file above doesn't contain the CI part, only the CD one.

To follow good practices for a robust CI pipeline, you should run an action that builds the Docker images and pushes them to a Docker registry.

Afterward, you would SSH to a testing environment and run your test suit. As a final step, you would SSH to the production VM, pull the images and run them.

The series got too long, and we wanted to keep it simple, but the good news is that you learned all the necessary tools and principles to do what we described above.
Set Secrets and Variables

At this point, you have to fork the energy-consumption repository to configure the GitHub Actions credentials with your own.

[Check out this doc to see how to fork a repository on GitHub](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

**Set Actions Variables**

Go to your forked repository. After click on: _"Settings -> Secrets and variables -> Actions."_

Now, click _"Variables."_ You can create a new variable by clicking _"New repository variable."_ See the image below 👇

![image](https://github.com/Hg03/mlops-paul/assets/69637720/c706d7a6-d2d6-431d-a194-df7c3dca155b)

You have to create 5 variables that the GitHub Actions scripts will use:

- **APP_INSTANCE_NAME**: the name of the web app VM. In our case, it is called "app". The default should be OK if you use our recommended naming conventions.
- **GCLOUD_PROJECT**: the ID of your GCP Project. Here, you have to change it with your project ID.
- **ML_PIPELINE_INSTANCE_NAME**: the name of the ML pipeline VM. In our case, it is "ml-pipeline". The default should be OK if you use our recommended naming conventions.
- **USER**: the user you used to connect to the VMs while settings up the machine using the SSH connection. Mine was "pauliusztin," but you must change it with yours. Go to the VM and run echo $USER .
- **ZONE**: the zone where you deployed the VMs. The default should be OK if you use our recommended naming conventions.

**Set Action Secrets**

In the same _"Secrets and variables/Actions"_ section, hit the _"Secrets"_ tab.

You can create a new secret by pressing the _“New repository secret”_ button.

These are similar to the variables we just completed, but after you fill in their values, you can't see them anymore. That is why these are called secrets.

Here is where you add all your sensitive information. In our case, the GCP credentials and private keys. See the image below 👇

![image](https://github.com/Hg03/mlops-paul/assets/69637720/059bf8d9-b311-42b8-9323-2d2a28274c16)

The _GCP_CREDENTIALS_ secret contains the content of the JSON key of your VM admin service account. By settings this up, the CI/CD pipeline will use that service account to authenticate to the VMs.

Because the content of the file is in JSON format, to format it properly, you have to do the following steps:

Install the **jq** CLI tool:

```
sudo apt update
sudo apt install -y jq
jq - version
```

Format your JSON Key file

`jq -c . /path/to/your/admin-vm.json`

Take the output of this command and create your GCP_CREDENTIALS secret with it.

The GCP_SSH_PRIVATE_KEY is your GCP private SSH key (not your personal one — GCP creates an additional one automatically), which was created on your local computer when you used SSH to connect to the GCP VMs.

To copy it, run the following:

```
cd ~/.ssh
cat google_compute_engine
```

Copy the output from the terminal and create the GCP_SSH_PRIVATE_KEY variable.

## Run the CI/CD Pipeline

Now make any change to the code, push it to the main branch, and the GitHub Actions files should trigger automatically.

Check your GitHub repository's _“Actions”_ tab to see their results.

![image](https://github.com/Hg03/mlops-paul/assets/69637720/6073c52a-1093-4be4-b3e7-346a7126068c)

Two actions will be triggered. One will build and deploy the ml-pipeline modules to your ml-pipeline GCP VM, and one will build and deploy the web app to your app GCP VM.

## Conclusion

Congratulations! You finished the **last lesson** from the **Full Stack 7-Steps MLOps Framework course**. It means that now you are a full-stack ML engineer 🔥

I apologize again for the highly technical article. It isn't a very entertaining read but a crucial step for finalizing this series.

In lesson 7, you learned how to:

- manually deploy the 3 ML pipelines to GCP;
- manually deploy the web app to GCP;
- build a CI/CD pipeline to automate the deployment process using GitHub Actions.

Now that you understand how to add real business value by deploying your ML system and putting it to work, it is time to build your awesome ML project.

_**No project is perfectly built, and this one is no exception.**_
