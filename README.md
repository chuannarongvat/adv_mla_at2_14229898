adv_mla_at2_14229898
==============================

### Advanced Machine Learning Assignment 2: Machine Learning as a Service

Welcome to our project as part of the "Advanced Machine Learning Application - Spring 2023" course. This initiative revolves around Assignment 2, which emphasizes "Machine Learning as a Service".

### Objective
We are collaborating with an esteemed American retailer that boasts a network of 10 stores strategically spread across three distinctive states: California (CA), Texas (TX), and Wisconsin (WI). Each of these stores offers a diversified product range spanning three major categories: hobbies, foods, and household. The central objectives are:

- **Predictive Modeling:** Using a state-of-the-art Machine Learning Regression Algorithm, we strive to predict the sales revenue accurately for a specific item in a designated store on any given date.

- **Total Sales Forecasting:** By harnessing the power of time-series analysis algorithms, we can forecast the cumulative sales revenue across all stores and items for the subsequent 7 days.

### Endpoints
- **/health/ (GET):** Health check endpoint confirming the service's operational status (status code 200).
- **/sales/stores/items/ (GET):** Interface to the predictive form, enabling users to estimate the sales revenue for a specific item, stores, and dates.
- **/sales/stores/items/ (POST):** Backend endpoint handling the form submission to generate sales revenue predictions.
- **/sales/national/ (GET):** Provides a forecast of the overall sales revenue across all items and stores for the upcoming week.

### Performance Metrics
To ensure our models are reliable, we'll use the Root Mean Square Error (RMSE) as the primary performance metric. This choice is backed by our data's nature and distribution, and RMSE's capability to provide a clear measure of model accuracy. Our models have been trained on historical sales store data spanning from January 29, 2011, to April 18, 2015.

### Running the Application
1. Ensure you have Docker installed on the system.
2. Navigate to the root directory of the project.
3. Build the Docker image: `docker build -t adv_mla_at2_14229898 .`
4. Run the Docker container: `docker run -p 8000:8000 adv_mla_at2_14229898`
5. Acess the application on: `http://localhost:8000`
6. Ensure that all dependencies are installed using `requirements.txt` file: `pip install -r requirements.txt`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── app                <- Application root directory with main.py for running the application.
    │   └── main.py        <- Main application script.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── Dockerfile         <- Dockerfile for building a Docker image of the project.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

Running the Application
1. Ensure you have Docker installed on the system.
2. Navigate to the root directory of the project.
3. Build the Docker image: `docker build -t adv_mla_at2_14229898 .`
4. Run the Docker container: `docker run -p 8000:8000 adv_mla_at2_14229898`
5. Acess the application on: `http://localhost:8000`
6. Ensure that all dependencies are installed using `requirements.txt` file: `pip install -r requirements.txt`



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
