# Alzheimer's Disease Prediction Using Kedro

## Overview
This project uses the Kedro framework to build and manage a data pipeline for predicting Alzheimer's disease using machine learning models. We utilize Random Forest, Logistic Regression, and Support Vector Machine (SVM) algorithms to analyze and predict based on patient data. The primary goal of this project is to demonstrate how Kedro can be used to streamline the process of building predictive models and to facilitate their transition into production environments.

## Project Structure
The project follows the standard Kedro project structure:

- `conf/`: Configuration files for datasets, model parameters, and the environment.
- `data/`: Data used by the project, which Kedro handles according to the definitions in `catalog.yml`.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experimentation.
- `src/`: Source code for the project including the pipeline definitions and nodes.
    - `pipelines/`: Modular pipelines for data preprocessing, feature engineering, and model training.
    - `nodes/`: Individual tasks (nodes) that are combined into pipelines.
- `tests/`: Tests for the project's codebase.

## Installation
To get started with this project, clone the repository and set up the environment:
```bash
git clone https://your-repository-url.git
cd your-project-directory
kedro install
```


## Running the Project
To run the full pipeline with Kedro:

```bash
kedro run 
```

You can also run specific pipelines or only parts of the data pipeline if needed:

```bash
kedro run --pipeline data_processing
kedro run --pipeline training
```

## Using the Machine Learning Models
The project uses three different models to predict Alzheimer's disease:

**Random Forest:** Used for its robustness and effectiveness in handling tabular data.
**Logistic Regression:** Provides a probabilistic approach suitable for binary classification.
**SVM:** Offers advantages in high-dimensional spaces and is effective in cases where the margin of separation is important.
Model configurations and parameters can be adjusted in parameters.yml.

## Contributing
Contributions to this project are welcome. Please follow the standard Git workflow:

Fork the repository.
Create your feature branch (git checkout -b feature/AmazingFeature).
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a pull request.

## Contact
For any further questions or partnership inquiries, please contact me at contact.julianafalves@gmail.com.

