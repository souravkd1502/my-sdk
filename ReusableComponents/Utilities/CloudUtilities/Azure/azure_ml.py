"""
azure_ml.py
----------------

This module provides a set of utility functions and classes to interact with Azure Machine Learning (ML) services, including creating and managing online deployments, retrieving models and datasets, and uploading data to Azure Blob Storage. It is designed to streamline various tasks in an MLOps pipeline by offering reusable components with robust logging and error handling.

Key Features:
---------------
1. **ML Client & Workspace Management**:
    - Provides functions to initialize the Azure ML client and workspace (`get_ml_client_and_workspace`), making it easier to manage connections to the Azure ML environment.

2. **Dataset Retrieval**:
    - Enables retrieval of datasets from Azure ML workspaces and converts them into pandas DataFrames for analysis (`get_data_store`).

3. **Model Management**:
    - Functions to fetch the latest model version (`get_latest_model_version`), retrieve a specific model version (`get_model_by_version`), and deploy models using `ManagedOnlineDeployment`.

4. **Online Endpoint and Deployment**:
    - Simplifies the process of creating or updating online endpoints (`create_or_update_managed_endpoint`) and deploying models to them (`create_online_deployment`).

5. **Azure Blob Storage Interaction**:
    - Utility to upload data to Azure Blob Storage containers (`upload_blob_data`), facilitating data storage and retrieval in MLOps workflows.

Logging:
---------
All operations are logged for traceability and easier debugging, with different levels of logs (info, warning, error). In case of failures, appropriate exceptions are raised with detailed messages to ensure robustness.

Usage Example:
---------------
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Initialize ML client and workspace
ml_client, workspace = get_ml_client_and_workspace("subscription_id", "resource_group", "workspace_name")

# Retrieve the latest version of a registered model
latest_model_version = get_latest_model_version(ml_client, "my_model")

# Create an online deployment for the model
deployment = create_online_deployment(
    ml_client=ml_client,
    registered_model_name="my_model",
    latest_model_version=latest_model_version,
    deployment_name="blue",
    online_endpoint_name="my_endpoint"
)

Modules and Libraries:
-----------------------
- azure.ai.ml: Core Azure ML SDK for managing ML models, deployments, and workspaces.
- azure.identity: Used for authentication with Azure services via DefaultAzureCredential.
- azure.storage.blob: Provides Blob Storage interactions for data upload.
- pandas: Handles dataset conversions into DataFrame for easier data manipulation.

Requirements:
--------------
- azure-ai-ml==1.23.0
- azure-core==1.18.0
- azure-identity==1.6.0
- azure-storage-blob==12.8.1
- azureml-core==1.59.0
- pandas==2.2.3


TODO:
-----

FIXME:
------

Author:
--------
Sourav Das

Date:
------
2024-12-27
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import logging
import pandas as pd
from azure.ai.ml import MLClient
from azureml.core import Workspace, Dataset
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import (
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)

# Set up logging 
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class AzureMLManager:
    """
    A class to manage Azure Machine Learning operations such as workspace, data retrieval,
    model management, and online deployments.

    Attributes:
        subscription_id (str): Azure subscription ID.
        resource_group (str): Azure resource group name.
        workspace_name (str): Azure Machine Learning workspace name.
    """

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """
        Initialize the AzureMLManager class.

        This class is responsible for managing operations with Azure Machine Learning services, including
        workspace and data retrieval, model management, and online deployments.

        Args:
            subscription_id (str): Azure subscription ID.
            resource_group (str): Azure resource group name.
            workspace_name (str): Azure ML workspace name.
            ml_client (MLClient): MLClient instance for interacting with Azure ML services.
            workspace (Workspace): Workspace instance for accessing Azure ML resources.
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.ml_client = None
        self.workspace = None
        self._initialize_clients()

    def _initialize_clients(self):
        """
        Initialize the MLClient and Workspace for Azure Machine Learning operations.

        This method sets up the necessary clients using Azure Default Credentials.
        It logs the initialization status and raises an exception if initialization fails.

        Raises:
            Exception: If there is an error during client initialization.
        """
        try:
            # Obtain Azure credentials using DefaultAzureCredential
            credential = DefaultAzureCredential()
            
            # Initialize the MLClient with the provided subscription, resource group, and workspace
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name,
            )
            
            # Initialize the Workspace object for accessing Azure ML resources
            self.workspace = Workspace(
                subscription_id=self.subscription_id,
                resource_group=self.resource_group,
                workspace_name=self.workspace_name,
            )
            
            _logger.info("Azure ML clients initialized successfully.")
        except Exception as e:
            _logger.error(f"Failed to initialize Azure ML clients: {e}")
            raise

    def get_data_store(self, data_asset_name: str) -> pd.DataFrame:
        """
        Retrieve a dataset from the Azure ML workspace and return it as a pandas DataFrame.

        Args:
            data_asset_name (str): The name of the dataset to retrieve.

        Returns:
            pd.DataFrame: The dataset converted to a pandas DataFrame.

        Raises:
            ValueError: If there is an error retrieving the dataset.
        """
        try:
            # Retrieve the dataset by name from the workspace
            dataset = Dataset.get_by_name(self.workspace, name=data_asset_name)
            
            # Convert the dataset to a pandas DataFrame
            df = dataset.to_pandas_dataframe()
            
            # Log a warning if the DataFrame is empty
            if df.empty:
                _logger.warning(f"The dataset '{data_asset_name}' is empty.")
            
            return df

        except Exception as e:
            # Log the error and raise a ValueError with additional context
            _logger.error(f"Error retrieving dataset '{data_asset_name}': {e}")
            raise ValueError(f"Failed to retrieve dataset '{data_asset_name}'.") from e

    def get_latest_model_version(self, registered_model_name: str) -> int:
        """
        Retrieve the latest version of a registered model.

        Args:
            registered_model_name (str): The name of the registered model.

        Returns:
            int: The latest version number of the registered model.

        Raises:
            ValueError: If no models are found with the given name.
            Exception: For any other errors encountered during retrieval.
        """
        try:
            # List all model versions for the given registered model name
            model_versions = self.ml_client.models.list(name=registered_model_name)

            # Raise an error if no model versions are found
            if not model_versions:
                raise ValueError(f"No models found for '{registered_model_name}'")

            # Return the maximum version number among the listed models
            return max(int(m.version) for m in model_versions)
        
        except Exception as e:
            # Log the error encountered during model version retrieval
            _logger.error(f"Error retrieving model versions: {e}")
            raise

    def get_model_by_version(self, registered_model_name: str, model_version: int) -> Model:
        """
        Retrieve a specific version of a registered model.

        Args:
            registered_model_name (str): The name of the registered model.
            model_version (int): The version number of the model to retrieve.

        Returns:
            Model: The specified version of the registered model.

        Raises:
            Exception: If an error occurs during the retrieval process.
        """
        try:
            # Retrieve the model by name and version from the workspace
            model = self.ml_client.models.get(name=registered_model_name, version=model_version)
            return model

        except Exception as e:
            # Log the error encountered during model retrieval
            _logger.error(f"Error retrieving model: {e}")
            raise

    def create_or_update_managed_endpoint(
        self, online_endpoint_name: str, description: str = "Online endpoint", auth_mode: str = "key", tags: dict = None
    ) -> ManagedOnlineEndpoint:
        """
        Create or update a managed online endpoint.

        Args:
            online_endpoint_name (str): The name of the online endpoint.
            description (str): The description of the online endpoint. Defaults to "Online endpoint".
            auth_mode (str): The authentication mode for the online endpoint. Defaults to "key".
            tags (dict): The tags to associate with the online endpoint. Defaults to {"default": "AzureML"}.

        Returns:
            ManagedOnlineEndpoint: The created or updated online endpoint.

        Raises:
            Exception: If an error occurs during the creation or update process.
        """
        tags = tags or {"default": "AzureML"}
        endpoint = ManagedOnlineEndpoint(
            name=online_endpoint_name, description=description, auth_mode=auth_mode, tags=tags
        )
        try:
            return self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        except Exception as e:
            _logger.error(f"Error creating or updating endpoint: {e}")
            raise

    def create_online_deployment(
        self,
        registered_model_name: str,
        latest_model_version: int,
        deployment_name: str,
        online_endpoint_name: str,
        conda_file: str = "./deploy/env.yaml",
        scoring_script_path: str = "./deploy",
        scoring_script: str = "score.py",
        docker_image: str = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        instance_type: str = "Standard_DS2_v2",
        instance_count: int = 1,
    ) -> ManagedOnlineDeployment:
        """
        Create an online deployment for a registered model.

        Args:
            registered_model_name (str): The name of the registered model.
            latest_model_version (int): The version number of the model to deploy.
            deployment_name (str): The name of the online deployment.
            online_endpoint_name (str): The name of the online endpoint to associate with the deployment.
            conda_file (str): The path to the Conda environment file. Defaults to ./deploy/env.yaml.
            scoring_script_path (str): The path to the folder containing the scoring script. Defaults to ./deploy.
            scoring_script (str): The name of the scoring script. Defaults to score.py.
            docker_image (str): The Docker image to use for the deployment. Defaults to mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04.
            instance_type (str): The type of instance to use for the deployment. Defaults to Standard_DS2_v2.
            instance_count (int): The number of instances to use for the deployment. Defaults to 1.

        Returns:
            ManagedOnlineDeployment: The created online deployment.

        Raises:
            Exception: If an error occurs during the creation process.
        """
        try:
            model = self.get_model_by_version(registered_model_name, latest_model_version)
            deployment = ManagedOnlineDeployment(
                name=deployment_name,
                endpoint_name=online_endpoint_name,
                model=model,
                environment={
                    "conda_file": conda_file,
                    "docker_image": docker_image,
                    "scoring_script": scoring_script_path + "/" + scoring_script,
                },
                instance_type=instance_type,
                instance_count=instance_count,
            )
            return self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        except Exception as e:
            _logger.error(f"Error creating deployment: {e}")
            raise
