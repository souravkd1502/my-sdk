"""


"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import logging
import pandas as pd
from dotenv import load_dotenv

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Workspace,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient
from azure.identity import ClientSecretCredential

from typing import Literal, List, Any, Dict, Optional, Union

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables
load_dotenv(override=True)


class AzureMLService:
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """
        Initialize the AzureMLService class for Azure ML operations.

        Args:
            subscription_id (str): Azure subscription ID.
            resource_group (str): Azure resource group name.
            workspace_name (str): Azure ML workspace name.
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.ml_client, self.workspace = self.get_ml_client_and_workspace()
        
    @staticmethod
    def _get_credentials(
        tenant_id: str = None,
        client_id: str = None,
        client_secret: str = None,
    ) -> ClientSecretCredential:
        """
        """
        try:
            # Validate input
            if not (tenant_id and client_id and client_secret):
                raise ValueError(
                    "tenant_id, client_id, and client_secret must be provided."
                )

            # Authenticate using Azure credentials
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
            
            return credential
        except Exception as e:
            _logger.error(f"Failed to authenticate with Azure credentials: {str(e)}")
            raise AzureError(f"Failed to authenticate with Azure credentials: {str(e)}")


    def get_ml_client_and_workspace(self) -> tuple[MLClient, Workspace]:
        """
        Initialize and return the MLClient & workspace client for Azure Machine Learning workspace.

        Returns:
            tuple[MLClient, Workspace]: A tuple containing MLClient and Workspace instances.
        """
        try:
            credential = {}
            _logger.info("Successfully obtained Azure credentials.")

            ml_client = MLClient(
                credential=credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name,
            )
            _logger.info(f"MLClient initialized for workspace: {self.workspace_name}")

            workspace = Workspace(
                self.subscription_id, self.resource_group, self.workspace_name
            )
            _logger.info(f"Workspace initialized for workspace: {self.workspace_name}")

            return ml_client, workspace
        except Exception as ex:
            _logger.error(f"Failed to initialize MLClient or workspace: {str(ex)}")
            raise

    def get_data_store(self, data_asset_name: str) -> pd.DataFrame:
        """
        Retrieve a dataset from the Azure ML Workspace by its name and convert it into a pandas DataFrame.

        Args:
            data_asset_name (str): The name of the dataset to retrieve.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the data from the dataset.
        """
        try:
            _logger.info(f"Attempting to retrieve dataset: {data_asset_name}")
            dataset = Dataset.get_by_name(self.workspace, name=data_asset_name)
            _logger.info(f"Converting dataset '{data_asset_name}' to pandas DataFrame.")
            df = dataset.to_pandas_dataframe()

            if df.empty:
                _logger.warning(f"The dataset '{data_asset_name}' is empty.")
            return df
        except Exception as e:
            _logger.error(
                f"Error retrieving or converting dataset '{data_asset_name}': {e}"
            )
            raise ValueError(
                f"Failed to retrieve or convert dataset '{data_asset_name}' due to {str(e)}"
            )

    def get_latest_model_version(self, registered_model_name: str) -> int:
        """
        Retrieves the latest version number of a registered model in Azure ML.

        Args:
            registered_model_name (str): The name of the registered model to retrieve the latest version for.

        Returns:
            int: The latest model version.
        """
        try:
            _logger.info(f"Fetching model list for: {registered_model_name}")
            model_versions = self.ml_client.models.list(name=registered_model_name)

            if not model_versions:
                _logger.error(f"No models found for '{registered_model_name}'")
                raise ValueError(f"No models found for '{registered_model_name}'")

            latest_version = max(int(m.version) for m in model_versions)
            _logger.info(
                f"Latest version for model '{registered_model_name}' is: {latest_version}"
            )

            return latest_version
        except Exception as e:
            _logger.error(
                f"Error retrieving the latest model version for '{registered_model_name}': {e}"
            )
            raise ValueError(
                f"Failed to retrieve the latest model version for '{registered_model_name}'"
            )

    def get_model_by_version(
        self, registered_model_name: str, latest_model_version: int
    ) -> Model:
        """
        Retrieves a registered model from Azure ML by its name and version.

        Args:
            registered_model_name (str): The name of the registered model.
            latest_model_version (int): The version number of the model to retrieve.

        Returns:
            Model: The retrieved model object.
        """
        try:
            latest_model_version = int(latest_model_version)
            _logger.info(
                f"Fetching model '{registered_model_name}' version {latest_model_version}."
            )
            model = self.ml_client.models.get(
                name=registered_model_name, version=latest_model_version
            )
            _logger.info(
                f"Successfully retrieved model '{model.name}' version {model.version}."
            )
            return model
        except Exception as e:
            _logger.error(
                f"Error fetching model '{registered_model_name}' version {latest_model_version}: {e}"
            )
            raise ValueError(
                f"Failed to retrieve model '{registered_model_name}' version {latest_model_version}"
            ) from e

    def create_or_update_managed_endpoint(
        self,
        online_endpoint_name: str,
        description: str = "this is an online endpoint",
        auth_mode: str = "key",
        tags: dict = None,
    ) -> ManagedOnlineEndpoint:
        """
        Creates or updates a managed online endpoint in Azure ML.

        Args:
            online_endpoint_name (str): The name of the online endpoint.
            description (str): A description for the online endpoint. Defaults to 'this is an online endpoint'.
            auth_mode (str): Authentication mode for the endpoint. Defaults to 'key'.
            tags (dict): Tags to associate with the endpoint. Defaults to None.

        Returns:
            ManagedOnlineEndpoint: The created or updated Managed online endpoint.
        """
        if tags is None:
            tags = {"default": "Alloc8"}

        try:
            _logger.info(f"Defining online endpoint '{online_endpoint_name}'.")
            endpoint = ManagedOnlineEndpoint(
                name=online_endpoint_name,
                description=description,
                auth_mode=auth_mode,
                tags=tags,
            )
        except Exception as ex:
            _logger.error(f"Error defining endpoint '{online_endpoint_name}': {ex}")
            raise AzureError(f"Failed to define online endpoint: {ex}") from ex

        try:
            _logger.info(f"Creating or updating the endpoint '{online_endpoint_name}'.")
            endpoint = self.ml_client.online_endpoints.begin_create_or_update(
                endpoint
            ).result()
            _logger.info(
                f"Endpoint '{online_endpoint_name}' created or updated successfully."
            )
            return endpoint
        except Exception as ex:
            _logger.error(
                f"Error creating or updating endpoint '{online_endpoint_name}': {ex}"
            )
            raise AzureError(
                f"Failed to create or update the online endpoint: {ex}"
            ) from ex

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
        Creates an online deployment for a model in Azure ML.
        And deploys it on an Online Managed Deployment.

        Args:
            registered_model_name (str): The name of the registered model to deploy.
            latest_model_version (int): The version of the registered model.
            deployment_name (str): The name of the deployment (e.g., "blue").
            online_endpoint_name (str): The name of the online endpoint.
            conda_file (str): Path to the conda environment YAML file. Defaults to './deploy/env.yaml'.
            scoring_script_path (str): Path to the parent directory for the scoring script (inference script).
            scoring_script (str): Name of the scoring script. Defaults to 'score.py'.
            docker_image (str): Docker image for the environment. Defaults to OpenMPI Ubuntu 20.04.
            instance_type (str): The VM instance type for the deployment. Defaults to 'Standard_DS2_v2'.
            instance_count (int): Number of instances to deploy. Defaults to 1.

        Returns:
            ManagedOnlineDeployment: The deployed model instance.
        """
        try:
            _logger.info(
                f"Fetching model '{registered_model_name}' version {latest_model_version}."
            )
            model = self.get_model_by_version(
                registered_model_name, latest_model_version
            )

            environment = Environment.from_conda_specification(
                name=deployment_name, conda_file=conda_file
            )
            _logger.info(
                f"Environment defined from Conda file for deployment '{deployment_name}'."
            )

            code_config = CodeConfiguration(
                source_directory=scoring_script_path, entry_script=scoring_script
            )

            deployment = ManagedOnlineDeployment(
                name=deployment_name,
                endpoint_name=online_endpoint_name,
                model=model,
                environment=environment,
                code_configuration=code_config,
                instance_type=instance_type,
                instance_count=instance_count,
                docker_image=docker_image,
            )
            _logger.info(f"Creating or updating the deployment '{deployment_name}'.")
            deployment = self.ml_client.online_deployments.begin_create_or_update(
                deployment
            ).result()
            _logger.info(
                f"Deployment '{deployment_name}' created or updated successfully."
            )
            return deployment
        except Exception as ex:
            _logger.error(
                f"Error creating or updating deployment '{deployment_name}': {ex}"
            )
            raise AzureError(
                f"Failed to create or update deployment '{deployment_name}': {ex}"
            ) from ex
