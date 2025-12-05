"""
models.py

Module for defining and validating ETL pipeline configuration using Pydantic models.

This module provides structured models to represent global settings, environment variables,
pipeline definitions, schedules, checkpointing, and ETL component configurations
(extractor, transformers, loader).

It is intended to:
1. Load configuration from YAML/JSON files.
2. Validate required fields, data types, and optional defaults.
3. Provide a standard interface for ETL framework components like Prefect flows.
4. Ensure that pipeline definitions are consistent, robust, and ready for execution.

Example Usage:
--------------
import yaml
from models import ETLConfig

with open("etl_config.yaml") as f:
    raw_config = yaml.safe_load(f)

etl_config = ETLConfig(**raw_config)
"""

from pydantic import BaseModel, Field, RootModel, field_validator
from typing import Dict, List, Optional, Literal


# ---------------------------
# Global / Environment Models
# ---------------------------


class GlobalConfig(BaseModel):
    """
    Represents global configuration settings applied to all pipelines.

    Attributes:
    -----------
    timezone : str
        Default timezone for all pipelines (used if pipeline-specific timezone is not set).
    log_level : str
        Logging level for the ETL framework (e.g., DEBUG, INFO, WARNING).
    default_retries : int
        Default number of retry attempts for tasks or pipeline failures.
    default_retry_delay : int
        Default delay in minutes between retry attempts.
    """

    timezone: str = Field("UTC", description="Default timezone for pipelines")
    log_level: str = Field("INFO", description="Logging level")
    default_retries: int = Field(
        3, ge=0, description="Default retry attempts for tasks"
    )
    default_retry_delay: int = Field(
        5, ge=0, description="Default retry delay in minutes"
    )


class EnvironmentConfig(RootModel[Dict[str, str]]):
    """
    Represents environment variables available to pipelines.

    Attributes:
    -----------
    root : Dict[str, str]
        Mapping of environment variable names to their values.
    """

    root: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )


# ---------------------------
# Compute Strategy Model
# ---------------------------


class ComputeStrategy(BaseModel):
    """
    Defines how a pipeline component (extractor, transformer, loader) should be executed.

    Attributes:
    -----------
    mode : str
        Execution mode. Options: 'local', 'thread', 'process', 'ray', 'dask', 'remote'.
    workers : Optional[int]
        Number of parallel workers or threads to use (if applicable).
    """

    mode: Literal["local", "thread", "process", "ray", "dask", "remote"] = Field(
        default="local",
        description="Execution mode (local, thread, process, ray, dask, remote)",
    )
    workers: Optional[int] = Field(
        default=1, ge=1, description="Number of workers if applicable"
    )


# ---------------------------
# Schedule & Checkpointing
# ---------------------------


class ScheduleConfig(BaseModel):
    """
    Represents the scheduling configuration for a pipeline.

    Attributes:
    -----------
    cron : str
        Cron expression defining pipeline execution frequency.
    timezone : Optional[str]
        Timezone for the schedule. Defaults to 'UTC'.
    enabled : Optional[bool]
        Whether the schedule is active. Defaults to True.
    """

    cron: str = Field(..., description="Cron expression for schedule")
    timezone: Optional[str] = Field("UTC", description="Timezone for the schedule")
    enabled: Optional[bool] = Field(True, description="Enable or disable the schedule")


class CheckpointConfig(BaseModel):
    """
    Represents checkpointing configuration for a pipeline.

    Attributes:
    -----------
    enabled : bool
        Whether checkpointing is enabled.
    location : Optional[str]
        Storage location for checkpoint data (e.g., S3 bucket, local path).
    strategy : Optional[str]
        Strategy for checkpointing: 'full' or 'incremental'.
    """

    enabled: bool = Field(default=False, description="Enable checkpointing")
    location: Optional[str] = Field(default=None, description="Checkpoint storage location")
    strategy: Literal["full", "incremental"] = Field(
        default="full", description="Checkpoint strategy: full | incremental"
    )


# ---------------------------
# ETL Component Models
# ---------------------------


class ExtractorConfig(BaseModel):
    """
    Represents configuration for a pipeline extractor component.

    Attributes:
    -----------
    type : str
        Type of extractor (e.g., PostgresExtractor, CSVExtractor).
    compute_strategy : Optional[ComputeStrategy]
        How the extractor task should be executed.
    config : Dict
        Component-specific configuration parameters (e.g., query, connection_id).
    """

    type: str = Field(..., description="Type of extractor")
    compute_strategy: Optional[ComputeStrategy] = Field(default_factory=ComputeStrategy)
    config: Dict = Field(
        default_factory=dict, description="Extractor-specific configuration"
    )


class TransformerConfig(BaseModel):
    """
    Represents configuration for a pipeline transformer component.

    Attributes:
    -----------
    type : str
        Type of transformer (e.g., CleanUserData, NormalizeFields).
    compute_strategy : Optional[ComputeStrategy]
        How the transformer task should be executed.
    config : Dict
        Component-specific configuration parameters.
    """

    type: str = Field(..., description="Type of transformer")
    compute_strategy: Optional[ComputeStrategy] = Field(default_factory=ComputeStrategy)
    config: Dict = Field(
        default_factory=dict, description="Transformer-specific configuration"
    )


class LoaderConfig(BaseModel):
    """
    Represents configuration for a pipeline loader component.

    Attributes:
    -----------
    type : str
        Type of loader (e.g., S3Loader, DatabaseLoader).
    compute_strategy : Optional[ComputeStrategy]
        How the loader task should be executed.
    config : Dict
        Component-specific configuration parameters.
    """

    type: str = Field(..., description="Type of loader")
    compute_strategy: Optional[ComputeStrategy] = Field(default_factory=ComputeStrategy)
    config: Dict = Field(
        default_factory=dict, description="Loader-specific configuration"
    )


# ---------------------------
# Pipeline Model
# ---------------------------


class PipelineConfig(BaseModel):
    """
    Represents the configuration for a single ETL pipeline.

    Attributes:
    -----------
    name : str
        Unique identifier for the pipeline.
    description : Optional[str]
        Human-readable description of the pipeline.
    version : Optional[str]
        Version of the pipeline configuration.
    timeout_seconds : Optional[int]
        Maximum execution time in seconds for the pipeline.
    tags : Optional[List[str]]
        List of tags for categorization and filtering.
    schedule : ScheduleConfig
        Scheduling configuration for the pipeline.
    checkpointing : Optional[CheckpointConfig]
        Checkpointing configuration.
    extractor : Optional[ExtractorConfig]
        Extractor component configuration.
    transformers : Optional[List[TransformerConfig]]
        List of transformer component configurations.
    loader : Optional[LoaderConfig]
        Loader component configuration.
    """

    name: str = Field(..., description="Unique pipeline identifier")
    description: Optional[str] = Field(default=None, description="Human-readable description")
    version: Optional[str] = Field(default="1.0.0", description="Pipeline version")
    timeout_seconds: Optional[int] = Field(
        default=3600, ge=1, description="Pipeline timeout in seconds"
    )
    tags: Optional[List[str]] = Field(default_factory=list, description="Pipeline tags")
    schedule: ScheduleConfig = Field(..., description="Pipeline schedule configuration")
    checkpointing: Optional[CheckpointConfig] = Field(default_factory=CheckpointConfig)
    extractor: Optional[ExtractorConfig] = Field(
        default=None, description="Extractor configuration"
    )
    transformers: Optional[List[TransformerConfig]] = Field(default_factory=list)
    loader: Optional[LoaderConfig] = Field(default=None, description="Loader configuration")


# ---------------------------
# Root ETL Config Model
# ---------------------------


class ETLConfig(BaseModel):
    """
    Represents the complete ETL configuration including global settings, environment variables, and pipelines.

    Attributes:
    -----------
    global_config : GlobalConfig
        Global settings applied to all pipelines (alias: 'global').
    env : EnvironmentConfig
        Environment variables available to pipelines.
    pipelines : List[PipelineConfig]
        List of defined ETL pipelines with their configurations.
    """

    global_config: GlobalConfig = Field(..., alias="global")
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    pipelines: List[PipelineConfig] = Field(
        default_factory=list, description="List of ETL pipelines"
    )
