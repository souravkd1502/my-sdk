"""
schedule.py

Module for loading, validating, and managing ETL pipeline schedule configurations.

This module provides the ScheduleConfigLoader class, which handles:

1. Loading schedule configurations from a YAML file.
2. Validating the structure and contents of the configuration using Pydantic models.
3. Verifying cron expressions for correctness using croniter.
4. Ensuring that specified timezones are valid.
5. Providing a standardized interface for accessing global settings,
    environment variables, and individual pipeline schedules.

Intended Usage:
---------------
- Integrate with ETL frameworks to dynamically schedule Prefect flows.
- Centralize pipeline scheduling logic and configuration management.
- Ensure robust validation of cron schedules, timezones, and other critical parameters
    before runtime execution.

Dependencies:
-------------
- PyYAML: For parsing YAML configuration files.
- pytz: For validating timezones.
- croniter: For validating cron expressions.
- core.models: Pydantic models for configuration validation.

Example:
--------
loader = ScheduleConfigLoader(config_path="config/etl_config.yaml")
for pipeline in loader.pipelines:
    print(f"Pipeline: {pipeline['name']}, Schedule: {pipeline['schedule']}")
"""

import yaml
import pytz
import logging
from typing import List, Dict, Any
from croniter import croniter

# Module logger - use proper initialization pattern
logger = logging.getLogger(__name__)

try:
    from core.models import ETLConfig, ScheduleConfig

except ImportError:
    # Fallback for standalone usage
    import sys
    from pathlib import Path

    etl_framework_path = Path(__file__).parent.parent
    sys.path.insert(0, str(etl_framework_path))

    try:
        from core.models import ETLConfig, ScheduleConfig
    except ImportError as e:
        raise ImportError(
            f"Could not import ETL framework dependencies: {e}. "
            "Ensure the ETL framework is properly installed."
        )


class ScheduleConfigLoader:
    """
    ScheduleConfigLoader loads and validates ETL pipeline configurations from YAML files.
    
    This class provides comprehensive validation for:
    - Global configuration settings
    - Environment variables
    - Pipeline definitions and schedules
    - Cron expressions
    - Timezone specifications
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the ScheduleConfigLoader with the path to the configuration file.
        
        Args:
            config_path (str): The path to the ETL configuration YAML file.
            
        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If the configuration is invalid.
        """
        self.config_path = config_path
        self.raw_config = self._load_yaml()
        self.etl_config = self._validate_full_config()
        self._validate_all_schedules()
        
        logger.info(f"Configuration loaded and validated successfully from {config_path}")
        logger.info(f"Loaded {len(self.pipelines)} pipeline(s)")

    def _load_yaml(self) -> Dict[str, Any]:
        """
        Load the YAML configuration file.
        
        Returns:
            dict: The raw configuration dictionary.
            
        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If the YAML is malformed.
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.debug(f"Successfully loaded YAML from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML: {e}")
            raise ValueError(f"Invalid YAML format: {e}")

    def _validate_full_config(self) -> ETLConfig:
        """
        Validate the complete configuration using Pydantic models.
        
        Returns:
            ETLConfig: Validated ETL configuration object.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        try:
            etl_config = ETLConfig(**self.raw_config)
            logger.debug("Configuration structure validated successfully")
            return etl_config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration structure: {e}")

    def _validate_all_schedules(self) -> None:
        """
        Validate all pipeline schedules including cron expressions and timezones.
        
        Raises:
            ValueError: If any schedule configuration is invalid.
        """
        for idx, pipeline in enumerate(self.etl_config.pipelines):
            pipeline_name = pipeline.name
            schedule = pipeline.schedule
            
            try:
                # Validate cron expression
                if not croniter.is_valid(schedule.cron):
                    raise ValueError(f"Invalid cron expression: {schedule.cron}")
                
                logger.debug(f"Pipeline '{pipeline_name}': Valid cron expression '{schedule.cron}'")
                
                # Validate timezone
                if schedule.timezone and schedule.timezone not in pytz.all_timezones:
                    raise ValueError(f"Invalid timezone: {schedule.timezone}")
                
                logger.debug(f"Pipeline '{pipeline_name}': Valid timezone '{schedule.timezone}'")
                
            except ValueError as e:
                logger.error(f"Validation failed for pipeline '{pipeline_name}': {e}")
                raise ValueError(f"Invalid schedule for pipeline '{pipeline_name}': {e}")

    @property
    def config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        return self.raw_config

    @property
    def pipelines(self) -> List[Dict[str, Any]]:
        """Get list of all pipeline configurations as dictionaries."""
        return self.raw_config.get("pipelines", [])

    @property
    def global_config(self) -> Dict[str, Any]:
        """Get the global configuration settings."""
        return self.raw_config.get("global", {})

    @property
    def env_config(self) -> Dict[str, str]:
        """Get environment variables configuration."""
        return self.raw_config.get("env", {})

    def get_pipeline_by_name(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a specific pipeline configuration by name.
        
        Args:
            name (str): The name of the pipeline.
            
        Returns:
            dict: The pipeline configuration.
            
        Raises:
            KeyError: If no pipeline with the given name exists.
        """
        for pipeline in self.pipelines:
            if pipeline.get("name") == name:
                return pipeline
        raise KeyError(f"Pipeline '{name}' not found in configuration")
