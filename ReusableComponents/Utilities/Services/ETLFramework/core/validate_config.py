"""
validate_config.py

Validation script to test ETL configuration files against the models and schedule loader.

This script:
1. Loads configuration files
2. Validates them against Pydantic models
3. Tests the ScheduleConfigLoader
4. Reports any validation errors

Usage:
    python validate_config.py
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from core.models import ETLConfig
    from core.schedule import ScheduleConfigLoader
except ImportError:
    # Add parent directory to path for imports
    etl_framework_path = Path(__file__).parent.parent
    sys.path.insert(0, str(etl_framework_path))
    from core.models import ETLConfig
    from core.schedule import ScheduleConfigLoader


def validate_config_file(config_path: str) -> bool:
    """
    Validate a configuration file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        bool: True if validation successful, False otherwise.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Validating: {config_path}")
    logger.info(f"{'='*70}")
    
    try:
        # Load and validate configuration
        loader = ScheduleConfigLoader(config_path)
        
        # Display summary
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"✓ Global config: {loader.global_config}")
        logger.info(f"✓ Environment variables: {len(loader.env_config)} defined")
        logger.info(f"✓ Pipelines: {len(loader.pipelines)} defined")
        
        # Display each pipeline
        for idx, pipeline in enumerate(loader.etl_config.pipelines, 1):
            logger.info(f"\n  Pipeline {idx}: {pipeline.name}")
            logger.info(f"    - Description: {pipeline.description}")
            logger.info(f"    - Version: {pipeline.version}")
            logger.info(f"    - Timeout: {pipeline.timeout_seconds}s")
            logger.info(f"    - Tags: {', '.join(pipeline.tags) if pipeline.tags else 'None'}")
            logger.info(f"    - Schedule: {pipeline.schedule.cron} ({pipeline.schedule.timezone})")
            logger.info(f"    - Schedule Enabled: {pipeline.schedule.enabled}")
            logger.info(f"    - Checkpointing: {pipeline.checkpointing.enabled}")
            if pipeline.checkpointing.enabled:
                logger.info(f"      • Location: {pipeline.checkpointing.location}")
                logger.info(f"      • Strategy: {pipeline.checkpointing.strategy}")
            logger.info(f"    - Extractor: {pipeline.extractor.type if pipeline.extractor else 'None'}")
            logger.info(f"    - Transformers: {len(pipeline.transformers)}")
            logger.info(f"    - Loader: {pipeline.loader.type if pipeline.loader else 'None'}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ SUCCESS: Configuration is valid!")
        logger.info(f"{'='*70}\n")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"❌ ERROR: File not found - {e}")
        return False
    except ValueError as e:
        logger.error(f"❌ ERROR: Validation failed - {e}")
        return False
    except Exception as e:
        logger.error(f"❌ ERROR: Unexpected error - {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Main validation function."""
    logger.info("ETL Framework Configuration Validator")
    logger.info("=" * 70)
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Configuration files to validate
    config_files = [
        script_dir / "test.yaml",
        script_dir.parent / "docs" / "yaml_example.yaml"
    ]
    
    results = {}
    
    for config_file in config_files:
        if config_file.exists():
            results[str(config_file)] = validate_config_file(str(config_file))
        else:
            logger.warning(f"⚠️  Configuration file not found: {config_file}")
            results[str(config_file)] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for config_file, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {Path(config_file).name}")
    
    logger.info(f"\nTotal: {passed}/{total} configurations validated successfully")
    logger.info("=" * 70)
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
