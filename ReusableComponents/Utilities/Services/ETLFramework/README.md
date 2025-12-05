# ETL Framework Configuration

A robust and flexible ETL (Extract, Transform, Load) framework with YAML-based configuration, comprehensive validation, and scheduling capabilities.

## 📋 Overview

This ETL framework provides:

- **YAML-based configuration** for pipeline definitions
- **Pydantic models** for type-safe configuration validation
- **Comprehensive schedule management** with cron expressions
- **Multi-pipeline support** with individual scheduling
- **Flexible compute strategies** (local, thread, process, ray, dask, remote)
- **Checkpointing support** for incremental data processing
- **Environment variable management**

## 📁 Project Structure

```
ETLFramework/
├── core/
│   ├── models.py           # Pydantic models for configuration validation
│   ├── schedule.py         # Schedule loader and validator
│   ├── test.yaml           # Test configuration file
│   └── validate_config.py  # Configuration validation script
├── docs/
│   └── yaml_example.yaml   # Complete example configuration
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pydantic pyyaml pytz croniter
```

### 2. Create Configuration File

Create a YAML configuration file following the structure below:

```yaml
global:
  timezone: "UTC"
  log_level: "INFO"
  default_retries: 3
  default_retry_delay: 5

env:
  DATABASE_URL: "postgres://user:pass@localhost:5432/db"
  API_KEY: "${API_KEY}"

pipelines:
  - name: "my_pipeline"
    description: "My ETL pipeline"
    version: "1.0.0"
    timeout_seconds: 3600
    tags:
      - analytics
    schedule:
      cron: "0 2 * * *"
      timezone: "UTC"
      enabled: true
    extractor:
      type: "PostgresExtractor"
      compute_strategy:
        mode: "local"
        workers: 1
      config:
        query: "SELECT * FROM users"
    transformers:
      - type: "CleanData"
        config:
          remove_duplicates: true
    loader:
      type: "S3Loader"
      config:
        bucket: "my-bucket"
```

### 3. Validate Configuration

```bash
python core/validate_config.py
```

Or validate programmatically:

```python
from core.schedule import ScheduleConfigLoader

# Load and validate configuration
loader = ScheduleConfigLoader("config/etl_config.yaml")

# Access pipelines
for pipeline in loader.pipelines:
    print(f"Pipeline: {pipeline['name']}")
```

## 📖 Configuration Reference

### Global Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timezone` | string | "UTC" | Default timezone for all pipelines |
| `log_level` | string | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `default_retries` | int | 3 | Default number of retry attempts |
| `default_retry_delay` | int | 5 | Default retry delay in minutes |

### Pipeline Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✓ | Unique pipeline identifier |
| `description` | string | ✗ | Human-readable description |
| `version` | string | ✗ | Pipeline version (default: "1.0.0") |
| `timeout_seconds` | int | ✗ | Max execution time (default: 3600) |
| `tags` | list[string] | ✗ | Tags for categorization |
| `schedule` | object | ✓ | Schedule configuration |
| `checkpointing` | object | ✗ | Checkpointing configuration |
| `extractor` | object | ✗ | Extractor configuration |
| `transformers` | list[object] | ✗ | List of transformers |
| `loader` | object | ✗ | Loader configuration |

### Schedule Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cron` | string | - | Cron expression (required) |
| `timezone` | string | "UTC" | Timezone for schedule |
| `enabled` | bool | true | Enable/disable schedule |

### Compute Strategy

| Field | Type | Default | Options | Description |
|-------|------|---------|---------|-------------|
| `mode` | string | "local" | local, thread, process, ray, dask, remote | Execution mode |
| `workers` | int | 1 | - | Number of parallel workers |

### Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | false | Enable checkpointing |
| `location` | string | null | Storage location (S3, local path) |
| `strategy` | string | "full" | Strategy: "full" or "incremental" |

## 🔧 Key Features

### 1. Type-Safe Configuration

Uses Pydantic v2 models with:
- `RootModel` for environment variables (replaces deprecated `__root__`)
- `Literal` types for enum-like fields
- Comprehensive field validation
- Default values for optional fields

### 2. Comprehensive Validation

The `ScheduleConfigLoader` validates:
- ✅ YAML syntax and structure
- ✅ Pydantic model compliance
- ✅ Cron expression validity
- ✅ Timezone validity
- ✅ All pipelines in configuration

### 3. Multi-Pipeline Support

- Define multiple pipelines in a single configuration
- Each pipeline has independent scheduling
- Shared global settings and environment variables
- Individual compute strategies per component

### 4. Flexible ETL Components

**Extractor**: Data source configuration
- Type specification (PostgresExtractor, CSVExtractor, etc.)
- Compute strategy
- Custom configuration parameters

**Transformers**: Data transformation steps
- Multiple transformers per pipeline
- Sequential execution
- Independent compute strategies

**Loader**: Data destination configuration
- Single loader per pipeline
- Compute strategy
- Custom configuration parameters

## 📝 Usage Examples

### Example 1: Load and Access Configuration

```python
from core.schedule import ScheduleConfigLoader

# Load configuration
loader = ScheduleConfigLoader("config/etl_config.yaml")

# Access global settings
print(loader.global_config)

# Access environment variables
print(loader.env_config)

# Get specific pipeline
pipeline = loader.get_pipeline_by_name("daily_user_metrics")
print(pipeline['schedule']['cron'])
```

### Example 2: Iterate Through Pipelines

```python
from core.schedule import ScheduleConfigLoader

loader = ScheduleConfigLoader("config/etl_config.yaml")

for pipeline in loader.etl_config.pipelines:
    print(f"Pipeline: {pipeline.name}")
    print(f"  Schedule: {pipeline.schedule.cron}")
    print(f"  Enabled: {pipeline.schedule.enabled}")
    print(f"  Transformers: {len(pipeline.transformers)}")
```

### Example 3: Validate Configuration

```python
from core.models import ETLConfig
import yaml

with open("config/etl_config.yaml") as f:
    raw_config = yaml.safe_load(f)

try:
    etl_config = ETLConfig(**raw_config)
    print("✅ Configuration is valid!")
except Exception as e:
    print(f"❌ Validation failed: {e}")
```

## 🧪 Testing

Run the validation script to test all configurations:

```bash
python core/validate_config.py
```

This will validate:
- `core/test.yaml` - Test configuration
- `docs/yaml_example.yaml` - Example configuration

## 📊 Recent Optimizations

### Models (models.py)
- ✅ Updated to Pydantic v2 syntax
- ✅ Replaced deprecated `__root__` with `RootModel`
- ✅ Added `Literal` types for better type safety
- ✅ Added missing fields: `version`, `timeout_seconds`
- ✅ Consistent use of `default` parameter
- ✅ Improved documentation

### Schedule Loader (schedule.py)
- ✅ Now validates ALL pipelines (not just first one)
- ✅ Added comprehensive error handling
- ✅ Improved logging with pipeline names
- ✅ Added property methods for easy access
- ✅ Added `get_pipeline_by_name()` method
- ✅ Better type hints

### YAML Files
- ✅ Aligned `test.yaml` with complete structure
- ✅ Updated `yaml_example.yaml` for consistency
- ✅ Added missing required fields
- ✅ Fixed `default_retry_delay` (2 → 5 minutes)
- ✅ Consistent indentation and formatting
- ✅ Added helpful comments

## 🎯 Best Practices

1. **Always validate configurations** before deployment
2. **Use appropriate compute strategies** based on workload
3. **Enable checkpointing** for long-running pipelines
4. **Tag pipelines** for easy categorization
5. **Set realistic timeouts** based on expected execution time
6. **Use environment variables** for sensitive data
7. **Test cron expressions** before deployment

## 🔍 Troubleshooting

### Invalid Cron Expression
```
ValueError: Invalid cron expression
```
**Solution**: Verify cron syntax at [crontab.guru](https://crontab.guru)

### Invalid Timezone
```
ValueError: Invalid timezone: XYZ
```
**Solution**: Use valid IANA timezone names (e.g., "UTC", "America/New_York")

### Missing Required Field
```
ValidationError: field required
```
**Solution**: Ensure all required fields are present in configuration

## 📚 Dependencies

- `pydantic>=2.0` - Data validation using Python type annotations
- `pyyaml` - YAML parser and emitter
- `pytz` - World timezone definitions
- `croniter` - Cron expression parsing and validation

## 📄 License

This ETL framework is part of the my-sdk project.

## 🤝 Contributing

When contributing:
1. Validate configurations with `validate_config.py`
2. Update documentation for new features
3. Follow existing code style
4. Add tests for new functionality
