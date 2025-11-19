# ZenML Pipeline Tutorial 🔄

Transform your MLPotion pipeline into a production-ready MLOps workflow with ZenML tracking, versioning, and reproducibility!

**Note:** ZenML is just one integration example. MLPotion is designed to be **orchestrator-agnostic** and works with Prefect, Airflow, Kubeflow, and any other orchestration platform. See [ZenML Integration Guide](../integrations/zenml.md) for extending to other orchestrators.

## Prerequisites 📋

```bash
pip install mlpotion[tensorflow,zenml]
zenml init
```

## Converting Your Pipeline 🔄

### Before: Standalone Pipeline

```python linenums="1"
--8<-- "docs/examples/tensorflow/standalone.py"
```

### After: ZenML Pipeline

```python linenums="1"
--8<-- "docs/examples/tensorflow/zenml_pipeline.py"
```

## Benefits You Get 🌟

- ✅ Automatic artifact versioning
- ✅ Full pipeline lineage tracking
- ✅ Experiment comparison
- ✅ Model registry integration
- ✅ Reproducible runs
- ✅ Caching of unchanged steps
- ✅ Step output tracking with metadata

## Viewing Pipeline Runs 🔍

```bash
# List all pipeline runs
zenml pipeline runs list

# View details of a specific run
zenml pipeline runs describe <RUN_ID>

# Compare different runs
zenml pipeline runs compare <RUN_ID_1> <RUN_ID_2>
```

See the [ZenML Integration Guide](../integrations/zenml.md) for complete documentation and examples with PyTorch and Keras!

---

<p align="center">
  <strong>Ready for production MLOps!</strong> 🚀
</p>
