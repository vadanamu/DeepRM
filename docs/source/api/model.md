# model package

```{eval-rst}
.. automodule:: deeprm.model
    :members:
    :undoc-members:
    :show-inheritance:
```


## Submodules

### `model.deeprm_v12`

```{eval-rst}
.. automodule:: deeprm.model.deeprm_v12
    :members: TransformerModel
    :show-inheritance:
```

### `model.deeprm_v12_engines`

```{eval-rst}
.. automodule:: deeprm.model.deeprm_v12_engines
    :members: resolve_engine, compile_status, fa4_status, fa4_installed, engine_form
    :show-inheritance:

.. autoclass:: deeprm.model.deeprm_v12_engines.FastEngine
    :members: forward, set_batch_size, compile_engine
    :show-inheritance:

.. autoclass:: deeprm.model.deeprm_v12_engines.Fa4Engine
    :members: forward, set_batch_size, compile_engine
    :show-inheritance:
```
