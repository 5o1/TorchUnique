# TorchUnique
Convert any serializable object into a shared object across ranks. This is useful for aggregating results from different ranks during distributed training, implemented based on Torch RPC.
