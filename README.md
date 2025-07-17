# TorchUnique
Convert any serializable object into a shared object across ranks. This is useful for aggregating results from different ranks during distributed training, implemented based on Torch RPC.

## ⚠️ **Note on Performance**

This tool is designed for aggregating image reconstruction results from multiple GPUs. 

**⚡ Important:** The `sync = True` mode can significantly degrade performance.  
It is highly recommended to avoid using the `sync = True` mode in training code whenever possible.

**💡 Hint:** If it is necessary to share objects during training, please try the `sync = False` mode with `obj.wait()`.