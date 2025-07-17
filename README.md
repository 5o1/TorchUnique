# TorchUnique
Convert any serializable object into a shared object across ranks. This is useful for aggregating results from different ranks during distributed training, implemented based on Torch RPC.

## ⚠️ **Note on Performance**

The device tensors within the object to be converted must be located on the CPU. If you need to aggregate tensors on the GPU, please use `torch.all_gather` and `torch.all_gather_object`. 

However, while `torch.all_gather` and `torch.all_gather_object` must be called synchronously across all ranks, this tool does not require synchronous calls.

This tool is designed for aggregating image reconstruction results from multiple GPUs. 

 [[Related issue]](https://discuss.pytorch.org/t/gathering-dictionaries-of-distributeddataparallel/51381)

**⚡ Important:** The `sync = True` mode can significantly degrade performance.  
It is highly recommended to avoid using the `sync = True` mode in training code whenever possible.

**💡 Hint:** If it is necessary to share objects during training, please try the `sync = False` mode with `obj.wait()`.