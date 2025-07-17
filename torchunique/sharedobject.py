from torch import distributed as dist
from torch.distributed import rpc
import torch
import functools
import inspect
import hashlib
import warnings
import atexit

class DistInitProcessGroupTracker:
    """
    A class to monkey patch `dist.init_process_group` with a tracker that captures
    the input arguments (`rank` and `init_method`) for RPC initialization.  
    It is useful in ddp strategy in ONE single node.  
    """

    _patch_applied = False

    @classmethod
    def tracker(cls, func):
        """
        A decorator to wrap `dist.init_process_group` and capture its input arguments.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            rank = bound_args.arguments.get("rank", None)
            init_method = bound_args.arguments.get("init_method", None)
            device_id = bound_args.arguments.get("device_id", None)

            TorchRpcUtils.rpc_init_method[rank] = init_method
            TorchRpcUtils.device_id[rank] = device_id

            return func(*args, **kwargs)

        return wrapper

    @classmethod
    def patch(cls):
        """
        Monkey patch `dist.init_process_group` to use the tracker.  
        This function should be executed when the package is initialized.  
        This method can only be called once. Subsequent calls will raise a RuntimeError.  
        """
        if cls._patch_applied:
            raise RuntimeError("InitProcessGroupTracker.patch can only be applied once.")

        cls._patch_applied = True

        dist.init_process_group = cls.tracker(dist.init_process_group)

class TorchRpcUtils:
    rpc_init_method = dict()
    device_id = dict()
    buffer = dict() # contextId: (Obj, [Consumers List])
    rank_0_worker_name = "worker0"

    @staticmethod
    def is_rpc_initialized():
        try:
            rpc.get_worker_info()
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def barrier():
        if dist.is_initialized():
            device_id = TorchRpcUtils.device_id[dist.get_rank()]
            if device_id is None:
                dist.barrier()
            else:
                dist.barrier(device_ids=[TorchRpcUtils.device_id[dist.get_rank()]])

    @staticmethod
    def rpc_close():
        if TorchRpcUtils.is_rpc_initialized():
            rpc.shutdown()

    @staticmethod
    def rpc_init(rank: int):
        if TorchRpcUtils.is_rpc_initialized():
            warnings.warn("Torch RPC has been initialized by other method. Please make sure the node name is worker{rank}.")
            return

        # rpc init
        with warnings.catch_warnings(record=True) as caught_warnings:
            if rank in TorchRpcUtils.rpc_init_method:
                init_method = TorchRpcUtils.rpc_init_method[rank]
                if init_method is not None:
                    rpc.init_rpc(
                        name=f"worker{rank}",
                        rank=rank,
                        world_size=dist.get_world_size(),
                        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                            init_method=init_method,
                        )
                    )
                    atexit.register(TorchRpcUtils.rpc_close)
                    return
            try:
                rpc.init_rpc(
                        name=f"worker{rank}",
                        rank=rank,
                        world_size=dist.get_world_size(),
                        rpc_backend_options=rpc.TensorPipeRpcBackendOptions()
                )
                atexit.register(TorchRpcUtils.rpc_close)
            except Exception as e:
                raise NotImplementedError(f"Attempt to initialize RPC using the same method failed. Initialization methods other than this are currently not supported. Captured init_method is {TorchRpcUtils.rpc_init_method}")
        for w in caught_warnings:
            if "You are using a Backend <class 'torch.distributed.distributed_c10d.ProcessGroupGloo'> as a ProcessGroup. This usage is deprecated since PyTorch 2.0. Please use a public API of PyTorch Distributed instead." in str(w.message):
                warnings.filterwarnings(
                    "ignore", 
                    message=r"You are using a Backend <class 'torch.distributed.distributed_c10d.ProcessGroupGloo'> as a ProcessGroup.*",
                    category=UserWarning
                )
                warnings.warn(
                    "TorchUnique uses `Torch RPC` as its backend. You are receiving this warning because Torch RPC uses the deprecated `ProcessGroupGloo` to initialize the `ProcessGroup`. If no other errors occur, you can safely ignore this warning.", UserWarning, stacklevel=2
                    )
            else:
                warnings.warn(w.message, w.category, stacklevel=2)
    
    @staticmethod
    def rpc_get_impl(rank: int, context_id: str):
        obj, consumers = TorchRpcUtils.buffer[context_id]
        consumers: set
        consumers.remove(rank)
        if not consumers:
            del TorchRpcUtils.buffer[context_id]  # Clean up if no consumers left
        return obj

    @staticmethod
    def run_impl(rref: rpc.RRef, func_name: str, *args, **kwargs):
        return getattr(rref.local_value(), func_name)(*args, **kwargs)

    @staticmethod
    def rpc_get(rank: int, context_id: str) -> rpc.RRef:
        rref = rpc.remote(TorchRpcUtils.rank_0_worker_name, TorchRpcUtils.rpc_get_impl, args = (rank, context_id), timeout = 30)
        return rref
    @staticmethod
    def rpc_set(rank: int, context_id: str, obj: object):
        TorchRpcUtils.buffer[context_id] = (obj, {i for i in range(dist.get_world_size()) if i != rank})
    
    @staticmethod
    def remote_attr_impl(rref: rpc.RRef, name: str):
        return getattr(rref.local_value(), name)
    
    @staticmethod
    def remote_attr(rref: rpc.RRef, name: str):
        """
        A remote attribute access to the object referenced by the RRef.
        """
        return rpc.remote(rref.owner(), TorchRpcUtils.remote_attr_impl, args=(rref, name), timeout=30)

    @staticmethod
    def rref_wait(rref: rpc.RRef):
        rpc.rpc_sync(rref.owner(), hasattr, (rref, "__name__"))

    @staticmethod
    def remote_call_sync(rref: rpc.RRef, func_name: str, *args, **kwargs):
        """
        A remote function call to the object referenced by the RRef. Return Results.
        """
        response = rpc.remote(rref.owner(), TorchRpcUtils.run_impl, args=(rref, func_name, *args, *kwargs), timeout=30)
        TorchRpcUtils.rref_wait(response)
        return response

    @staticmethod
    def remote_call(rref: rpc.RRef,func_name:str,  *args, **kwargs):
        """
        A remote function call to the object referenced by the RRef. Return RRef.
        """
        return rpc.remote(rref.owner(), TorchRpcUtils.run_impl, args=(rref, func_name, *args, *kwargs), timeout=30)

    @staticmethod
    def isinstance_rref(obj: object):
        return isinstance(obj, (rpc.RRef, torch._C._distributed_rpc.PyRRef))

    @staticmethod
    def make_context_id(extend = "", max_depth=5) -> str:
        stack = inspect.stack()
        call_chain = [extend]
        for frame in stack[:max_depth]:
            function_name = frame.function
            call_chain.append(function_name)

        call_chain =  "<-".join(call_chain)
        hash_object = hashlib.new("sha256")
        hash_object.update(call_chain.encode("utf-8"))
        context_id = hash_object.hexdigest()

        return context_id

class Unique(TorchRpcUtils):
    _obj: rpc.RRef
    is_distributed: bool
    """
    A function to ensure that the object is unique across distributed ranks.  
    It initializes the object on rank 0 and retrieves it on other ranks.  
    The context_depth parameter controls how many frames to consider for generating a unique context ID.
    
    Args:
        obj (object): The object to be made unique.
        context_depth (int): The depth of the call stack to consider for generating a unique context ID.
    
    Returns:
        object: The unique object across distributed ranks.
    
    Raises:
        ValueError: If the input object is None.
        RuntimeError: If the object could not be initialized or retrieved successfully.
    """
    def __init__(self,obj: object, context_depth=5, dont_distribute: bool = False, sync: bool = True):
        self.context_id = self.make_context_id(obj.__class__.__name__, context_depth)
        self.is_distributed = False
        self.sync = sync
        self._obj = obj

        if dont_distribute:
            self.is_distributed = True
            return
        
        if obj is None:
            raise ValueError("Unique obj counld not be None.")

        if not dist.is_initialized():
            return

        if isinstance(obj, (rpc.RRef, torch._C._distributed_rpc.PyRRef)): # New a Unique with a RRef
            self.is_distributed = True
            return
        
        if not self.is_distributed:
            self.rpc_distribute()
            self.is_distributed = True

    def rpc_distribute(self):
        """
        Distributes the object across distributed ranks, ensuring that it is unique.
        """
        if not dist.is_initialized():
            warnings.warn("Use Unique without distributed context. This will not ensure uniqueness across ranks.")
            return
        if self._obj is None:
            raise ValueError("None object cannot be distributed through rpc.")
        
        rank = dist.get_rank()

        if rank != 0:
            self._obj = None

        if not self.is_rpc_initialized():
            self.rpc_init(rank)

        # Step0: rank0 prepares resources
        if rank == 0:
            self.rpc_set(rank, self.context_id, self._obj)
        self.barrier()

        # Step1: rank# gets resources
        if rank != 0:
            self._obj = self.rpc_get(rank, self.context_id)

        if self._obj is None:
            raise RuntimeError(f"Failed to initialize or retrieve the object for rank {rank}. Context ID: {self.context_id}")
        self.barrier()

        self.is_distributed = True

    def close(self):
        if self.is_rpc_initialized():
            self.rpc_close()

    def wait(self):
        if self.isinstance_rref(self._obj):
            self.rref_wait(self._obj)

    def to_here(self):
        if self.isinstance_rref(self._obj):
            if self.sync:
                self.rref_wait(self._obj)
            return self._obj.to_here(timeout=30)
        return self._obj

    def __call__(self, *args, **kwargs):
        if not dist.is_initialized():
            return Unique(self._obj(*args, **kwargs), dont_distribute=True, sync = self.sync)

        if not self.is_distributed:
            self.rpc_distribute()

        if self.isinstance_rref(self._obj):
            if self.sync:
                return Unique(self.remote_call_sync(self._obj, "__call__", *args, **kwargs),dont_distribute=True, sync=self.sync)
            else:
                return Unique(self.remote_call(self._obj, "__call__", *args, **kwargs),dont_distribute=True, sync=self.sync)
        return Unique(self._obj(*args, **kwargs), dont_distribute=True, sync=self.sync)

    def __getattr__(self, name: str):
        if not dist.is_initialized():
            return Unique(getattr(self._obj, name), dont_distribute=True)
        
        if not self.is_distributed:
            self.rpc_distribute()

        if self.isinstance_rref(self._obj):
            response = self.remote_attr(self._obj, name)
            if self.sync:
                self.rref_wait(response)
            return Unique(response, dont_distribute=True, sync = self.sync)
        
        return Unique(getattr(self._obj, name), dont_distribute=True, sync = self.sync)