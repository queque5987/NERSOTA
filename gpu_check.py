from torch import cuda
import torch
print(torch.__version__)
print(torch.version.cuda)
print(cuda.is_available())
print(cuda.device_count())
print(cuda.get_device_name(cuda.current_device()))