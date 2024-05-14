import os

import paddle
import torch

input_dir = "models/utc-base"
paddle_paddle_params = paddle.load(os.path.join(input_dir, "model_state.pdparams"))
for weight_name, weight_value in paddle_paddle_params.items():
    print(weight_name, weight_value.shape)


input_dir = "models/convert/utc_base"
paddle_paddle_params = torch.load(os.path.join(input_dir, "pytorch_model.bin"))
for weight_name, weight_value in paddle_paddle_params.items():
    print(weight_name, weight_value.shape)

input_dir = "models/utc_base_pytorch"
paddle_paddle_params = torch.load(os.path.join(input_dir, "pytorch_model.bin"))
for weight_name, weight_value in paddle_paddle_params.items():
    print(weight_name, weight_value.shape)