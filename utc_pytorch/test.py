import numpy as np

# input_ids = [    1, 39980,  1059,    49,    21,   227, 39980,  1815,   423,   751,
#            85, 39980,   300,    97,   625,    66, 39980,  1046,  1815,   423,
#           453, 39980,    59,    16,    70,    66, 39980,  1059,    49,    59,
#           529, 39980,   177,   218,   200,  1005, 39980,   588,   136,  1046,
#           453, 39980,    97,   221,   417,  1059, 39980,   300,    97,  1436,
#           112, 39980,  1059,   152,   625,    66, 39980,    72,    91,   417,
#          1059, 39980,    50,   911,   393,   112, 39980,   465,  1059,    21,
#            85, 39980,   582,   153,   836,   523, 39980,   154,   367,   443,
#           454, 39980,   987,   755,  2062,  1494, 39980,    59,   529,   240,
#            17, 39980,    85,   291,    59,    88, 39980,    27,     8,   625,
#            66,     2,  1648,    17,   884,   136,  1660,   139,   250,     6,
#           171,   612,   417,  1059,    36,    67,   929,   302,    33,   443,
#           454,     4,  1059,    21,    85,   501,   908,  1630,   190,   171,
#           612,   690,   908,   908,  1815,   423,     4,    89,   245,    22,
#            78,  1648,    77,   707,  3160,    72,   119,   590,   436,   131,
#          2783, 10043,  9485,   500,   119,   104,   290,   273,   146,    51,
#           310,   443,   454,   103,   390,   840,    22,    15,   524,   418,
#             4,   310,   443,   454,   248,     9,    72,   646,   344,    90,
#             4,    51,   250,     6,   171,   612,   653,    58,   428,     9,
#           450,  1196,    90, 12043,     2,     2,     0,     0,     0,     0]
# tokenizer = BertTokenizer.from_pretrained("models/utc_base_pytorch")
# res = [tokenizer._convert_id_to_token(item) for item in input_ids]
# print(res)
import torch
from model import UTC
from torch.hub import load_state_dict_from_url
from transformers import AutoTokenizer, BertTokenizer

# 构建输入
input_data = np.random.rand(1, 3, 224, 224).astype("float32")
# 获取PyTorch Module
torch_module = UTC()
torch_state_dict = torch.load("models/utc-base")
torch_module.load_state_dict(torch_state_dict)
# 设置为eval模式
torch_module.eval()
# 进行转换
from x2paddle.convert import pytorch2paddle

pytorch2paddle(
    torch_module,
    save_dir="pd_model_trace",
    jit_type="trace",
    input_examples=[torch.tensor(input_data)],
)
