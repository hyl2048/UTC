
from transformers import BertTokenizer

input_ids = [1    , 39980, 1059 , 49   , 21   , 227  , 39980, 1815 , 423  , 751  ,
         85   , 39980, 300  , 97   , 625  , 66   , 39980, 1046 , 1815 , 423  ,
         453  , 39980, 59   , 16   , 70   , 66   , 39980, 1059 , 49   , 59   ,
         529  , 39980, 177  , 218  , 200  , 1005 , 39980, 588  , 136  , 1046 ,
         453  , 39980, 97   , 221  , 417  , 1059 , 39980, 300  , 97   , 1436 ,
         112  , 39980, 1059 , 152  , 625  , 66   , 39980, 72   , 91   , 417  ,
         1059 , 39980, 50   , 911  , 393  , 112  , 39980, 465  , 1059 , 21   ,
         85   , 39980, 582  , 153  , 836  , 523  , 39980, 154  , 367  , 443  ,
         454  , 39980, 987  , 755  , 2062 , 1494 , 39980, 59   , 529  , 240  ,
         17   , 39980, 85   , 291  , 59   , 88   , 39980, 27   , 8    , 625  ,
         66   , 2    , 1648 , 17   , 884  , 136  , 1660 , 139  , 250  , 6    ,
         171  , 612  , 417  , 1059 , 36   , 67   , 929  , 302  , 33   , 443  ,
         454  , 4    , 1059 , 21   , 85   , 501  , 908  , 1630 , 190  , 171  ,
         612  , 690  , 908  , 908  , 1815 , 423  , 4    , 89   , 245  , 22   ,
         78   , 1648 , 77   , 707  , 3160 , 72   , 119  , 590  , 436  , 131  ,
         2783 , 10043, 9485 , 500  , 119  , 104  , 290  , 273  , 146  , 51   ,
         310  , 443  , 454  , 103  , 390  , 840  , 22   , 15   , 524  , 418  ,
         4    , 310  , 443  , 454  , 248  , 9    , 72   , 646  , 344  , 90   ,
         4    , 51   , 250  , 6    , 171  , 612  , 653  , 58   , 428  , 9    ,
         450  , 1196 , 90   , 12043, 2    , 2    ]
tokenizer = BertTokenizer.from_pretrained("utc_pytorch/models/utc_base_pytorch")
res = [tokenizer._convert_id_to_token(item) for item in input_ids]
print(res)
# import torch
# from model import UTC
# from template import UTCTemplate
# from torch.hub import load_state_dict_from_url
# from transformers import AutoTokenizer, BertConfig, BertTokenizer
# from utils import DataCollatorWithPadding

# paddle.disable_signal_handler()
# # 构建输入
# input_ids = np.random.rand(8, 200).astype("float32")
# token_type_ids = np.random.rand(8, 200).astype("float32")
# position_ids = np.random.rand(8, 200).astype("float32")
# attention_mask = np.random.rand(8, 200, 200).astype("float32")
# omask_positions = np.random.rand(8, 200).astype("float32")
# cls_positions = np.random.rand(8, 200).astype("float32")
# input_dict = {
#     "input_ids": torch.tensor(input_ids),
#     "token_type_ids": torch.tensor(token_type_ids),
#     "position_ids": torch.tensor(position_ids),
#     "attention_mask": torch.tensor(attention_mask),
#     "omask_positions": torch.tensor(omask_positions),
#     "cls_positions": torch.tensor(cls_positions),
# }
# input_list = [
#     torch.tensor(input_ids),
#     torch.tensor(token_type_ids),
#     torch.tensor(position_ids),
#     torch.tensor(attention_mask),
#     torch.tensor(omask_positions),
#     torch.tensor(cls_positions),
# ]

# tokenizer = BertTokenizer.from_pretrained(
#     "/root/UTC/utc_pytorch/models/convert/utc_base"
# )
# example = {
#     "text_a": "综上，原告现要求变更女儿李乙抚养关系的请求，本院应予支持。",
#     "text_b": "",
#     "question": "",
#     "choices": [
#         "婚后生育",
#         "抚养孩子",
#         "共同财产",
#         "付抚养费",
#         "分不动产",
#         "婚后分居",
#         "二次起诉",
#         "按月付费",
#         "同意离婚",
#         "共同债务",
#         "婚前财产",
#         "法定离婚",
#         "家庭义务",
#         "非婚生子",
#         "适当帮助",
#         "无视协议",
#         "损害赔偿",
#         "分居两年",
#         "子女分开",
#         "个人财产",
#     ],
#     "labels": [0, 1],
# }
# collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
# utc_template = UTCTemplate(tokenizer, max_length=512)

# input_utc_template = utc_template(example)
# input_content = collator([input_utc_template])
# input_content_format = [v for k, v in input_content.items()]
# input_content_format[2] = input_content_format[2].int()
# # 获取PyTorch Module
# config_dir = "/root/UTC/utc_pytorch/models/convert/utc_base/config.json"
# model_path = "/root/UTC/utc_pytorch/models/convert/utc_base"
# config = BertConfig.from_json_file(config_dir)
# torch_module = UTC.from_pretrained(model_path, config=config)
# # torch_state_dict = torch.load("models/convert/utc-base")
# # torch_module.load_state_dict(torch_state_dict)
# # 设置为eval模式
# torch_module.eval()
# # 进行转换
# from x2paddle.convert import pytorch2paddle

# pytorch2paddle(
#     torch_module,
#     save_dir="pd_model_trace",
#     jit_type="trace",
#     input_examples=input_content_format,
# )
