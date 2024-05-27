#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
ernie3.0 series model conversion based on paddlenlp repository
ernie2.0 series model conversion based on paddlenlp repository
official repo: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo
Author: nghuyong liushu
Mail: nghuyong@163.com 1554987494@qq.com
Created Time: 2022/8/17
"""
import collections
import json
import os

import numpy as np
import paddle
import torch
from template import UTCTemplate
from utils import DataCollatorWithPadding, logger


def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict(
        {
            "ernie.embeddings.word_embeddings.weight": "ernie.embeddings.word_embeddings.weight",
            "ernie.embeddings.position_embeddings.weight": "ernie.embeddings.position_embeddings.weight",
            "ernie.embeddings.token_type_embeddings.weight": "ernie.embeddings.token_type_embeddings.weight",
            "ernie.embeddings.task_type_embeddings.weight": "ernie.embeddings.task_type_embeddings.weight",
            "ernie.embeddings.layer_norm.weight": "ernie.embeddings.LayerNorm.gamma",
            "ernie.embeddings.layer_norm.bias": "ernie.embeddings.LayerNorm.beta",
        }
    )
    # add attention layers
    for i in range(attention_num):
        weight_map[f"ernie.encoder.layers.{i}.self_attn.q_proj.weight"] = (
            f"ernie.encoder.layer.{i}.attention.self.query.weight"
        )
        weight_map[f"ernie.encoder.layers.{i}.self_attn.q_proj.bias"] = (
            f"ernie.encoder.layer.{i}.attention.self.query.bias"
        )
        weight_map[f"ernie.encoder.layers.{i}.self_attn.k_proj.weight"] = (
            f"ernie.encoder.layer.{i}.attention.self.key.weight"
        )
        weight_map[f"ernie.encoder.layers.{i}.self_attn.k_proj.bias"] = (
            f"ernie.encoder.layer.{i}.attention.self.key.bias"
        )
        weight_map[f"ernie.encoder.layers.{i}.self_attn.v_proj.weight"] = (
            f"ernie.encoder.layer.{i}.attention.self.value.weight"
        )
        weight_map[f"ernie.encoder.layers.{i}.self_attn.v_proj.bias"] = (
            f"ernie.encoder.layer.{i}.attention.self.value.bias"
        )
        weight_map[f"ernie.encoder.layers.{i}.self_attn.out_proj.weight"] = (
            f"ernie.encoder.layer.{i}.attention.output.dense.weight"
        )
        weight_map[f"ernie.encoder.layers.{i}.self_attn.out_proj.bias"] = (
            f"ernie.encoder.layer.{i}.attention.output.dense.bias"
        )
        weight_map[f"ernie.encoder.layers.{i}.norm1.weight"] = (
            f"ernie.encoder.layer.{i}.attention.output.LayerNorm.gamma"
        )
        weight_map[f"ernie.encoder.layers.{i}.norm1.bias"] = (
            f"ernie.encoder.layer.{i}.attention.output.LayerNorm.beta"
        )
        weight_map[f"ernie.encoder.layers.{i}.linear1.weight"] = (
            f"ernie.encoder.layer.{i}.intermediate.dense.weight"
        )
        weight_map[f"ernie.encoder.layers.{i}.linear1.bias"] = (
            f"ernie.encoder.layer.{i}.intermediate.dense.bias"
        )
        weight_map[f"ernie.encoder.layers.{i}.linear2.weight"] = (
            f"ernie.encoder.layer.{i}.output.dense.weight"
        )
        weight_map[f"ernie.encoder.layers.{i}.linear2.bias"] = (
            f"ernie.encoder.layer.{i}.output.dense.bias"
        )
        weight_map[f"ernie.encoder.layers.{i}.norm2.weight"] = (
            f"ernie.encoder.layer.{i}.output.LayerNorm.gamma"
        )
        weight_map[f"ernie.encoder.layers.{i}.norm2.bias"] = (
            f"ernie.encoder.layer.{i}.output.LayerNorm.beta"
        )
    # add pooler
    weight_map["ernie.pooler.dense.weight"] = "ernie.pooler.dense.weight"
    weight_map["ernie.pooler.dense.bias"] = "ernie.pooler.dense.bias"
    weight_map["linear_q.weight"] = "linear_q.weight"
    weight_map["linear_q.bias"] = "linear_q.bias"
    weight_map["linear_k.weight"] = "linear_k.weight"
    weight_map["linear_k.bias"] = "linear_k.bias"
    return weight_map


def extract_and_convert(input_dir, output_dir):
    """
    抽取并转换
    :param input_dir:
    :param output_dir:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("=" * 20 + "save config file" + "=" * 20)
    config = json.load(
        open(os.path.join(input_dir, "config.json"), "rt", encoding="utf-8")
    )
    # if 'init_args' in config:
    #     config = config['init_args'][0]
    # del config['init_class']
    # config['layer_norm_eps'] = 1e-12
    # config['model_type'] = 'ernie'
    # config['architectures'] = ["ErnieForMaskedLM"]  # or 'BertModel'
    # config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(
        config,
        open(os.path.join(output_dir, "config.json"), "wt", encoding="utf-8"),
        indent=4,
    )
    print("=" * 20 + "save vocab file" + "=" * 20)
    with open(os.path.join(input_dir, "vocab.txt"), "rt", encoding="utf-8") as f:
        words = f.read().splitlines()
    words = [word.split("\t")[0] for word in words]
    with open(os.path.join(output_dir, "vocab.txt"), "wt", encoding="utf-8") as f:
        for word in words:
            f.write(word + "\n")
    print("=" * 20 + "extract weights" + "=" * 20)
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config["num_hidden_layers"])
    paddle_paddle_params = paddle.load(os.path.join(input_dir, "model_state.pdparams"))
    for weight_name, weight_value in paddle_paddle_params.items():
        if "weight" in weight_name and "weight" in weight_map[weight_name]:
            if (
                "ernie.encoder" in weight_name
                or "ernie.pooler" in weight_name
                or "linear_q" in weight_name
                or "linear_k" in weight_name
            ):
                weight_value = weight_value.transpose([1, 0])
        if weight_name not in weight_map:
            print("=" * 20, "[SKIP]", weight_name, "=" * 20)
            continue
        state_dict[weight_map[weight_name]] = torch.from_numpy(weight_value.numpy())
        print(weight_name, "->", weight_map[weight_name], weight_value.shape)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


def convert(model, pd_model_weight_path, save_path):
    # 加载paddle参数
    paddle_key_params = paddle.load(pd_model_weight_path)

    paddle_state_dict = change_paddle_key()
    state_dict = model.state_dict()
    for key in state_dict.keys():

        if key in paddle_state_dict.keys():
            param = paddle_key_params[paddle_state_dict[key]]
            if (
                "weight" in key
                and "LayerNorm" not in key
                and "embeddings" not in key
                and "decoder" not in key
            ):
                param = param.transpose((1, 0))
            state_dict[key] = torch.from_numpy(param.numpy())
        else:
            print(key)
    model.load_state_dict(state_dict, strict=False)
    torch.save(model.state_dict(), save_path)


def validate_model(tokenizer, pt_model, pd_model, model_type="uie", atol: float = 1e-5):
    logger.info("Validating PyTorch model...")

    # batch_size = 2
    # seq_length = 6
    # seq_length_with_token = seq_length + 2
    # max_seq_length = 512
    # dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
    # encoded_inputs = dict(
    #     tokenizer(
    #         dummy_input,
    #         pad_to_max_seq_len=True,
    #         max_seq_len=512,
    #         return_attention_mask=True,
    #         return_position_ids=True,
    #     )
    # )
    example = {
        "text_a": "综上，原告现要求变更女儿李乙抚养关系的请求，本院应予支持。",
        "text_b": "",
        "question": "",
        "choices": [
            "婚后生育",
            "抚养孩子",
            "共同财产",
            "付抚养费",
            "分不动产",
            "婚后分居",
            "二次起诉",
            "按月付费",
            "同意离婚",
            "共同债务",
            "婚前财产",
            "法定离婚",
            "家庭义务",
            "非婚生子",
            "适当帮助",
            "无视协议",
            "损害赔偿",
            "分居两年",
            "子女分开",
            "个人财产",
        ],
        "labels": [0, 1],
    }
    collator_paddle = DataCollatorWithPadding(
        tokenizer=tokenizer, return_tensors="pd", return_attention_mask=True
    )

    utc_template = UTCTemplate(tokenizer, max_length=512)

    input_utc_template = utc_template(example)
    input_content = collator_paddle([input_utc_template])

    # input_content_format = [v for k, v in input_content.items()]
    # input_content_format[2] = input_content_format[2].int()

    # paddle_inputs = {}
    # for name, value in encoded_inputs.items():
    #     if name == "attention_mask":
    #         if model_type == "uie-m":
    #             continue
    #         name = "att_mask"
    #     if name == "position_ids":
    #         name = "pos_ids"
    #     paddle_inputs[name] = paddle.to_tensor(value, dtype=paddle.int64)

    paddle_named_outputs = ["option_logits"]
    paddle_outputs = pd_model(**input_content)

    torch_inputs = {}
    for name, value in input_content.items():
        if name == "attention_mask":
            if model_type == "uie-m":
                continue
        torch_inputs[name] = torch.tensor(value.numpy(), dtype=torch.int64)
    torch_outputs = pt_model(**torch_inputs)
    torch_outputs_dict = {}

    for name, value in torch_outputs.items():
        torch_outputs_dict[name] = value

    torch_outputs_set, ref_outputs_set = set(torch_outputs_dict.keys()), set(
        paddle_named_outputs
    )
    if not torch_outputs_set.issubset(ref_outputs_set):
        logger.info(
            f"\t-[x] Pytorch model output names {torch_outputs_set} do not match reference model {ref_outputs_set}"
        )

        raise ValueError(
            "Outputs doesn't match between reference model and Pytorch converted model: "
            f"{torch_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        logger.info(
            f"\t-[✓] Pytorch model output names match reference model ({torch_outputs_set})"
        )

    # Check the shape and values match
    for name, ref_value in zip(paddle_named_outputs, paddle_outputs):
        ref_value = ref_value.numpy()
        pt_value = torch_outputs_dict[name].detach().numpy().squeeze()  ## 加了 squeeze
        logger.info(f'\t- Validating PyTorch Model output "{name}":')

        # Shape
        if not pt_value.shape == ref_value.shape:
            logger.info(
                f"\t\t-[x] shape {pt_value.shape} doesn't match {ref_value.shape}"
            )
            raise ValueError(
                "Outputs shape doesn't match between reference model and Pytorch converted model: "
                f"Got {ref_value.shape} (reference) and {pt_value.shape} (PyTorch)"
            )
        else:
            logger.info(f"\t\t-[✓] {pt_value.shape} matches {ref_value.shape}")

        # Values
        if not np.allclose(ref_value, pt_value, atol=atol):
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Outputs values doesn't match between reference model and Pytorch converted model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - pt_value))}"
            )
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")


if __name__ == "__main__":
    from model import UTC
    from paddlenlp.transformers import UTC as UTCPaddle
    from paddlenlp.transformers import AutoTokenizer, ErnieTokenizer

    os.environ["OMP_NUM_THREADS"] = str(15)

    input_model = "/root/UTC/ERNIE-Pytorch/models/utc-base"
    output_model = "/root/UTC/ERNIE-Pytorch/models/convert/utc_base"
    # extract_and_convert(input_model, output_model)
    tokenizer = ErnieTokenizer.from_pretrained(
        pretrained_model_name_or_path=input_model, return_dict=False
    )
    model = UTC.from_pretrained(pretrained_model_name_or_path=output_model)
    model.eval()
    paddle_model = UTCPaddle.from_pretrained(pretrained_model_name_or_path=input_model)
    paddle_model.eval()
    model_type = "uie"
    validate_model(tokenizer, model, paddle_model, model_type)
