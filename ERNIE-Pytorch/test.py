#!/usr/bin/env python
# encoding: utf-8
"""
File Description: 
Author: liushu
Mail: 1554987494@qq.com
Created Time: 2022/8/17
"""
import paddle
import torch
import transformers
import paddlenlp
import argparse


def cloze_check():
    """
    compare cloze test
    :return:
    """
    tokenizer = transformers.BertTokenizer.from_pretrained('./convert')
    model = transformers.ErnieForMaskedLM.from_pretrained('./convert')
    input_ids = torch.tensor([tokenizer.encode(text="[MASK][MASK][MASK]是中国神魔小说的经典之作，与《三国演义》《水浒传》《红楼梦》并称为中国古典四大名著。",
                                               add_special_tokens=True)])
    model.eval()
    with torch.no_grad():
        predictions = model(input_ids)[0][0]
    predicted_index = [torch.argmax(predictions[i]).item() for i in range(predictions.shape[0])]
    predicted_token = [tokenizer._convert_id_to_token(predicted_index[i]) for i in
                       range(1, (predictions.shape[0] - 1))]
    print('huggingface result')
    print('predict result:\t', predicted_token)
    print('[CLS] logit:\t', predictions[0].numpy())
    tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained("ernie-1.0-base-zh")
    model = paddlenlp.transformers.ErnieForMaskedLM.from_pretrained("ernie-1.0-base-zh")
    inputs = tokenizer("[MASK][MASK][MASK]是中国神魔小说的经典之作，与《三国演义》《水浒传》《红楼梦》并称为中国古典四大名著。")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    model.eval()
    with paddle.no_grad():
        predictions = model(**inputs)[0]
    predicted_index = [paddle.argmax(predictions[i]).item() for i in range(predictions.shape[0])]
    predicted_token = [tokenizer._convert_id_to_token(predicted_index[i]) for i in
                       range(1, (predictions.shape[0] - 1))]
    print('paddle result')
    print('predict result:\t', predicted_token)
    print('[CLS] logit:\t', predictions[0].numpy())


def logit_check():
    """
    compare bert logit
    :return:
    """
    tokenizer = transformers.BertTokenizer.from_pretrained('./convert')
    model = transformers.ErnieModel.from_pretrained('./convert')
    input_ids = torch.tensor([tokenizer.encode(text="welcome to ernie pytorch project", add_special_tokens=True)])
    model.eval()
    with torch.no_grad():
        pooled_output = model(input_ids)[0]
    print('huggingface result')
    print('pool output:', pooled_output.numpy())
    tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-1.0-base-zh")
    model = paddlenlp.transformers.AutoModel.from_pretrained("ernie-1.0-base-zh")
    inputs = tokenizer("welcome to ernie pytorch project")
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    model.eval()
    with paddle.no_grad():
        pooled_output = model(**inputs)
    print('paddle result')
    print('pool output:', pooled_output[0].numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['cloze_check', 'logit_check'], default='cloze_check')
    args = parser.parse_args()
    if args.task == 'cloze_check':
        cloze_check()
    else:
        logit_check()
