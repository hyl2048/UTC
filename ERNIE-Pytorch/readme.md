<h1 align="center">ERNIE-Pytorch</h1>

<p align="center">This project is to convert ERNIE from paddlepaddle to huggingface's format (in Pytorch).</p>

<p align="center">
  <a href="https://github.com/nghuyong/ERNIE-Pytorch/stargazers">
    <img src="https://img.shields.io/github/stars/nghuyong/ERNIE-Pytorch.svg?colorA=orange&colorB=orange&logo=github"
         alt="GitHub stars">
  </a>
  <a href="https://github.com/nghuyong/ERNIE-Pytorch/issues">
        <img src="https://img.shields.io/github/issues/nghuyong/ERNIE-Pytorch.svg"
             alt="GitHub issues">
  </a>
  <a href="https://github.com/nghuyong/ERNIE-Pytorch/">
        <img src="https://img.shields.io/github/last-commit/nghuyong/ERNIE-Pytorch.svg">
  </a>
   <a href="https://github.com/nghuyong/ERNIE-Pytorch/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/nghuyong/ERNIE-Pytorch.svg">
  </a>
  
</p>



**News: ERNIE has been merged
into [huggingface/transformers@v4.22.0](https://github.com/huggingface/transformers/releases/tag/v4.22.0) !!**


## Get Started

```
pip install --upgrade transformers
```

Take `ernie-1.0-base-zh` as an example:

```Python
from transformers import BertTokenizer, ErnieModel

tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
model = ErnieModel.from_pretrained("nghuyong/ernie-1.0-base-zh")
```

### Supported Models

|     Model Name      | Language |           Description           |
|:-------------------:|:--------:|:-------------------------------:|
|  ernie-1.0-base-zh  | Chinese  | Layer:12, Heads:12, Hidden:768  |
|  ernie-2.0-base-en  | English  | Layer:12, Heads:12, Hidden:768  |
| ernie-2.0-large-en  | English  | Layer:24, Heads:16, Hidden:1024 |
| ernie-3.0-xbase-zh  | Chinese  | Layer:20, Heads:16, Hidden:1024 |
|  ernie-3.0-base-zh  | Chinese  | Layer:12, Heads:12, Hidden:768  |
| ernie-3.0-medium-zh | Chinese  |  Layer:6, Heads:12, Hidden:768  |
|  ernie-3.0-mini-zh  | Chinese  |  Layer:6, Heads:12, Hidden:384  |
| ernie-3.0-micro-zh  | Chinese  |  Layer:4, Heads:12, Hidden:384  |
|  ernie-3.0-nano-zh  | Chinese  |  Layer:4, Heads:12, Hidden:312  |
|   ernie-health-zh   | Chinese  | Layer:12, Heads:12, Hidden:768  |
|    ernie-gram-zh    | Chinese  | Layer:12, Heads:12, Hidden:768  |

You can find all the supported models from huggingface's model
hub: [huggingface.co/nghuyong](https://huggingface.co/nghuyong),
and model details from paddle's official
repo: [PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html)
and [ERNIE](https://github.com/PaddlePaddle/ERNIE/blob/repro).

## Details

<details>
    <summary>I want to convert the model from paddle version by myself 😉</summary>


The following will take `ernie-1.0-base-zh` as an example to show how to convert.

1. Download the paddle-paddle version ERNIE model. Execute the following code
  ```
  import paddlenlp
  tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained("ernie-1.0-base-zh")
  model = paddlenlp.transformers.ErnieForMaskedLM.from_pretrained("ernie-1.0-base-zh")
  ```
  And then you will get the model in `~/.paddlenlp/models/ernie-1.0-base-zh/`, move to this project path.
  
2. ```pip install -r requirements.txt```
3. ```python convert.py```
4. Now, a folder named `convert` will be in the project path, and there will be three files in this
   folder: `config.json`,`pytorch_model.bin` and `vocab.txt`.

</details>

<details>
    <summary>I want to check the calculation results before and after model conversion 😁</summary>

```bash
python test.py --task logit_check
```

You will get the output:

```output
huggingface result
pool output: [-1.         -1.          0.9981035  -0.9996652  -0.78173476 -1.          -0.9994901   0.97012603  0.85954666  0.9854131 ]

paddle result
pool output: [-0.99999976 -0.99999976  0.9981028  -0.9996651  -0.7815545  -0.99999976  -0.9994898   0.97014064  0.8594844   0.985419  ]
```

It can be seen that the result of our convert version is the same with the official paddlepaddle's version.

</details>

<details>
    <summary>I want to reproduce the cloze test in ERNIE1.0's paper 😆</summary>

```bash
python test.py --task cloze_check
```

You will get the output:

```bash
huggingface result
prediction shape:	 torch.Size([47, 18000])
predict result:	 ['西', '游', '记', '是', '中', '国', '神', '魔', '小', '说', '的', '经', '典', '之', '作', '，', '与', '《', '三', '国', '演', '义', '》', '《', '水', '浒', '传', '》', '《', '红', '楼', '梦', '》', '并', '称', '为', '中', '国', '古', '典', '四', '大', '名', '著', '。']
[CLS] logit:	 [-15.693626 -19.522263 -10.429456 ... -11.800728 -12.253127 -14.375117]

paddle result
prediction shape:	 [47, 18000]
predict result:	 ['西', '游', '记', '是', '中', '国', '神', '魔', '小', '说', '的', '经', '典', '之', '作', '，', '与', '《', '三', '国', '演', '义', '》', '《', '水', '浒', '传', '》', '《', '红', '楼', '梦', '》', '并', '称', '为', '中', '国', '古', '典', '四', '大', '名', '著', '。']
[CLS] logit:	 [-15.693538 -19.521954 -10.429307 ... -11.800765 -12.253114 -14.375412]
```

</details>

## Citation

If you use this work in a scientific publication, I would appreciate that you can also cite the following BibTex entry:

```latex
@misc{nghuyong2019@ERNIE-Pytorch,
  title={ERNIEPytorch},
  author={Yong Hu},
  howpublished={\url{https://github.com/nghuyong/ERNIE-Pytorch}},
  year={2019}
}
```














