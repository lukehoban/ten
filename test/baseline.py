# From https://github.com/jaymody/picoGPT

import os
import requests
import numpy as np
from tqdm import tqdm


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(
        variance + eps
    )  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(
    q, k, v, mask
):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(
        map(lambda x: np.split(x, n_head, axis=-1), qkv)
    )  # [3, n_seq, n_embd] -> [n_head, 3, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0])) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [
        attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)
    ]  # [n_head, 3, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head] # type: ignore

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(
    x, mlp, attn, ln_1, ln_2, n_head
):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(
        layer_norm(x, **ln_1), **attn, n_head=n_head
    )  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # print(inputs, wte.shape, wpe.shape)
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    # print(x.shape)

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(
            x, **block, n_head=n_head
        )  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    # print(x.shape)
    ret = x @ wte.T
    # print(ret.shape)

    return ret  # [n_seq, n_embd] -> [n_seq, n_vocab]


import tensorflow as tf
import re


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams):

    download_gpt2_model()

    def set_in_nested_dict(d, keys, val):
        if not keys:
            return val
        if keys[0] not in d:
            d[keys[0]] = {}
        d[keys[0]] = set_in_nested_dict(d[keys[0]], keys[1:], val)
        return d

    init_vars = tf.train.list_variables(tf_ckpt_path)
    # print(init_vars)
    # for name, dims in init_vars:
    #     print(f"{name}: {dims} - {np.prod(dims)}")
    params = {"blocks": [{} for _ in range(hparams["n_layer"])]}
    for name, _ in init_vars:
        array = np.squeeze(tf.train.load_variable(tf_ckpt_path, name))
        # print(f"{name} = {array}")
        name = name.removeprefix("model/")
        if name.startswith("h"):
            m = re.match(r"h([0-9]+)/(.*)", name)
            if m is None:
                raise Exception("invalid name:" + name)
            n = int(m[1])
            sub_name = m[2]
            set_in_nested_dict(params["blocks"][n], sub_name.split("/"), array)
        else:
            set_in_nested_dict(params, name.split("/"), array)

    return params


def download_gpt2_model():
    subdir = os.path.join("test", "model", "gpt2")
    if os.path.exists(subdir):
        return
    os.makedirs(subdir)
    subdir = subdir.replace("\\", "/")  # needed for Windows

    for filename in [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]:
        r = requests.get(
            "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/"+ filename,
            stream=True,
        )

        with open(os.path.join(subdir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(
                ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True
            ) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
