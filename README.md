# Ten Language

Ten is a programming language for defining AI models.

Ten has the following features:
* Succint syntax and operators tailored to AI model definition
* Fully statically typed tensors
* First-class hyper-parameters, model parameters and model arguments
* Compilation directly to ONNX graphs for efficient inference execution
* EinOps-style reshaping and reductions - tensor dimensions are explicit not implicit

Future:
* Support for training (via ONNX Runtime Training)

Example (GPT2 implementation inspired by [PicoGPT](https://github.com/jaymody/picoGPT)):

```ten
Gelu(x: {...}) -> {...}:
    return 0.5 * x * (1 + Tanh(0.7978845608 * x + 0.044715 * x**3))

SoftMax[N](x: {...,N}) -> {...,N}:
    let exp_x = Exp(x - Max(x))
    return exp_x / Sum(exp_x)

LayerNorm[S,E]|g:{E},b:{E}|(x: {S,E}) -> {S,E}:
    let mean = Mean(x)
    let variance = Var(x)
    return g * (x - mean) / Sqrt(variance + 1e-5) + b

Linear[N,K]|w:{N,K},b:{K}|(x: {...N}) -> {...K}:
    return @{...K}(x{...N}, w{N,K}) + b

FFN[S,E]|c_fc, c_proj|(x:{S,E}) -> {S,E}:
    let a = Gelu(Linear[E,E*4]|c_fc...|(x))
    return Linear[E*4,E]|c_proj...|(a)

Attention[Q, K, N, V](q:{...,Q,K}, k:{...,N,K}, v:{...,N,V}, mask:{Q,N}) -> {...,Q,V}:
    return @(Softmax[N]((@(q, Transpose(k)) / Sqrt(K)) + mask), v)

MHA[H,S,E,K=E/H]|c_attn, c_proj|(x:{S,E}) -> {S,E}:
    let q, k, v = Linear[E, E*3]|c_attn|(x) {S,(3,H,K) -> 3,H,S,K}
    let causal_mask = (1 - Tri[S]()) * -1e10
    let out = Attention[S,K,S,K](q, k, v, causal_mask) {H,S,K -> S,(H,K)}   
    return Linear[E,E]|c_proj|(out)

Transformer[H,S,E]|mlp, attn, ln_1, ln_2|(x: {S,E}) -> {S, E}:
    let y = x + MHA[H,S,E]|attn|(LayerNorm[S,E]|ln_1|(x))
    return y + FFN[S,E]|mlp|(LayerNorm[S,E]|ln_2|(y))

GPT2[H,S,E,B,V]|wte, wpe, blocks|(inputs: {S}) -> {S,V}:
    let x = wte[inputs] + wpe[Range[S]()]
    let z = for i in 0..B: x, y => Transformer[H,S,E]|blocks[i]|(y)
    return @(LayerNorm[S,E]|ln_f|(z), Transpose[V,E](wte))
```


Notes:
* Could/should all parameters live inside the body?
* Should parameters have (optional) initializers for training initialization?
* Where does the loss function and optimizer definition live?
* Reshaping can be a bit more different than einops, since types are statically known
* Inference of hyper-params?
* How to best select axis for reductions? (Postfix with an index reduction { IJ -> I}?)
