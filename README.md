# Ten Language

Ten is a statically typed tensor programming language for defining AI models.

Ten has the following features:
* Succint syntax and operators tailored to AI model definition
* Fully statically typed tensors, including generic functions over tensor dimension and batch dimensions (...)
* First-class hyper-parameters, model parameters and model arguments for explicit model specification
* EinOps-style reshaping and reductions - tensor dimensions are explicit not implicit

Future:
* Compilation directly to ONNX graphs for efficient inference execution
* Support for training (via ONNX Runtime Training)

Example (GPT2 implementation inspired by [PicoGPT](https://github.com/jaymody/picoGPT) in 36 lines):

```ten
Gelu(x: {...}) -> {...}:
    return 0.5 * x * (1 + Tanh(0.7978845608 * x + 0.044715 * x**3))

SoftMax[N](x: {...,N}) -> {...,N}:
    exp_x = Exp(x - Max(x))
    return exp_x / Sum(exp_x)

LayerNorm[S,E]|g:{E},b:{E}|(x:{S,E}) -> {S,E}:
    mean = Mean(x)
    variance = Var(x)
    return g * (x - mean) / Sqrt(variance + 1e-5) + b

Linear[N,K]|w:{N,K},b:{K}|(x:{...,N}) -> {...K}:
    return x@w + b

FFN[S,E]|c_fc, c_proj|(x:{S,E}) -> {S,E}:
    a = Gelu(Linear[E,E*4]|c_fc|(x))
    return Linear[E*4,E]|c_proj|(a)

Attention[Q,K,N,V](q:{...,Q,K}, k:{...,N,K}, v:{...,N,V}, mask:{Q,N}) -> {...,Q,V}:
    return Softmax[N](q @ Transpose[N,K](k) / Sqrt(K) + mask) @ v

MHA[H,S,E,K]|c_attn, c_proj|(x:{S,E}) -> {S,E}:
    q, k, v = Linear[E,E*3]|c_attn|(x) {S,(3,H,K) -> 3,H,S,K}
    causal_mask = (Tri[S]() - 1) * 1e10
    out = Attention[S,K,S,K](q, k, v, causal_mask) {H,S,K -> S,(H,K)}   
    return Linear[E,E]|c_proj|(out)

Transformer[H,S,E]|mlp, attn, ln_1, ln_2|(x:{S,E}) -> {S, E}:
    y = x + MHA[H,S,E,E/H]|attn|(LayerNorm[S,E]|ln_1|(x))
    return y + FFN[S,E]|mlp|(LayerNorm[S,E]|ln_2|(y))

GPT2[H,S,E,B,V]|wte, wpe, blocks|(inputs:{S}) -> {S,V}:
    x = wte.[inputs] + wpe.[Range[S]()]
    z = for i in 0...B: x, y -> Transformer[H,S,E]|blocks.[i]|(y)
    return LayerNorm[S,E]|ln_f|(z) @ Transpose[V,E](wte)
```

Running `GPT2[12,10,768,12,50257]|weights from paper|([464, 1266, 8300, 3303, 329, 16215, 9552, 4981, 318])` using the trained params loaded from the GPT2 124M model from the GPT2 paper, and passing in the encoded form of "The best programming language for defining AI models is", returns a result `ret` for which `argmax(ret[-1])` indicates that the most likely next token is `11361`, the encoded form of " Python" :-).

## Implementation Status

The current implementation type-checks and compiles, then interprets the Ten program using numpy.  This is obviously innefficient, but very flexible.  It's also largely incompatible with supporting training, which will require a higher-level execution environment.

The goal is to replace this implementation with compilation directly into an ONNX graph, and then run that graph for interpretation (currently inference and in the future perhaps training).

## Grammar

```peg
Program         <- Function*
Function        <- Ident ('[' IdentList? ']')? ('|' ParamListOptType? '|')? '(' ParamList? ')' '->' Type (':' Statement+)?
IdentList       <- Ident (',' Ident)*
ParamList       <- Param (',' Param)*
Param           <- Ident ':' Type
ParamListOptType<- ParamOptType (',' ParamOptType)*
ParamOptType    <- Ident (':' Type)?
Type            <- TensorType
TensorType      <- '{' (Dimension (',' Dimension)*)? '}'
Dimension       <- Ident / '...' / Number
Statement       <- ReturnStatement / LetStatement
ReturnStatement <- 'return' Expr
LetStatement    <- IdentList '=' Expr
Expr            <- MaybeSum
MaybeSum        <- MaybeProduct (('+' / '-') MaybeProduct)*
MaybeProduct    <- MaybePower (('*' / '/') MaybePower)*
MaybePower      <- MaybeMatmul ('**' MaybeMatmul)?
MaybeMatmul     <- MaybeReshape ('@' MaybeReshape)*
MaybeReshape    <- PrimitiveExpr ('{' ReshapeType '->' ReshapeType '}')?
PrimitiveExpr   <- ParenExpr / CallExpr / IndexExpr / ForExpr / Ident / Number 
ParenExpr       <- '(' Expr ')'
CallExpr        <- Ident ('[' ArgList? ']')? ('|' ArgList? '|')? '(' ArgList? ')'
ArgList         <- Expr (',' Expr)*
IndexExpr       <- Ident '.' '[' Expr ']'
ForExpr         <- 'for' Ident 'in' Expr '...' Expr ':' Expr ',' Ident '->' Expr
ReshapeType     <- (ReshapeDimension (',' ReshapeDimension)*)?
ReshapeDimension<- '(' ReshapeType ')' / Ident / Number 
Ident           <- [A-Za-z][A-Za-z_0-9]*
Number          <- '-'? [0-9]+ ('.' [0-9]+) ('e' '-'? [0-9]+)?
```

## Notes

Design questions:
* Could/should all parameters live inside the body?
* Should parameters have (optional) initializers for training initialization?
* Where does the loss function and optimizer definition live?
* Reshaping can be a bit more different than einops, since types are statically known
* Inference of hyper-params?
* How to best select axis for reductions? (Postfix with an index reduction { IJ -> I}?)
  * We have chosen to always operate on the last dimension, which is actually reasonable as it allows a reshape to be applied to collect appropriate dimensions prior to the reduction
* Can we make broadcasting more implicit in the type system (instead of requiring ... prefix?)

## Running CLI and Tests

Run tests:

```shell
$ python3 -m unittest test.test_parser
$ python3 -m unittest test.test_onnx # Not fully functional
```

Run the CLI (note - this isn't fully functional yet):

```shell
$ python3 . ./examples/gpt2.ten
```