# Copyright 2023 Luke Hoban

import onnx
from onnx import helper as onnxmod, shape_inference
import onnxruntime as ort
import numpy as np


def make_graph_linear() -> onnx.GraphProto:
    N = 10
    K = 7
    DotDotDot = ["A", "B"]

    # Inputs and Outputs
    x = onnxmod.make_tensor_value_info("X", onnx.TensorProto.FLOAT, DotDotDot + [N])
    ret = onnxmod.make_tensor_value_info("Ret", onnx.TensorProto.FLOAT, DotDotDot + [K])

    # Constants
    wnp = np.random.randn(N, K).astype(np.float32)
    w = onnxmod.make_tensor("W", onnx.TensorProto.FLOAT, [N, K], wnp)
    bnp = np.random.randn(K).astype(np.float32)
    b = onnxmod.make_tensor("B", onnx.TensorProto.FLOAT, [K], bnp)

    # Nodes
    t1 = onnxmod.make_node("MatMul", ["X", "W"], ["T1"])
    t2 = onnxmod.make_node("Add", ["T1", "B"], ["Ret"])

    return onnxmod.make_graph(
        name="test",
        inputs=[x],
        outputs=[ret],
        initializer=[w, b],
        nodes=[t1, t2],
    )


if __name__ == "__main__":
    graph = make_graph_linear()
    model = onnxmod.make_model(graph, producer_name="tensorlang")
    model.opset_import[0].version = 13

    print("MODEL BEFORE INFERENCE")
    print(model.graph)

    model = shape_inference.infer_shapes(model)

    print("MODEL AFTER INFERENCE")
    print(model.graph)

    ort_sess = ort.InferenceSession(model.SerializeToString())
    outputs = ort_sess.run(
        None,
        {"X": [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]},
    )
    print(outputs)
