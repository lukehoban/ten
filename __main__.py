import torch
import onnx
from onnx import helper, shape_inference
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Optional

from src import parse

def make_graph() -> onnx.GraphProto:
    X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [10])
    Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [10])
    node = helper.make_node("Softmax", ["X"], ["Y"])
    return helper.make_graph(nodes=[node], name="test", inputs=[X], outputs=[Y])


def main() -> None:
    graph = make_graph()
    model = helper.make_model(graph, producer_name="tensorlang")
    model.opset_import[0].version = 13

    print("MODEL BEFORE INFERENCE")
    print(model.graph)

    model = shape_inference.infer_shapes(model)

    print("MODEL AFTER INFERENCE")
    print(model.graph)

    ort_sess = ort.InferenceSession(model.SerializeToString())
    outputs = ort_sess.run(None, {"X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    print(outputs)
    
    parse.Parser


if __name__ == "__main__":
    main()
