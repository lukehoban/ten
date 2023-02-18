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

def make_graph_2() -> onnx.GraphProto:
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [])
    c1 = helper.make_node("Constant", [], ["c1"], value=helper.make_tensor("cc1", onnx.TensorProto.FLOAT, [], [0.7978845608]))
    c2 = helper.make_node("Constant", [], ["c2"], value=helper.make_tensor("cc2", onnx.TensorProto.FLOAT, [], [0.044715]))
    c3 = helper.make_node("Constant", [], ["c3"], value=helper.make_tensor("cc3", onnx.TensorProto.FLOAT, [], [3]))
    c4 = helper.make_node("Constant", [], ["c4"], value=helper.make_tensor("cc4", onnx.TensorProto.FLOAT, [], [1]))
    c5 = helper.make_node("Constant", [], ["c5"], value=helper.make_tensor("cc5", onnx.TensorProto.FLOAT, [], [0.5]))
    p1 = helper.make_node("Mul", ["x", "c1"], ["p1"])
    p2 = helper.make_node("Pow", ["x", "c3"], ["p2"])
    p3 = helper.make_node("Pow", ["c2", "p2"], ["p3"])
    p4 = helper.make_node("Tanh", ["p3"], ["p4"])
    p5 = helper.make_node("Add", ["c4", "p4"], ["p5"])
    p6 = helper.make_node("Mul", ["x", "p5"], ["p6"])
    res = helper.make_node("Mul", ["c5", "p6"], ["y"])
    return helper.make_graph(nodes=[c1,c2,c3,c4,c5,p1, p2, p3,p4,p5,p6,res], name="test", inputs=[x], outputs=[y])


def main() -> None:
    graph = make_graph_2()
    model = helper.make_model(graph, producer_name="tensorlang")
    model.opset_import[0].version = 13

    print("MODEL BEFORE INFERENCE")
    print(model.graph)

    model = shape_inference.infer_shapes(model)

    print("MODEL AFTER INFERENCE")
    print(model.graph)

    ort_sess = ort.InferenceSession(model.SerializeToString())
    outputs = ort_sess.run(None, {"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    print(outputs)
    
    parse.Parser


if __name__ == "__main__":
    main()
