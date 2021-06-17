import graphsurgeon as gs
import tensorflow as tf

Input = gs.create_node("conv2d_input",
    op="Placeholder",
    dtype=tf.float32,
    shape=[-1, 1, 28, 28])

isrlu_activation = gs.create_node(name="activation_1", op="ISRLU",trt_plugin=True, dtype=tf.float32)

namespace_plugin_map = { "activation_1": ISRLU}

def preprocess(dynamic_graph):
    # Now create a new graph by collapsing namespaces
    dynamic_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node
    dynamic_graph.remove(dynamic_graph.graph_outputs, remove_exclusive_dependencies=False)
