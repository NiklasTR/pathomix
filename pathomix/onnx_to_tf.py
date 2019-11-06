import tensorflow as tf

with tf.gfile.FastGFile('resnext50.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def, name="")

sess = tf.Session(graph=g_in)

