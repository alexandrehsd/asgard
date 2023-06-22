import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub


def build_model(encoder_handler, preprocess_handler, n_outputs, optimizer, loss, metrics, dropout):
    # Define bert model and preprocessing
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(preprocess_handler, name="preprocessing")

    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(encoder_handler, trainable=True, name="BERT_encoder")
    outputs = encoder(encoder_inputs)

    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(n_outputs, activation=None, name="classifier")(net)
    model = tf.keras.Model(text_input, net)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
