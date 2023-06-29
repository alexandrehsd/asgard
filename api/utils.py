from pydantic import BaseModel
import tensorflow as tf
import wandb
import numpy as np
from tensorflow import keras
from official.nlp import optimization
import tensorflow_hub as hub
import tensorflow_text as text
from asgard.callbacks.callbacks import EarlyStoppingHammingScore
from asgard.metrics.metrics import HammingScoreMetric
from asgard.utils.weights import get_class_weight, get_weighted_loss


class Payload(BaseModel):
    """
    Pydantic model for the payload
    """
    # def __init__(self, text, **kwargs):
        #   super().__init__(**kwargs)
    text: str


class ModelsLoaderSingleton:
    """
        Singleton class to load the models only once
    """
    __instance = None

    def __init__(self):
        
        # Define number of epochs
        epochs = 4
        steps_per_epoch = 24813 # tf.data.experimental.cardinality(train_set).numpy()

        # Define optimizer
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.05 * num_train_steps)
        
        init_lr = 1e-4
        
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_lr,
            decay_steps=num_train_steps,
            end_learning_rate=0.0,
            power=1.0
            )
        
        
        run = wandb.init()
        artifact = run.use_artifact('alexandre-hsd/ASGARD-DistilBERT/model-avid-sweep-1:v0', type='model')
        artifact_dir = artifact.download()
        
        target_folder = artifact_dir
        
        weights = np.zeros((16, 2))
        weights[:, 0] = 1.
        weights[:, 1] = 2.
        
        with tf.device("/CPU:0"):
            self.model = tf.keras.models.load_model(target_folder,
                                                    custom_objects={"weighted_loss": get_weighted_loss(weights),
                                                                    "AdamWeightDecay": optimization.create_optimizer(init_lr=init_lr,
                                                                                                                    num_train_steps=num_train_steps,
                                                                                                                    num_warmup_steps=num_warmup_steps,
                                                                                                                    optimizer_type='adamw'),
                                                                    "WarmUp": optimization.WarmUp(initial_learning_rate=init_lr,
                                                                                                decay_schedule_fn=lr_schedule,
                                                                                                warmup_steps=num_warmup_steps),
                                                                    "HammingScoreMetric": HammingScoreMetric()}
                                                    )


    @classmethod
    def instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
