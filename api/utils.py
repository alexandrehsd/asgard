import os
import numpy as np
import wandb

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
from official.nlp import optimization

from asgard.metrics.metrics import HammingScoreMetric
from asgard.utils.weights import get_weighted_loss

from pydantic import BaseModel


def is_model_downloaded(model_root_path):
    for root, dirs, files in os.walk(model_root_path):
        for dir_name in dirs:
            if dir_name.startswith("model-"):
                return True
    return False
    

def get_model_path(model_root_path):
    class ModelFolderNotFoundException(Exception):
        pass

    for root, dirs, files in os.walk(model_root_path):
        for dir_name in dirs:
            if dir_name.startswith("model"):
                return os.path.join("./api/model", dir_name)
    raise ModelFolderNotFoundException("Model Not Found.")


def get_run_configs(run_id):
    api = wandb.Api()
    run = api.run(run_id)
    
    return run.config
    

def load_model():
    model_root_path = "./api/model"
        
    if is_model_downloaded(model_root_path):
        model_path = get_model_path(model_root_path)
    else:
        # TODO: Find a way to get the model_name dinamically
        model_name = "model-avid-sweep-1:v2"
        default_model = f"alexandre-hsd/ASGARD-DistilBERT/{model_name}"
        
        # TODO: WandB credentials must be passed to the container to load run configs
        run = wandb.init()
        artifact = run.use_artifact(default_model, type='model')
        model_path = artifact.download(f"./api/model/{model_name}")
        
    # TODO: Find a way to get the run_id dinamically
    # run_id = "alexandre-hsd/ASGARD-DistilBERT/ab419eqb"
    
    # Set the model parameters needed to load the custom_objects of the model
    # TODO: WandB credentials must be passed to the container to load run configs
    # run_configs = get_run_configs(run_id)
    # init_lr = run_configs["learning_rate"]
    # epochs = run_configs["epochs"]
    # class_weight_kind = run_configs["class_weight_kind"]
    
    init_lr = 1e-4
    epochs = 4
    class_weight_kind = "two-to-one"
    
    if (class_weight_kind is None) or (class_weight_kind == "None"):
        weights = None
    elif class_weight_kind == "two-to-one":
        weights = np.zeros((16, 2))
        weights[:, 0] = 1.
        weights[:, 1] = 2.
    elif class_weight_kind == "balanced":
        weights = [[0.51822883, 14.21454078],[0.55565963, 4.99158589], [0.56621277, 4.27570651],
                   [0.55837284, 4.78281348], [0.53639343, 7.36937204], [0.56483329, 4.35604386],
                   [0.56483008, 4.35623454], [0.56487897, 4.35332845], [0.56475475, 4.36072039],
                   [0.56470268, 4.36382771], [0.56472831, 4.3622974 ], [0.56227303, 4.51457929],
                   [0.56466022, 4.36636465], [0.54509005, 6.04446064], [0.55036404, 5.46385923],
                   [0.55393068, 5.13558076]]
    
    steps_per_epoch = 24813
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr,
        decay_steps=num_train_steps,
        end_learning_rate=0.0,
        power=1.0
        )
    
    model = tf.keras.models.load_model(model_path,
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
    return model


class Payload(BaseModel):
    text: str


class ModelsLoaderSingleton:
    __instance = None

    def __init__(self):
        self.model = load_model()

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
