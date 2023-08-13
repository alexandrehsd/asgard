import os
import numpy as np
import wandb

import tensorflow as tf
import tensorflow_hub as hub  # noqa: F401
import tensorflow_text as text  # noqa: F401
from official.nlp import optimization

from asgard.metrics.metrics import HammingScoreMetric
from asgard.utils.weights import get_weighted_loss
from asgard.utils.monitor import LOGGER

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


def get_weights(class_weight_kind):
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
        
    return weights

def normalize(metric):
    return metric / 20.525 


def invert(metric):
    # apply linear transformation to revert metrics
    return 1 - metric

def fbeta(metric1, metric2, beta=1):
    return (1 + beta**2) * metric1 * metric2 / ((beta**2) * metric1 + metric2)

def throne():
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("alexandre-hsd/ASGARD")

    summary_list, config_list, name_list, id_list = [], [], [], []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
        id_list.append(run.id)
        
    config_throne = None
    epochs_throne = None
    model_name_throne = None
    id_throne = None
    fgreen_throne = 0

    for summary, config, model_name, run_id in zip(summary_list, config_list, name_list, id_list):

        f1 = summary["Overall F1"]
        co2_score = invert(normalize(summary["Co2 Emissions"]))
        fgreen_curr = fbeta(f1, co2_score)
        
        if fgreen_curr > fgreen_throne:
            fgreen_throne = fgreen_curr
            config_throne = config
            model_name_throne = model_name
            id_throne = run_id
            epochs_throne = summary["best_epoch"]
    
    model_name = f"model-{model_name_throne}:v{epochs_throne}"
    model_path = f"alexandre-hsd/ASGARD/{model_name}"
    
    # check if the selected model was the same model
    model_dir = f"./api/model/{model_name}"

    if not os.path.exists(model_dir):
        LOGGER.INFO(f"Downloading artifact from {model_path}")
        run = wandb.init(project="ASGARD")
        artifact = run.use_artifact(model_name, type='model')
        model_path = artifact.download(f"./api/model/{model_name}")
    else:
        model_path = model_dir
        
    return model_path, model_name, config_throne, id_throne 


def load_model(current_model=None):

    model_path, model_name, config, run_id = throne()

    # TODO: Find a way to get the run_id dinamically
    run_id = f"alexandre-hsd/ASGARD/{run_id}"
    
    # Set the model parameters needed to load the custom_objects of the model
    init_lr = config["learning_rate"]
    epochs = config["epochs"]
    class_weight_kind = config["class_weight_kind"]
    
    weights = get_weights(class_weight_kind)
    
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
    return model_name, model


class Payload(BaseModel):
    text: str


class ModelsLoaderSingleton:
    __instance = None

    def __init__(self):
        name, model = load_model()
        
        self.name = name
        self.model = model

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance
    
    @classmethod
    def update_model(cls):
        current_model = cls.__instance.name
        
        _, model_name, _, _ = throne()
        if current_model != model_name:
            LOGGER.info(f"Loading new model: {model_name}")
            cls.__instance.model = load_model(current_model)
        else:
            LOGGER.info("The current model is still the best")
