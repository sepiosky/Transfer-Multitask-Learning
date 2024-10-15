import os
import warnings

#from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode as ConfigurationNode
from pathlib import Path
from src.utils import get_backbone_embeddings_size

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

__C = ConfigurationNode(new_allowed=True)

# importing default as a global singleton
cfg = __C
__C.MODULES_PATH = ConfigurationNode(new_allowed=True)


__C.DESCRIPTION = 'Default config from the Singleton'
__C.EXPERIMENT_NAME = 'baseline'

__C.DATASET = ConfigurationNode(new_allowed=True)
__C.DATASET.NAME = 'cifar10'
__C.DATASET.ALPHABET = '' # Omniglot alphabet name if its single task


__C.DATASET.DOWNLOAD = True
__C.DATASET.TRAIN_DATA_PATH = ''
__C.DATASET.VAL_DATA_PATH = ''
__C.DATASET.TRAIN_DATA_SAMPLE = ''
__C.DATASET.VALID_DATA_SAMPLE = ''


__C.DATASET.PREPROCESS = True
__C.DATASET.DEFAULT_SIZE = (32, 32)
__C.DATASET.RESIZE_SHAPE = (32, 32) #if not using preprocessor set it equal to default value
__C.DATASET.INPUT_CHANNELS = 3 # should be set to backbone input_channels. data channels is set via TO_RGB
__C.DATASET.REVERSE_BACKGROUND = False

__C.DATASET.BATCH_SIZE = 32
__C.DATASET.CPU_NUM = 0
__C.DATASET.TO_RGB = False
__C.DATASET.NORMALIZE = False
__C.DATASET.NORMALIZE_MEAN = [0.5, 0.5, 0.5]
__C.DATASET.NORMALIZE_STD = [0.5, 0.5, 0.5]

__C.DATASET.AUGMENTATION = ConfigurationNode(new_allowed=True)
__C.DATASET.AUGMENTATION.BLURRING_PROB = 0.25 #0.25
__C.DATASET.AUGMENTATION.BLUR_LIMIT = 7
__C.DATASET.AUGMENTATION.GAUSS_NOISE_PROB = 0.25 #0.25
__C.DATASET.AUGMENTATION.GAUSS_VAR_LIMIT =(10.0, 40.0)
__C.DATASET.AUGMENTATION.BRIGHTNESS_CONTRAST_PROB = 0.3 #0.5
__C.DATASET.AUGMENTATION.GRID_DISTORTION_PROB = 0 #0.5
__C.DATASET.AUGMENTATION.ROTATION_PROB = 0.5 #0.5
__C.DATASET.AUGMENTATION.ROTATION_DEGREE = 25
__C.DATASET.AUGMENTATION.GRID_MASK_PROB = 0 #0.0
__C.DATASET.AUGMENTATION.HORIZONTAL_FLIP_PROB = 0.1 #0.0
__C.DATASET.AUGMENTATION.CUTOUT_PROB = 0 #0.4
__C.DATASET.AUGMENTATION.HEIGHT = 0
__C.DATASET.AUGMENTATION.WIDTH = 0

__C.MODEL = ConfigurationNode(new_allowed=True)

__C.MODEL.PARALLEL = False
__C.MODEL.LOAD_FROM_FILE = False
__C.MODEL.PRETRAINED_PATH = ''
__C.MODEL.META_ARCHITECTURE = 'baseline'
__C.MODEL.NORMALIZATION_FN = 'BN'

__C.MODEL.BACKBONE = ConfigurationNode(new_allowed=True)
__C.MODEL.BACKBONE.NAME = 'protonet'
__C.MODEL.BACKBONE.INPUT_CHANNELS = __C.DATASET.INPUT_CHANNELS
# __C.MODEL.BACKBONE.PRETRAINED_PATH = ''
__C.MODEL.BACKBONE.PRETRAINED_PATH = ''
__C.MODEL.BACKBONE.FREEZE = False


__C.MODEL.BODY = ConfigurationNode(new_allowed=True)
__C.MODEL.BODY.NAME = 'simple_body'
# __C.MODEL.BODY.PRETRAINED_PATH = ''
__C.MODEL.BODY.ACTIVATION = 'leaky_relu'
__C.MODEL.BODY.INPUT_DIM = 512
__C.MODEL.BODY.HIDDEN_DIMS = [512, 256]
__C.MODEL.BODY.HIDDEN_DIMS_FREEZE = [False, False]
__C.MODEL.BODY.OUTPUT_DIM = 10 #cifar10
__C.MODEL.BODY.BN = True
__C.MODEL.BODY.DROPOUT = -1.0

__C.MODEL.HEADS = ConfigurationNode(new_allowed=True)
__C.MODEL.HEADS.NAMES = [] #['simple_head' for _ in range(len(__C.MODEL.BODY.OUTPUT_DIMS))]
# __C.MODEL.HEADS.PRETRAINED_PATH = ''
__C.MODEL.HEADS.INPUT_DIMS = []
__C.MODEL.HEADS.HIDDEN_DIMS = [] # [ [512, 256] for _ in range(30)]
__C.MODEL.HEADS.HIDDEN_DIMS_FREEZE = [] #[ [False, False] for _ in range(30)]
__C.MODEL.HEADS.ACTIVATION = 'leaky_relu'
__C.MODEL.HEADS.OUTPUT_DIMS = [__C.MODEL.BODY.OUTPUT_DIM]
__C.MODEL.HEADS.BN = True
__C.MODEL.HEADS.DROPOUT = -1.0

__C.MODEL.SOLVER = ConfigurationNode(new_allowed=True)
#__C.MODEL.SOLVER.LABELS_WEIGHTS =
__C.MODEL.SOLVER.OPTIMIZER = ConfigurationNode(new_allowed=True)
__C.MODEL.SOLVER.OPTIMIZER.PRETRAINED_PATH = __C.MODEL.PRETRAINED_PATH
__C.MODEL.SOLVER.OPTIMIZER.BASE_LR = 0.001
__C.MODEL.SOLVER.OPTIMIZER.NAME = 'adam'

# sgd config
__C.MODEL.SOLVER.OPTIMIZER.SGD = ConfigurationNode(new_allowed=True)
__C.MODEL.SOLVER.OPTIMIZER.SGD.MOMENTUM = 0.9
__C.MODEL.SOLVER.OPTIMIZER.SGD.NESTEROV = False

__C.MODEL.SOLVER.SCHEDULER = ConfigurationNode(new_allowed=True)
__C.MODEL.SOLVER.SCHEDULER.PRETRAINED_PATH = __C.MODEL.PRETRAINED_PATH
__C.MODEL.SOLVER.SCHEDULER.NAME = 'unchange'
__C.MODEL.SOLVER.SCHEDULER.LR_REDUCE_GAMMA = 0.1
__C.MODEL.SOLVER.SCHEDULER.MULTI_STEPS_LR_MILESTONES = []

# OneCycleLR hyperparams
__C.MODEL.SOLVER.SCHEDULER.PCT_START = 0.5
__C.MODEL.SOLVER.SCHEDULER.ANNEAL_STRATEGY = 'cos'
__C.MODEL.SOLVER.SCHEDULER.DIV_FACTOR = 30
__C.MODEL.SOLVER.SCHEDULER.MAX_LR = 0.01

__C.MODEL.SOLVER.TOTAL_EPOCHS = 40
__C.MODEL.SOLVER.AMP = False

__C.MODEL.SOLVER.LOSS = ConfigurationNode(new_allowed=True)
__C.MODEL.SOLVER.LOSS.OHEM_RATE = 1.0
__C.MODEL.SOLVER.LOSS.NAME = 'xentropy'
__C.MODEL.SOLVER.LOSS.EPS = 0.1
__C.MODEL.SOLVER.LOSS.REDUCTION = 'mean'
__C.MODEL.SOLVER.LOSS.LABELS_WEIGHTS_PATH = ''

# focal loss related
__C.MODEL.SOLVER.LOSS.FOCAL_LOSS = ConfigurationNode(new_allowed=True)
__C.MODEL.SOLVER.LOSS.FOCAL_LOSS.GAMMA = 1
__C.MODEL.SOLVER.LOSS.FOCAL_LOSS.ALPHA = -1

__C.OUTPUT_PATH = os.path.abspath('src/results')
__C.RESUME_PATH = ''
__C.MULTI_GPU_TRAINING = False
__C.DEBUG = False


def get_cfg_defaults():
  """
  Get a yacs CfgNode object with default values
  """
  # Return a clone so that the defaults will not be altered
  # It will be subsequently overwritten with local YAML.
  return __C.clone()

def combine_cfgs(path_cfg_data: Path=None, path_cfg_override: Path=None):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_output: path to output files
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data=Path(path_cfg_data)
    if path_cfg_override is not None:
        path_cfg_override=Path(path_cfg_override)
    # Path order of precedence is:
    # Priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 4:
    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    # Priority 3:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 2:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # Merge from .env
    # Priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg_base.merge_from_list(list_cfg)

    return cfg_base

def update_cfg_using_dotenv() -> list:
    """
    In case when there are dotenvs, try to return list of them.

    # It is returning a list of hard overwrite.
    :return: empty list or overwriting information
    """
    # If .env not found, bail
    if find_dotenv() == '':
        warnings.warn(".env files not found. YACS config file merging aborted.")
        return []

    # Load env.
    load_dotenv(find_dotenv(), verbose=True)

    # Load variables
    list_key_env = {
        "DATASET.TRAIN_DATA_PATH",
        "DATASET.VAL_DATA_PATH",
        "MODEL.BACKBONE.PRETRAINED_PATH",
        "MODEL.SOLVER.LOSS.LABELS_WEIGHTS_PATH"
    }

    # Instantiate return list.
    path_overwrite_keys = []

    # Go through the list of key to be overwritten.
    for key in list_key_env:

        # Get value from the env.
        value = os.getenv("path_overwrite_keys")

        # If it is none, skip. As some keys are only needed during training and others during the prediction stage.
        if value is None:
            continue

        # Otherwise, adding the key and the value to the dictionary.
        path_overwrite_keys.append(key)
        path_overwrite_keys.append(value)

    return path_overwrite_keys