import os

# Training commands
TRAIN_STOP = {
    "cmd": 0,
    "data": 0,
}
TRAIN_SETUP = {
    "cmd": 0,
    "data": 1,
}
TRAIN_SETUP_DONE = {
    "cmd": 0,
    "data": 2,
}
TRAIN_START = {
    "cmd": 0,
    "data": 3,
}
TRAIN_START_MODEL_DONE = {
    "cmd": 0,
    "data": 4,
}

# Predicting commands
PREDICT_STOP = {
    "cmd": 1,
    "data": 0,
}
PREDICT_SETUP = {
    "cmd": 1,
    "data": 1,
}
PREDICT_SETUP_DONE = {
    "cmd": 1,
    "data": 2,
}
PREDICT_START = {
    "cmd": 1,
    "data": 3,
}
PREDICT_DONE = {
    "cmd": 1,
    "data": 4,
}

# Debugging commands
DEBUG_STOP = {
    "cmd": 2,
    "data": 0,
}
DEBUG_START = {
    "cmd": 2,
    "data": 1,
}
DEBUG_DONE = {
    "cmd": 2,
    "data": 2,
}

# Model
MODEL_NAME = "ddqn_quant_rbot"
MODEL_DIR = os.path.dirname(os.path.realpath(__file__)) + "/saved_model/"

MODEL_PATH = MODEL_DIR + MODEL_NAME + ".tflite"

EPISODES = 300

STATE_SIZE = 4
ACTION_SIZE = 2
PITCH_DATA_THRESHOLD = 0.4
