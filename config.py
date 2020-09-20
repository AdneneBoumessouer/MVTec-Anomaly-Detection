# Data augmentation parameters (only for training)
ROT_ANGLE = 5
W_SHIFT_RANGE = 0.05
H_SHIFT_RANGE = 0.05
FILL_MODE = "nearest"
BRIGHTNESS_RANGE = [0.95, 1.05]
VAL_SPLIT = 0.1

# Learning Rate Finder parameters
START_LR = 1e-5
LR_MAX_EPOCHS = 10
LRF_DECREASE_FACTOR = 0.85

# Training parameters
EARLY_STOPPING = 12
REDUCE_ON_PLATEAU = 6

# Finetuning parameters
FINETUNE_SPLIT = 0.1
STEP_MIN_AREA = 5
START_MIN_AREA = 5
STOP_MIN_AREA = 1005
