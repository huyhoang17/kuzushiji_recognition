import os


# COMMON
MODEL_FD = "../models"
CPS_FD = "../cps"
LOG_FD = "../logs"
FONT_FP = os.path.abspath('./fonts/NotoSansCJKjp-Regular.otf')
LE_FP = os.path.abspath("./models/le.pkl")
UNICODE_MAP_FP = os.path.abspath("./models/unicode_map.pkl")

FONT_SIZE = 50

# SEGMENTATION MODEL
IMG_SIZE_SEGMENT = 512
BATCH_SIZE_SEGMENT = 8
LR_SEGMENT = 0.01

# CLASSIFICATION MODEL
NO_CLASSES = 3422
IMG_SIZE_CLASSIFY = 64
