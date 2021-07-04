# backbone models
PRETRAINED_MODELS = {
    "resnet50":"checkpoints/pretrained/resnet50.pth",
    "densenet161":"checkpoints/pretrained/densenet161.pth",
}


# Datasets
DATA_ROOT = './data/CUB_200_2011/'

RESIZE = 550
CROP = 448
TRAIN_BATCH_SIZE = 5
TEST_BATCH_SIZE = 1
NUM_WORKERS = 2
NUM_CLASS = 200

# Training
TOTAL_EPOCH = 200
WARM_UP_EPOCH = 5
LR = 0.001
MOMENTUM = 0.9
WD = 5e-4

# Checkpoints
CHECKPOINT_SAVED_FOLDER = "./checkpoints/"
SAVE_MODEL_NAME = "lrcm-cubs-densenet"
SAVE_MODEL_FREQ = 10

# Visualize.
RECORD_FREQ = 10
RECORD_DIR = "./record/"
