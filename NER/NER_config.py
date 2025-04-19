
LABELS2ID = {
    "O": 0,
    "B-street": 1,
    "I-street": 2,
    "B-village": 3,
    "I-village": 4,
    "B-ward": 5,
    "I-ward": 6,
    "B-district": 7,
    "I-district": 8,
    "B-city": 9,
    "I-city": 10,
    "B-nameshop": 11,
    "I-nameshop": 12,
}

ID2LABELS = {v: k for k, v in LABELS2ID.items()}
LABEL_LIST = list(LABELS2ID.keys())

# Hyperparameters
MAX_LENGTH       = 128
BATCH_SIZE       = 16
LEARNING_RATE    = 5e-5
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY     = 0.01

# Scheduler
LR_STEP_SIZE = 1  
LR_GAMMA     = 0.95

# Checkpoint & logging
RUN_DIR         = "./NER/out/checkpoints"
LOG_DIR         = "./NER/out/logs"
SAVE_EPOCH_FREQ = 10    
SAVE_PRED_DIR   = "predictions"

# Training data paths
PATH_TRAIN = "./NER/data/train.txt"
PATH_VAL   = "./NER/data/val.txt"