
LABELS2ID = {
    "B-LOCATION": 0,
    "B-MISCELLANEOUS": 1,
    "B-ORGANIZATION": 2,
    "B-PERSON": 3,
    "I-LOCATION": 4,
    "I-MISCELLANEOUS": 5,
    "I-ORGANIZATION": 6,
    "I-PERSON": 7,
    "O": 8
  }

ID2LABELS = {
    "0": "B-LOCATION",
    "1": "B-MISCELLANEOUS",
    "2": "B-ORGANIZATION",
    "3": "B-PERSON",
    "4": "I-LOCATION",
    "5": "I-MISCELLANEOUS",
    "6": "I-ORGANIZATION",
    "7": "I-PERSON",
    "8": "O"
  }

LABEL_LIST = list(LABELS2ID.keys())

# Hyperparameters
MAX_LENGTH       = 128
BATCH_SIZE       = 32
LEARNING_RATE    = 5e-5
NUM_TRAIN_EPOCHS = 20
WEIGHT_DECAY     = 0.01

# Scheduler
LR_STEP_SIZE = 1  
LR_GAMMA     = 0.95

# Checkpoint & logging
RUN_DIR         = "./NER/out/checkpoints"
LOG_DIR         = "./NER/out/logs"
SAVE_EPOCH_FREQ = 10    
SAVE_PRED_DIR   = "predictions"
PATH_SAVE_TOKENIZER = "./NER/out/tokenizer/"

# Training data paths
PATH_TRAIN = "./NER/data/train_electra.txt"
PATH_VAL   = "./NER/data/eval_electra.txt"