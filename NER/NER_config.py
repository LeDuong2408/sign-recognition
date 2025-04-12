
LABLES2ID = {
    "O": 0,
    "B-homenumber": 1,
    "I-homenumber": 2,
    "B-street": 3,
    "I-street": 4,
    "B-village": 5,
    "I-village": 6,
    "B-ward": 7,
    "I-ward": 8,
    "B-district": 9,
    "I-district": 10,
    "B-city": 11,
    "I-city": 12,
    "B-nameshop": 13,
    "I-nameshop": 14,
}

ID2LABLES = {v: k for k, v in LABLES2ID.items()}
LABEL_LIST = list(LABLES2ID.keys())
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128
EVAL_STEPS = 500
SAVE_STEPS = 500
PATH_SAVE = "./out_ner"
MODEL_NAME = "vinai/bert-base-uncased"