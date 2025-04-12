from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_dataset
from NER_config import *
from utils import compute_metrics


ds = load_dataset("truongpdd/NER-covid-vietnamese")

model = AutoModelForTokenClassification.from_pretrained("vinai/phoBERT_base",
                                                          num_labels=len(LABEL_LIST),
                                                          id2label=ID2LABLES,
                                                          label2id=LABLES2ID)


training_args = TrainingArguments(
    output_dir=PATH_SAVE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="steps",
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # Nếu có tập validation, cung cấp ở đây
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained(PATH_SAVE)
tokenizer.save_pretrained(PATH_SAVE)
