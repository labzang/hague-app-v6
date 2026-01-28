from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. 모델 & 토크나이저 로드
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,               # 이진 분류
    problem_type="single_label_classification"
)

# 2. 데이터셋 로드 (예: jsonl 파일)
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

def preprocess(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )
    tokenized["labels"] = examples["label"]
    return tokenized

tokenized_datasets = dataset.map(preprocess, batched=True)
tokenized_datasets.set_format("torch")

# 3. 평가 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    return {"accuracy": acc, "f1": f1}

# 4. Trainer 설정
training_args = TrainingArguments(
    output_dir="./koelectra_orchestrator_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,                   # GPU 있으면 활성화
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# 5. 학습 시작
trainer.train()

# 6. 저장 & 푸시 (선택)
trainer.save_model("./koelectra_orchestrator_finetuned")
tokenizer.save_pretrained("./koelectra_orchestrator_finetuned")

# Hugging Face Hub에 업로드하려면
# trainer.push_to_hub("your-username/koelectra-soccer-orchestrator")
