# vit_training.py
from pathlib import Path
import torch
from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer, EarlyStoppingCallback, get_scheduler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import default_data_collator
import numpy as np
from PIL import Image

# Data transformations (handled within the Dataset)
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)

# Prepare ImageFolder datasets
data_dir = Path('../Datasets/CSL_Dataset/aug_imgs_split')
train_image_folder = ImageFolder(root=f"{data_dir}/train")
val_image_folder = ImageFolder(root=f"{data_dir}/val")

def get_image_paths_and_labels(image_folder):
    paths = [str(path) for path, _ in image_folder.imgs]
    labels = [label for _, label in image_folder.imgs]
    return {"image_path": paths, "label": labels}

train_data = get_image_paths_and_labels(train_image_folder)
val_data = get_image_paths_and_labels(val_image_folder)

train_dataset_hf = Dataset.from_dict(train_data)
val_dataset_hf = Dataset.from_dict(val_data)

# Define preprocessing function
def preprocess_function(examples):
    images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
    processed = feature_extractor(images=images, return_tensors="pt")
    # Flatten the batch dimension
    pixel_values = processed["pixel_values"]
    return {"pixel_values": pixel_values, "label": examples["label"]}

# Apply preprocessing
train_dataset_hf = train_dataset_hf.map(preprocess_function, batched=True, batch_size=32, remove_columns=["image_path"])
val_dataset_hf = val_dataset_hf.map(preprocess_function, batched=True, batch_size=32, remove_columns=["image_path"])

# Set format for PyTorch
train_dataset_hf.set_format(type="torch", columns=["pixel_values", "label"])
val_dataset_hf.set_format(type="torch", columns=["pixel_values", "label"])

# Model setup
num_classes = len(train_image_folder.classes)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
    problem_type="single_label_classification"
)

# Unfreeze the layers (freeze only the first 4 layers)
for name, param in model.named_parameters():
    if "encoder.layer" in name:
        parts = name.split(".")
        layer_num = int(parts[3])
        if layer_num < 4:
            param.requires_grad = False

# Training arguments
training_args = TrainingArguments(
    output_dir=Path("../Trained_Results/CSL_ViT_results"),
    load_best_model_at_end=True,  # Save the best model based on validation metrics
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    learning_rate=5e-6,
    weight_decay=0.05,
    logging_dir=Path("../Trained_Results/CSL_vit_results_logs"),
    logging_steps=10,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_ratio=0.2,
    max_grad_norm=0.5,
    label_smoothing_factor=0.0,
    gradient_accumulation_steps=2
)

# Optimizer setup
optimizer = torch.optim.AdamW([
    {"params": model.classifier.parameters(), "lr": 1e-5},
    {"params": [p for n, p in model.named_parameters() if "classifier" not in n and "encoder.layer" in n and int(n.split(".")[3]) >= 4], "lr": 5e-6}
])

# Learning rate scheduler
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(training_args.warmup_ratio * len(train_dataset_hf) // training_args.per_device_train_batch_size),
    num_training_steps=training_args.num_train_epochs * len(train_dataset_hf) // training_args.per_device_train_batch_size
)

# Early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

# Metrics computation
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "precision": precision_score(p.label_ids, preds, average="weighted", zero_division=0),
        "recall": recall_score(p.label_ids, preds, average="weighted", zero_division=0),
        "f1": f1_score(p.label_ids, preds, average="weighted", zero_division=0)
    }

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_hf,
    eval_dataset=val_dataset_hf,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[early_stopping]
)

# Start training
trainer.train()
