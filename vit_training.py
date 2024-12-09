import torch
from torch import nn
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer, EarlyStoppingCallback, get_scheduler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import default_data_collator
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.ToTensor()
])

data_dir = "./Datasets/Chinese Hand Gestures Number Recognition Dataset/aug_imgs_split"
train_dataset = ImageFolder(root=f"{data_dir}/train", transform=transform_train)
val_dataset = ImageFolder(root=f"{data_dir}/val", transform=transform_val)
test_dataset = ImageFolder(root=f"{data_dir}/test", transform=transform_val)

# Convert to HuggingFace dataset
def convert_to_hf_dataset(image_folder_dataset, feature_extractor):
    pixel_values = []
    labels = []
    for img, label in image_folder_dataset:
        img_np = img.permute(1, 2, 0).numpy()
        processed = feature_extractor(images=img_np, return_tensors="pt")
        pixel_values.append(processed["pixel_values"][0])
        labels.append(label)
    return Dataset.from_dict({"pixel_values": pixel_values, "label": labels})

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", do_rescale=False)
train_dataset_hf = convert_to_hf_dataset(train_dataset, feature_extractor)
val_dataset_hf = convert_to_hf_dataset(val_dataset, feature_extractor)

# Model setup
num_classes = len(train_dataset.classes)
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
    output_dir="./vit_results",
    load_best_model_at_end=True,  # Save the best model based on validation metrics
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    learning_rate=5e-6,
    weight_decay=0.05,
    logging_dir="./logs",
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
trainer.save_model("./vit_trained_model")
