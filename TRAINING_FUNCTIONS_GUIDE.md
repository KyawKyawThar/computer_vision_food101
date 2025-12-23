# Advanced Training Functions & Best Practices
## Food Vision 101 - Production Guide

---

## 📋 Missing Training Functions Added

### New Module: `training_utils.py`

A comprehensive module has been created with production-ready training utilities. Here's what was added:

### 1. **TrainingMetrics Class** ✅
Tracks and logs all training metrics for analysis.

```python
from training_utils import TrainingMetrics

metrics = TrainingMetrics(log_dir="training_metrics")
metrics.record(history)
metrics.save("metrics.json")
```

**Features:**
- Records loss, accuracy, learning rate across epochs
- Saves metrics to JSON for later analysis
- Easy integration with model.fit() results

---

### 2. **create_training_callbacks()** ✅
Enhanced callback creation with 5 essential callbacks.

```python
from training_utils import create_training_callbacks

callbacks = create_training_callbacks(
    model_dir="model_checkpoints",
    tensorboard_dir="training_logs",
    experiment_name="feature_extract",
    patience=5,
    reduce_lr_patience=3
)
```

**Callbacks Included:**
1. **ModelCheckpoint** - Save best weights based on validation accuracy
2. **EarlyStopping** - Stop if no improvement for N epochs
3. **ReduceLROnPlateau** - Dynamically reduce learning rate
4. **TensorBoard** - Real-time visualization with histograms
5. **LambdaCallback** - Custom epoch-end logging

---

### 3. **train_model()** ✅
Wrapper function for model.fit() with error handling.

```python
from training_utils import train_model

history, training_time = train_model(
    model=model,
    train_data=train_data,
    val_data=val_data,
    epochs=10,
    initial_epoch=0,
    callbacks=callbacks
)

print(f"Training took {training_time/60:.2f} minutes")
```

**Features:**
- Automatic error handling and validation
- Measures total training time
- Detailed logging with timestamps
- Validates model is compiled

---

### 4. **evaluate_model()** ✅
Enhanced evaluation with detailed metrics display.

```python
from training_utils import evaluate_model

metrics = evaluate_model(
    model=model,
    test_data=test_data
)

# Returns: {"loss": 0.45, "accuracy": 0.92}
```

**Features:**
- Handles different model output formats
- Pretty prints results
- Returns metrics as dictionary for further analysis

---

### 5. **get_model_summary_info()** ✅
Extract comprehensive model statistics.

```python
from training_utils import get_model_summary_info

info = get_model_summary_info(model)
print(info["total_parameters"])  # Total params in model
print(info["trainable_parameters"])  # Params that can be updated
```

**Returns:**
- Total/trainable/non-trainable parameters
- Model name and shapes
- Layer count

---

### 6. **print_model_info()** ✅
Pretty print model information.

```python
from training_utils import print_model_info

print_model_info(model)
# Outputs formatted table with all model stats
```

---

### 7. **unfreeze_layers()** ✅
Selectively unfreeze layers for fine-tuning.

```python
from training_utils import unfreeze_layers

# Unfreeze last 50 layers for fine-tuning
unfreeze_layers(model, num_layers=50)

# Now recompile with lower learning rate
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"]
)
```

**Best Practice:**
- Freeze early layers (learned general features)
- Unfreeze later layers (dataset-specific features)
- Use lower learning rate (1e-5 to 1e-4) for fine-tuning

---

### 8. **display_training_plan()** ✅
Show complete training plan before starting.

```python
from training_utils import display_training_plan

display_training_plan(
    model=model,
    epochs=10,
    learning_rate=1e-3,
    batch_size=32,
    train_samples=76000,
    val_samples=25000
)
```

**Shows:**
- Data statistics
- Configuration parameters
- Model parameter count
- **Estimated training time**

---

### 9. **TrainingMetrics.record() & save()** ✅
Track metrics across training phases.

```python
metrics = TrainingMetrics()

# After first training phase
metrics.record(history_phase1)

# After second training phase
metrics.record(history_phase2)

# Save everything
metrics.save("complete_training_metrics.json")
```

---

### 10. **compare_models()** ✅
Compare two models side-by-side.

```python
from training_utils import compare_models

comparison = compare_models(
    model1=feature_extract_model,
    model2=fine_tuned_model,
    test_data=test_data
)

print(comparison["winner"])  # "Model 1" or "Model 2"
print(f"Accuracy gain: {comparison['accuracy_improvement']:+.4f}")
```

---

## 🎯 How to Use in Your Notebook

### **Phase 1: Feature Extraction**

```python
from training_utils import (
    create_training_callbacks,
    train_model,
    display_training_plan,
    TrainingMetrics,
    print_model_info
)

# Step 1: Display plan
display_training_plan(
    model=model,
    epochs=3,
    learning_rate=1e-3,
    batch_size=32,
    train_samples=76000,
    val_samples=25000
)

# Step 2: Create callbacks
callbacks = create_training_callbacks(
    experiment_name="efficientnetv2b0_feature_extract"
)

# Step 3: Train
history_fe, training_time = train_model(
    model=model,
    train_data=train_data,
    val_data=test_data,
    epochs=3,
    callbacks=callbacks
)

# Step 4: Track metrics
metrics = TrainingMetrics()
metrics.record(history_fe)
```

### **Phase 2: Fine-Tuning**

```python
from training_utils import unfreeze_layers, train_model

# Step 1: Unfreeze layers
unfreeze_layers(loaded_model, num_layers=50)

# Step 2: Recompile with lower learning rate
loaded_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"]
)

# Step 3: Create new callbacks
callbacks = create_training_callbacks(
    experiment_name="efficientnetv2b0_fine_tune"
)

# Step 4: Train
history_ft, training_time = train_model(
    model=loaded_model,
    train_data=train_data,
    val_data=test_data,
    epochs=10,
    initial_epoch=3,
    callbacks=callbacks
)

# Step 5: Track additional metrics
metrics.record(history_ft)
metrics.save("full_training_metrics.json")
```

### **Phase 3: Evaluation & Comparison**

```python
from training_utils import evaluate_model, compare_models

# Evaluate fine-tuned model
eval_metrics = evaluate_model(model=loaded_model, test_data=test_data)

# Compare feature extraction vs fine-tuning
comparison = compare_models(
    model1=created_model,  # Feature extraction
    model2=loaded_model,   # Fine-tuned
    test_data=test_data
)
```

---

## 🔧 Configuration Best Practices

### **For Feature Extraction (Frozen Base):**
```python
config = {
    "phase": "feature_extraction",
    "base_model_trainable": False,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 3,
    "early_stopping_patience": 3
}
```

### **For Fine-Tuning (Unfrozen Base):**
```python
config = {
    "phase": "fine_tuning",
    "base_model_trainable": True,
    "unfrozen_layers": 50,
    "learning_rate": 1e-5,  # Much lower!
    "batch_size": 32,
    "epochs": 7,
    "early_stopping_patience": 5
}
```

---

## 📊 Monitoring & Logging

### **Real-time Monitoring:**
```bash
# In terminal
tensorboard --logdir training_logs
# Visit http://localhost:6006
```

### **Access Saved Metrics:**
```python
import json

with open("training_metrics/metrics.json") as f:
    metrics = json.load(f)
    
print(f"Best validation accuracy: {max(metrics['val_accuracy']):.4f}")
```

---

## ⚠️ Common Training Issues & Solutions

### **Issue 1: Model Not Improving**
```python
# Check if model is compiled
if model.optimizer is None:
    model.compile(...)  # Compile before training!

# Check learning rate
current_lr = model.optimizer.learning_rate.numpy()
print(f"Current LR: {current_lr}")
```

### **Issue 2: Out of Memory (OOM)**
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Or reduce model size
# Use EfficientNetV2S instead of L
```

### **Issue 3: Overfitting**
```python
# Increase regularization
early_stop_cb = tf.keras.callbacks.EarlyStopping(
    patience=3  # Lower patience = stop earlier
)

# Add more data augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),  # Add vertical
    layers.RandomRotation(0.3),     # Increase rotation
    layers.RandomZoom(0.3),         # Increase zoom
])
```

---

## 📈 Performance Tips

### **1. Mixed Precision Training** (Faster, uses less memory)
```python
# Enable at start of notebook
mixed_precision = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(mixed_precision)
```

### **2. Gradient Accumulation** (For larger effective batch size)
```python
# Implement custom training loop
# Accumulate gradients over N steps before updating
```

### **3. Distributed Training** (Multi-GPU)
```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = create_model()
    model.compile(...)
```

---

## ✅ Production Checklist

Before deploying to production:

- [ ] Use `display_training_plan()` to verify configuration
- [ ] Set up `TrainingMetrics` to track all training
- [ ] Enable `TensorBoard` callbacks for monitoring
- [ ] Use `unfreeze_layers()` for systematic fine-tuning
- [ ] Save training config with `save_training_config()`
- [ ] Compare models with `compare_models()` before deployment
- [ ] Validate model on separate test set
- [ ] Document final training configuration
- [ ] Keep backup of best checkpoint
- [ ] Test inference speed on production hardware

---

## 🚀 Quick Start Template

```python
from training_utils import *

# 1. Display plan
display_training_plan(model, epochs=10, learning_rate=1e-3, 
                     batch_size=32, train_samples=76000, val_samples=25000)

# 2. Create callbacks
callbacks = create_training_callbacks(experiment_name="my_exp")

# 3. Train
history, duration = train_model(model, train_data, test_data, epochs=10, callbacks=callbacks)

# 4. Evaluate
metrics = evaluate_model(model, test_data)

# 5. Save metrics
metrics_tracker = TrainingMetrics()
metrics_tracker.record(history)
metrics_tracker.save()

print(f"✅ Training completed in {duration/60:.2f} minutes!")
```

---

## 📚 Reference Functions Summary

| Function | Purpose | Key Feature |
|----------|---------|------------|
| `create_training_callbacks()` | Setup all training callbacks | 5-in-1 callbacks |
| `train_model()` | Execute training with monitoring | Error handling + timing |
| `evaluate_model()` | Evaluate and display results | Formatted output |
| `display_training_plan()` | Preview training before start | **Estimates duration** |
| `unfreeze_layers()` | Selective layer unfreezing | For fine-tuning |
| `print_model_info()` | Display model statistics | Pretty formatted |
| `compare_models()` | A/B test models | Side-by-side comparison |
| `TrainingMetrics` | Track across training phases | JSON export |

---

**Created:** December 23, 2025  
**Status:** Production-Ready  
**Python Version:** 3.7+  
**TensorFlow Version:** 2.10+
