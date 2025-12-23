# Quick Reference: Training Functions Cheat Sheet

## Import All Functions
```python
from training_utils import *
```

---

## 🎯 Most Used Functions (Copy & Paste)

### 1. Setup Callbacks (One-liner)
```python
callbacks = create_training_callbacks(experiment_name="my_experiment")
```

### 2. Train Model (With monitoring)
```python
history, training_time = train_model(
    model=model,
    train_data=train_data,
    val_data=test_data,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)
print(f"Training took {training_time/60:.2f} minutes")
```

### 3. Display Training Plan (Before starting)
```python
display_training_plan(
    model=model,
    epochs=10,
    learning_rate=1e-3,
    batch_size=32,
    train_samples=76000,
    val_samples=25000
)
```

### 4. Unfreeze Layers (For fine-tuning)
```python
unfreeze_layers(model, num_layers=50)
```

### 5. Recompile with Lower LR (After unfreezing)
```python
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"]
)
```

### 6. Track Metrics (Across phases)
```python
metrics = TrainingMetrics()
metrics.record(history1)
metrics.record(history2)
metrics.save("my_metrics.json")
```

### 7. Evaluate Model (With formatted output)
```python
eval_metrics = evaluate_model(model, test_data)
```

### 8. Compare Two Models (A/B testing)
```python
comparison = compare_models(model1, model2, test_data)
print(f"Winner: {comparison['winner']}")
print(f"Accuracy gain: {comparison['accuracy_improvement']:+.4f}")
```

### 9. Print Model Info (Check parameters)
```python
print_model_info(model)
```

### 10. Get Model Stats (For custom reporting)
```python
info = get_model_summary_info(model)
print(f"Total params: {info['total_parameters']:,}")
print(f"Trainable: {info['trainable_parameters']:,}")
```

---

## 📋 Complete Two-Phase Training Template

```python
from training_utils import *

# ===== PHASE 1: FEATURE EXTRACTION (Frozen Base) =====
print("🎯 PHASE 1: FEATURE EXTRACTION")

# Show plan
display_training_plan(model, epochs=3, learning_rate=1e-3, batch_size=32,
                     train_samples=76000, val_samples=25000)

# Setup & train
callbacks1 = create_training_callbacks(experiment_name="phase1_feature_extract")
history1, time1 = train_model(model, train_data, test_data, epochs=3, callbacks=callbacks1)

# Track metrics
metrics = TrainingMetrics()
metrics.record(history1)

# Evaluate
eval1 = evaluate_model(model, test_data)
print(f"⏱️  Phase 1: {time1/60:.2f} minutes\n")


# ===== PHASE 2: FINE-TUNING (Unfrozen Base) =====
print("🎯 PHASE 2: FINE-TUNING")

# Unfreeze layers
unfreeze_layers(model, num_layers=50)

# Recompile with LOWER learning rate
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # 100x lower!
    metrics=["accuracy"]
)

# Show new plan
display_training_plan(model, epochs=10, learning_rate=1e-5, batch_size=32,
                     train_samples=76000, val_samples=25000)

# Setup & train
callbacks2 = create_training_callbacks(experiment_name="phase2_fine_tune")
history2, time2 = train_model(model, train_data, test_data, epochs=10, 
                             initial_epoch=3, callbacks=callbacks2)

# Track metrics
metrics.record(history2)
metrics.save("complete_metrics.json")

# Evaluate
eval2 = evaluate_model(model, test_data)
print(f"⏱️  Phase 2: {time2/60:.2f} minutes")
print(f"⏱️  Total: {(time1 + time2)/60:.2f} minutes\n")


# ===== PHASE 3: COMPARISON =====
print("🎯 PHASE 3: MODEL COMPARISON")
# (Compare if you saved the phase1 model earlier)
# comparison = compare_models(saved_phase1_model, model, test_data)
```

---

## 🔧 Common Tasks

### Task: Resume Training from Checkpoint
```python
import glob
import os

# Find latest checkpoint
checkpoints = glob.glob("model_checkpoints/*.index")
if checkpoints:
    latest = max(checkpoints, key=os.path.getctime)
    ckpt_path = latest.replace(".index", "")
    model.load_weights(ckpt_path)
    
    # Resume
    callbacks = create_training_callbacks(experiment_name="resumed")
    history, _ = train_model(model, train_data, test_data, 
                            epochs=20, initial_epoch=3, callbacks=callbacks)
```

### Task: Save Configuration for Reproducibility
```python
config = {
    "model": "EfficientNetV2B0",
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs_phase1": 3,
    "epochs_phase2": 7,
    "data_augmentation": ["flip", "rotate", "zoom"],
    "callbacks": ["checkpoint", "early_stop", "reduce_lr", "tensorboard"]
}

from training_utils import save_training_config
save_training_config(config, "training_config.json")

# Load later
config = load_training_config("training_config.json")
```

### Task: Analyze Training Results
```python
import json

# Load metrics
with open("complete_metrics.json") as f:
    metrics = json.load(f)

# Find best epoch
val_acc = metrics['val_accuracy']
best_epoch = val_acc.index(max(val_acc))
best_acc = max(val_acc)

print(f"Best accuracy: {best_acc:.4f} at epoch {best_epoch}")

# Check for overfitting
train_acc = metrics['train_accuracy'][-1]
val_acc_final = metrics['val_accuracy'][-1]
gap = train_acc - val_acc_final

if gap > 0.15:
    print(f"⚠️  Overfitting! Gap: {gap:.4f}")
else:
    print(f"✅ Good generalization. Gap: {gap:.4f}")
```

### Task: Monitor with TensorBoard
```bash
# In terminal
tensorboard --logdir training_logs
# Visit http://localhost:6006
```

### Task: Diagnose Training Issues
```python
# Check if model is compiled
if model.optimizer is None:
    print("❌ Not compiled!")
else:
    print("✅ Model compiled")
    print(f"LR: {model.optimizer.learning_rate}")

# Check model structure
info = get_model_summary_info(model)
print(f"Trainable: {info['trainable_parameters']:,} / {info['total_parameters']:,}")

# Print full summary
print_model_info(model)
```

---

## 📊 Learning Rates by Phase (Recommended)

| Phase | Base Model | Learning Rate | Epochs |
|-------|-----------|--------------|--------|
| Feature Extract | Frozen | 1e-3 | 3-5 |
| Fine-tune (mild) | Partial unfreeze | 1e-4 | 5-10 |
| Fine-tune (aggressive) | Full unfreeze | 1e-5 | 3-7 |

---

## ⏱️ Training Time Estimates (GPU)

| Model Size | Batch Size | Train Samples | Estimated Time/Epoch |
|-----------|-----------|---------------|---------------------|
| EfficientNetV2B0 | 32 | 76000 | 2-3 min |
| EfficientNetV2B0 | 16 | 76000 | 4-5 min |
| EfficientNetV2L | 32 | 76000 | 8-10 min |

**Total Two-Phase Training (EfficientNetV2B0):**
- Phase 1 (3 epochs): ~6-9 minutes
- Phase 2 (7 epochs): ~14-21 minutes
- **Total: ~20-30 minutes**

---

## 🚨 When to Use Each Function

| Situation | Use This |
|-----------|----------|
| Starting training | `display_training_plan()` |
| Creating callbacks | `create_training_callbacks()` |
| Running training | `train_model()` |
| Checking results | `evaluate_model()` |
| Recording metrics | `TrainingMetrics()` + `.record()` + `.save()` |
| Fine-tuning | `unfreeze_layers()` |
| Comparing models | `compare_models()` |
| Checking parameters | `get_model_summary_info()` or `print_model_info()` |
| Debugging | `print_model_info()` |
| Saving config | `save_training_config()` |

---

## ✅ Checklist Before Training

```python
# 1. Check model is compiled
assert model.optimizer is not None, "Model not compiled!"

# 2. Check data
assert len(train_data) > 0, "No training data!"
assert len(test_data) > 0, "No validation data!"

# 3. Display plan
display_training_plan(model, epochs=10, learning_rate=1e-3, batch_size=32, 
                     train_samples=76000, val_samples=25000)

# 4. Check model size
info = get_model_summary_info(model)
print(f"Model size: {info['total_parameters']/1e6:.1f}M parameters")

# 5. Setup callbacks
callbacks = create_training_callbacks()

# 6. Train!
history, time = train_model(model, train_data, test_data, epochs=10, callbacks=callbacks)

# 7. Save metrics
metrics = TrainingMetrics()
metrics.record(history)
metrics.save()

print("✅ All checks passed and training complete!")
```

---

## 🎓 One-Liner Examples

```python
# Setup everything
callbacks = create_training_callbacks(); display_training_plan(model, 10, 1e-3, 32, 76000, 25000)

# Train
history, t = train_model(model, train_data, test_data, 10, callbacks=callbacks)

# Evaluate
evaluate_model(model, test_data)

# Track
m = TrainingMetrics(); m.record(history); m.save()

# Compare (if you have two models)
compare_models(model1, model2, test_data)
```

---

## 📞 Troubleshooting

| Error | Solution |
|-------|----------|
| "Model not compiled" | Call `model.compile()` before `train_model()` |
| "No checkpoints found" | Make sure training ran at least once |
| TensorBoard not showing | Check log_dir path exists |
| Out of memory | Reduce batch_size in `train_model()` |
| Training too slow | Use `unfreeze_layers()` selectively |
| Overfitting | Increase `patience` in `create_training_callbacks()` |
| Not improving | Reduce learning rate or increase data augmentation |

---

**Version:** 1.0  
**Last Updated:** December 23, 2025  
**Status:** Production Ready ✅
