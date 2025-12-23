"""
Practical Examples: How to Use Advanced Training Functions
===========================================================

This file contains ready-to-use code examples for the training_utils module.
Copy and paste these directly into your notebook!
"""

# ============================================================================
# EXAMPLE 1: Complete Training Pipeline with All Utilities
# ============================================================================

"""
This example shows the complete workflow using all new training utilities.
Run this after loading data and creating your base model.
"""

from training_utils import (
    create_training_callbacks,
    train_model,
    evaluate_model,
    display_training_plan,
    print_model_info,
    TrainingMetrics,
    unfreeze_layers,
    compare_models,
    get_model_summary_info
)
import tensorflow as tf

# ---- PHASE 1: FEATURE EXTRACTION ----

def example_1_feature_extraction():
    """Complete feature extraction training with utilities."""
    
    # Step 1: Display training plan
    print("\n🎯 PHASE 1: FEATURE EXTRACTION")
    print("=" * 60)
    
    display_training_plan(
        model=model,
        epochs=3,
        learning_rate=1e-3,
        batch_size=32,
        train_samples=76000,
        val_samples=25000
    )
    
    # Step 2: Print model information
    print_model_info(model)
    
    # Step 3: Create training callbacks
    fe_callbacks = create_training_callbacks(
        model_dir="model_checkpoints",
        tensorboard_dir="training_logs",
        experiment_name="efficientnetv2b0_feature_extract",
        patience=3,
        reduce_lr_patience=2
    )
    
    # Step 4: Train model
    history_fe, fe_time = train_model(
        model=model,
        train_data=train_data,
        val_data=test_data,
        epochs=3,
        initial_epoch=0,
        callbacks=fe_callbacks,
        verbose=1
    )
    
    # Step 5: Evaluate on test set
    fe_metrics = evaluate_model(
        model=model,
        test_data=test_data,
        verbose=1
    )
    
    # Step 6: Track metrics
    metrics_tracker = TrainingMetrics(log_dir="training_metrics")
    metrics_tracker.record(history_fe)
    
    print(f"\n✅ Feature extraction phase completed in {fe_time/60:.2f} minutes")
    
    return history_fe, metrics_tracker


# ---- PHASE 2: FINE-TUNING ----

def example_2_fine_tuning(loaded_model, metrics_tracker):
    """Complete fine-tuning with gradual layer unfreezing."""
    
    print("\n🎯 PHASE 2: FINE-TUNING")
    print("=" * 60)
    
    # Step 1: Unfreeze layers for fine-tuning
    unfreeze_layers(loaded_model, num_layers=50)
    
    # Step 2: Recompile with much lower learning rate
    loaded_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"]
    )
    
    # Step 3: Display new training plan
    display_training_plan(
        model=loaded_model,
        epochs=10,
        learning_rate=1e-5,
        batch_size=32,
        train_samples=76000,
        val_samples=25000
    )
    
    # Step 4: Create callbacks for fine-tuning
    ft_callbacks = create_training_callbacks(
        model_dir="model_checkpoints",
        tensorboard_dir="training_logs",
        experiment_name="efficientnetv2b0_fine_tune",
        patience=5,
        reduce_lr_patience=3,
        reduce_lr_factor=0.2,
        min_lr=1e-7
    )
    
    # Step 5: Train with fine-tuning
    history_ft, ft_time = train_model(
        model=loaded_model,
        train_data=train_data,
        val_data=test_data,
        epochs=10,
        initial_epoch=3,
        callbacks=ft_callbacks,
        verbose=1
    )
    
    # Step 6: Evaluate fine-tuned model
    ft_metrics = evaluate_model(
        model=loaded_model,
        test_data=test_data,
        verbose=1
    )
    
    # Step 7: Track metrics from fine-tuning phase
    metrics_tracker.record(history_ft)
    metrics_tracker.save("complete_training_metrics.json")
    
    print(f"\n✅ Fine-tuning phase completed in {ft_time/60:.2f} minutes")
    
    return history_ft, loaded_model


# ---- PHASE 3: MODEL COMPARISON ----

def example_3_compare_models(fe_model, ft_model):
    """Compare feature extraction vs fine-tuned model."""
    
    print("\n🎯 PHASE 3: MODEL COMPARISON")
    print("=" * 60)
    
    comparison = compare_models(
        model1=fe_model,      # Feature extraction version
        model2=ft_model,      # Fine-tuned version
        test_data=test_data
    )
    
    # Access comparison results
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"   Winner: {comparison['winner']}")
    print(f"   Accuracy Gain: {comparison['accuracy_improvement']:+.4f}")
    print(f"   Loss Reduction: {comparison['loss_improvement']:+.4f}")
    
    return comparison


# ============================================================================
# EXAMPLE 2: Simplified Two-Phase Training
# ============================================================================

def simple_training_example():
    """Minimal code for two-phase training."""
    
    from training_utils import create_training_callbacks, train_model
    
    # PHASE 1: Feature Extraction (3 epochs, frozen base)
    callbacks1 = create_training_callbacks(experiment_name="phase1")
    history1, _ = train_model(model, train_data, test_data, epochs=3, 
                              callbacks=callbacks1)
    
    # PHASE 2: Fine-tuning (7 more epochs, unfrozen)
    unfreeze_layers(model, num_layers=50)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"]
    )
    
    callbacks2 = create_training_callbacks(experiment_name="phase2")
    history2, _ = train_model(model, train_data, test_data, epochs=10, 
                              initial_epoch=3, callbacks=callbacks2)
    
    print("✅ Training complete!")


# ============================================================================
# EXAMPLE 3: Custom Training with Progress Monitoring
# ============================================================================

def example_with_metrics_tracking():
    """Track detailed metrics across multiple training phases."""
    
    from training_utils import TrainingMetrics, create_training_callbacks, train_model
    
    # Initialize metrics tracker
    metrics = TrainingMetrics(log_dir="my_training_metrics")
    
    # Phase 1
    callbacks1 = create_training_callbacks(experiment_name="exp1")
    history1, time1 = train_model(model, train_data, test_data, epochs=5, 
                                  callbacks=callbacks1)
    metrics.record(history1)
    print(f"Phase 1 took {time1/60:.2f} minutes")
    
    # Phase 2
    unfreeze_layers(model, num_layers=50)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"]
    )
    
    callbacks2 = create_training_callbacks(experiment_name="exp2")
    history2, time2 = train_model(model, train_data, test_data, epochs=10, 
                                  initial_epoch=5, callbacks=callbacks2)
    metrics.record(history2)
    print(f"Phase 2 took {time2/60:.2f} minutes")
    
    # Save all metrics
    metrics.save("full_training_metrics.json")
    print(f"Total time: {(time1 + time2)/60:.2f} minutes")


# ============================================================================
# EXAMPLE 4: Handling Different Scenarios
# ============================================================================

def example_resume_training():
    """Resume training from a checkpoint."""
    
    from training_utils import create_training_callbacks, train_model
    import glob
    import os
    
    # Find latest checkpoint
    checkpoints = glob.glob("model_checkpoints/*.index")
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        ckpt_path = latest_ckpt.replace(".index", "")
        
        # Load weights
        model.load_weights(ckpt_path)
        print(f"✅ Loaded checkpoint from epoch")
        
        # Resume training
        callbacks = create_training_callbacks(experiment_name="resumed_training")
        history, _ = train_model(
            model=model,
            train_data=train_data,
            val_data=test_data,
            epochs=20,
            initial_epoch=5,  # Resume from epoch 5
            callbacks=callbacks
        )
    else:
        print("❌ No checkpoints found!")


def example_multi_optimizer_schedule():
    """Use different learning rates for different phases."""
    
    from training_utils import create_training_callbacks, train_model, display_training_plan
    
    # Phase 1: High learning rate
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    
    display_training_plan(model, epochs=5, learning_rate=1e-3, 
                         batch_size=32, train_samples=76000, val_samples=25000)
    callbacks1 = create_training_callbacks(experiment_name="high_lr")
    history1, _ = train_model(model, train_data, test_data, epochs=5, callbacks=callbacks1)
    
    # Phase 2: Medium learning rate
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"]
    )
    
    display_training_plan(model, epochs=10, learning_rate=1e-4, 
                         batch_size=32, train_samples=76000, val_samples=25000)
    callbacks2 = create_training_callbacks(experiment_name="medium_lr")
    history2, _ = train_model(model, train_data, test_data, epochs=10, 
                             initial_epoch=5, callbacks=callbacks2)
    
    # Phase 3: Low learning rate (fine-tuning)
    unfreeze_layers(model, num_layers=50)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"]
    )
    
    display_training_plan(model, epochs=15, learning_rate=1e-5, 
                         batch_size=32, train_samples=76000, val_samples=25000)
    callbacks3 = create_training_callbacks(experiment_name="low_lr_finetune")
    history3, _ = train_model(model, train_data, test_data, epochs=15, 
                             initial_epoch=10, callbacks=callbacks3)


# ============================================================================
# EXAMPLE 5: Debugging & Diagnostics
# ============================================================================

def example_diagnose_training_issues():
    """Tools to diagnose training problems."""
    
    from training_utils import get_model_summary_info, print_model_info
    
    # Problem 1: Model not training
    print("🔍 Checking model status...")
    
    if model.optimizer is None:
        print("❌ Model not compiled! Call model.compile() first")
    else:
        print("✅ Model is compiled")
        lr = model.optimizer.learning_rate
        print(f"   Learning rate: {lr}")
    
    # Problem 2: Too many parameters
    info = get_model_summary_info(model)
    trainable_pct = 100 * info['trainable_parameters'] / info['total_parameters']
    print(f"✅ Trainable: {trainable_pct:.1f}% of {info['total_parameters']:,} params")
    
    # Problem 3: Unexpected model structure
    print_model_info(model)


def example_check_convergence(history):
    """Check if model is converging properly."""
    
    import numpy as np
    
    val_acc = history.history['val_accuracy']
    
    # Check if improving
    recent_acc = np.mean(val_acc[-3:])
    early_acc = np.mean(val_acc[:3])
    
    improvement = recent_acc - early_acc
    
    if improvement > 0.05:
        print(f"✅ Model is improving! (+{improvement:.4f})")
    elif improvement > 0:
        print(f"⚠️  Slow improvement (+{improvement:.4f})")
    else:
        print(f"❌ Model not improving or overfitting ({improvement:+.4f})")


# ============================================================================
# EXAMPLE 6: Post-Training Analysis
# ============================================================================

def example_analyze_training_results():
    """Analyze training results after completion."""
    
    import json
    
    # Load saved metrics
    with open("training_metrics/metrics.json") as f:
        metrics = json.load(f)
    
    # Find best epoch
    val_acc = metrics['val_accuracy']
    best_epoch = val_acc.index(max(val_acc))
    best_accuracy = max(val_acc)
    
    print(f"\n📊 TRAINING ANALYSIS:")
    print(f"   Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"   Achieved at Epoch: {best_epoch}")
    
    # Check for overfitting
    final_train_acc = metrics['train_accuracy'][-1]
    final_val_acc = metrics['val_accuracy'][-1]
    overfit_gap = final_train_acc - final_val_acc
    
    if overfit_gap > 0.15:
        print(f"⚠️  Overfitting detected! Gap: {overfit_gap:.4f}")
        print("    → Try: More data augmentation, reduce epochs, or increase dropout")
    else:
        print(f"✅ Good generalization. Train-Val gap: {overfit_gap:.4f}")


# ============================================================================
# READY-TO-USE TEMPLATES
# ============================================================================

# Template 1: Minimal (Copy & paste into notebook)
"""
from training_utils import *

# Feature extraction
callbacks = create_training_callbacks(experiment_name="fe")
history1, _ = train_model(model, train_data, test_data, epochs=3, callbacks=callbacks)

# Fine-tuning
unfreeze_layers(model, 50)
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=tf.keras.optimizers.Adam(1e-5), metrics=["accuracy"])
callbacks = create_training_callbacks(experiment_name="ft")
history2, _ = train_model(model, train_data, test_data, epochs=10, 
                          initial_epoch=3, callbacks=callbacks)

print("✅ Done!")
"""

# Template 2: Professional (With all checks and reporting)
"""
from training_utils import *

# Display plan
display_training_plan(model, epochs=3, learning_rate=1e-3, batch_size=32, 
                     train_samples=76000, val_samples=25000)

# Phase 1
callbacks = create_training_callbacks(experiment_name="phase1")
history1, time1 = train_model(model, train_data, test_data, epochs=3, callbacks=callbacks)
metrics = TrainingMetrics()
metrics.record(history1)

# Phase 2
unfreeze_layers(model, 50)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(1e-5), metrics=["accuracy"])
callbacks = create_training_callbacks(experiment_name="phase2")
history2, time2 = train_model(model, train_data, test_data, epochs=10, 
                              initial_epoch=3, callbacks=callbacks)
metrics.record(history2)
metrics.save()

# Evaluation
evaluate_model(model, test_data)
compare_models(fe_model, model, test_data)

print(f"✅ Total time: {(time1 + time2)/60:.2f} minutes")
"""
