# Food Vision 101 🍕

A deep learning project that classifies food images into 101 different food categories using TensorFlow and EfficientNetV2B0 transfer learning.

## Project Overview

Food Vision 101 is a computer vision project that leverages transfer learning to classify images of food into 101 distinct categories. The project uses the **Food101** dataset from TensorFlow Datasets and implements both feature extraction and fine-tuning approaches with the EfficientNetV2B0 pre-trained model.

### Key Features

- **Transfer Learning**: Uses pre-trained EfficientNetV2B0 model for efficient feature extraction
- **Data Augmentation**: Implements random flipping, rotation, zoom, and height/width adjustments
- **Two-Stage Training**: 
  - Feature extraction (frozen base model weights)
  - Fine-tuning (unfrozen base model weights)
- **Model Checkpointing**: Saves best model weights based on validation accuracy
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Learning Rate Reduction**: Dynamically reduces learning rate when validation accuracy plateaus
- **TensorBoard Logging**: Track training metrics and visualize model performance

## Dataset

- **Source**: TensorFlow Datasets (Food101)
- **Classes**: 101 food categories
- **Training Samples**: ~76,000 images
- **Validation Samples**: ~25,000 images
- **Image Size**: 224×224 pixels (resized during preprocessing)
- **Color Channels**: RGB (3 channels)

## Project Structure

```
food_vision_101/
├── README.md                                          # This file
├── food_vison_101.ipynb                              # Main Jupyter notebook
├── helper.py                                          # Helper functions and utilities
├── __pycache__/                                       # Python cache files
├── model_checkpoints/                                 # Saved model checkpoints
│   └── model_epoch_XX_val_accX.XX.ckpt.*            # Checkpoint files
├── efficientnetv2b0_feature_extract_model_precision/ # Feature extraction model
│   ├── saved_model.pb
│   ├── keras_metadata.pb
│   ├── fingerprint.pb
│   ├── assets/
│   └── variables/
├── efficientnetv2b0_fine_tune_model_precision/       # Fine-tuned model
│   ├── saved_model.pb
│   ├── keras_metadata.pb
│   ├── fingerprint.pb
│   ├── assets/
│   └── variables/
├── training_logs/                                     # TensorBoard logs (feature extraction)
└── training_fine_logs/                                # TensorBoard logs (fine-tuning)
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- TensorFlow Datasets
- TensorFlow Hub
- NumPy
- Matplotlib
- Pandas

Install dependencies:
```bash
pip install tensorflow tensorflow-datasets tensorflow-hub numpy matplotlib pandas
```

## Usage

### Running the Notebook

1. Open `food_vison_101.ipynb` in Jupyter Notebook or Jupyter Lab
2. Follow the notebook cells sequentially to:
   - Load the Food101 dataset
   - Preprocess and batch the data
   - Create the model architecture
   - Train the feature extraction model
   - Fine-tune the model
   - Evaluate and visualize results

### Key Functions from helper.py

#### Data Utilities
- `walk_through_directory(dir_path)`: Display directory structure and image counts
- `view_random_image(target_dir, target_class)`: Display a random image from a class

#### Visualization
- `plot_loss_curves(history)`: Plot training/validation loss and accuracy curves
- `load_and_prep_image(filename, image_shape=224)`: Load and preprocess images for prediction
- `pred_and_plot(model, filename, class_names)`: Make predictions and visualize results

#### Model Training
- `create_tensorboard_callback(dir_name, experiment_name)`: Create TensorBoard logging callback

### Data Preprocessing Pipeline

The project implements an efficient data pipeline:

```
Original Data → Preprocessing → Shuffling → Batching → Prefetching
```

**Preprocessing Steps**:
1. Resize images to 224×224 pixels
2. Convert datatype from uint8 to float32/float64
3. Apply data augmentation (during training only)

**Data Augmentation**:
- Random horizontal flips
- Random rotations (0.2)
- Random zoom (0.2)
- Random width adjustments (0.2)
- Random height adjustments (0.2)

### Model Architecture

**Base Model**: EfficientNetV2B0 (pre-trained on ImageNet)

**Feature Extraction Model**:
- Input Layer: (224, 224, 3)
- Data Augmentation
- EfficientNetV2B0 Base (frozen weights)
- Global Average Pooling 2D
- Dense Output Layer (101 classes, softmax activation)

**Callbacks Used**:
- **ModelCheckpoint**: Saves best weights based on validation accuracy
- **EarlyStopping**: Stops training if no improvement for 5 epochs
- **ReduceLROnPlateau**: Reduces learning rate by 20% after 3 epochs of no improvement

## Training Details

### Feature Extraction Phase
- **Model Weights**: Base model frozen
- **Trainable Parameters**: Only top layers (Global Average Pooling + Dense)
- **Epochs**: 3
- **Batch Size**: 32
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam

### Fine-Tuning Phase (if applied)
- **Model Weights**: Base model unfrozen (fully trainable)
- **Learning Rate**: Lower learning rate (typically 1e-5 to 1e-4)
- **Epochs**: Additional training rounds
- **Strategy**: Gradual unfreezing of base model layers

## Model Performance

The trained models are saved in two formats:

1. **Feature Extraction Model** (`efficientnetv2b0_feature_extract_model_precision/`)
   - Baseline performance with frozen base model weights
   - Faster training, good baseline accuracy

2. **Fine-Tuned Model** (`efficientnetv2b0_fine_tune_model_precision/`)
   - Improved performance with unfrozen base model weights
   - Longer training time, typically higher accuracy

Model checkpoints are saved as:
```
model_checkpoints/model_epoch_XX_val_accX.XX.ckpt
```

## TensorBoard Visualization

Monitor training progress with TensorBoard:

```bash
# For feature extraction training
tensorboard --logdir training_logs

# For fine-tuning training
tensorboard --logdir training_fine_logs
```

Then navigate to `http://localhost:6006` in your browser.

## Making Predictions

Use the trained model to classify new food images:

```python
from helper import pred_and_plot, load_and_prep_image

# Load model
model = tf.keras.models.load_model('efficientnetv2b0_fine_tune_model_precision')

# Make prediction on a single image
pred_and_plot(model, 'path/to/image.jpg', class_names)
```

## Key Insights

- **Transfer Learning Efficiency**: Using a pre-trained model significantly reduces training time while achieving good accuracy
- **Data Augmentation Impact**: Augmentation helps improve model generalization
- **Mixed Precision Training**: Improves training speed with float16 computations
- **Batch Processing**: Critical for handling 101,000+ images efficiently
- **Learning Rate Scheduling**: Dynamically reducing LR helps fine-tune model performance

## Notes

- The dataset is automatically downloaded from TensorFlow Datasets on first run
- Mixed precision training (float16) can be enabled for faster training on compatible hardware
- Image shapes in the Food101 dataset vary; preprocessing resizes them to a consistent 224×224
- The notebook includes visualization examples for sample images and training curves

## Future Improvements

- Implement ensemble methods combining multiple models
- Experiment with different architectures (EfficientNetV2-L, ViT)
- Add confidence thresholding for uncertain predictions
- Create a web API for real-time food classification
- Implement explainability features (attention maps, GradCAM)

## References

- [TensorFlow Food101 Dataset](https://www.tensorflow.org/datasets/catalog/food101)
- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [TensorFlow Data Performance Guide](https://www.tensorflow.org/guide/data_performance)

## License

This project is for educational purposes.

## Author

Created as part of TensorFlow deep learning studies.
