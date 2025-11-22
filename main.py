# Voice Speaker Recognition - Main Training Pipeline
"""
This module orchestrates the complete end-to-end training pipeline for the EchoID system.
It handles the full lifecycle of the machine learning process, from data loading to
final model evaluation.

Pipeline Steps:
---------------
1. Dataset Loading: Loads audio files and splits them into training/testing sets.
2. Waveform Augmentation: Applies noise, pitch shifts, and scaling to raw audio.
3. Feature Extraction: Converts augmented waveforms into Mel-spectrograms.
4. Mel Augmentation: Applies SpecAugment and time/frequency masking to spectrograms.
5. Model Training: Constructs and trains the CNN model with configured callbacks.
6. Evaluation: Calculates performance metrics on the unseen test set.

Usage:
------
    python main.py

Name: EchoID
Author: Muhd Uwais
Project: Deep Voice Speaker Recognition CNN
Purpose: Main Execution (Training Pipeline)
License: MIT
"""

from src.data.dataset_loader import AudioDatasetLoader
from src.data.audio_augmentor import AudioAugmentor
from src.data.mel_processor import WaveformToMel, MelAugmentor
from src.models.trainer import Trainer
from src.models.evaluation import evaluate_model


# =============================================================
# Main Training Pipeline
# =============================================================

def main():
    """
    Execute the full training and evaluation workflow.

    This function connects all data processing and modeling components:
    Loader -> Audio Augmentor -> Mel Processor -> Mel Augmentor -> Trainer -> Evaluator
    """

    # ------------------ 1. Load Dataset ------------------
    loader = AudioDatasetLoader(file_path="data")
    x_train, x_test, y_train, y_test = loader.load_dataset()

    # ------------------ 2. Waveform Augmentation ------------------
    audio_augmentor = AudioAugmentor(sr=16000, batch_size=32)

    # Increase training data diversity (3x augmentation factor)
    x_train_aug, y_train_aug = audio_augmentor.run(
        x_train, y_train, num_aug=3, shuffle=True
    )

    # ------------------ 3. Mel-Spectrogram Conversion ------------------
    mel_processor = WaveformToMel()

    # Convert both training and testing sets
    x_train_mel, y_train_mel = mel_processor.run(x_train_aug, y_train_aug)
    x_test_mel, y_test_mel = mel_processor.run(x_test, y_test)

    # ------------------ 4. Mel-Level Augmentation ------------------
    mel_augmentor = MelAugmentor()

    # Apply additional augmentation on features (2x augmentation factor)
    x_train_mel_aug, y_train_mel_aug = mel_augmentor.run(
        x_train_mel, y_train_mel, num_aug=2, shuffle=True
    )

    # ------------------ 5. Model Training ------------------
    trainer = Trainer()

    # Train the model (Checkpoints and logs are handled by the Trainer class)
    model, history = trainer.train(
        x_train_mel_aug, y_train_mel_aug, verbose=1, shuffle=True
    )

    # ------------------ 6. Model Evaluation ------------------
    metrics = evaluate_model(model, history, x_test_mel, y_test_mel)

    # Display final performance report
    print("\n🏆 Final Evaluation Metrics:")
    print(f"   • Accuracy:   {metrics['accuracy']:.4f}")
    print(f"   • Precision:  {metrics['precision']:.4f}")
    print(f"   • Recall:     {metrics['recall']:.4f}")
    print(f"   • F1-Score:   {metrics['f1_score']:.4f}")
    print(f"   • ROC-AUC:    {metrics['roc_auc']:.4f}")


# ---------------------------------------------------------
# Script Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🟢 Running Main Script...")
    main()
    print("🏁 Main Script Finished Successfully.")
