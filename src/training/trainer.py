"""
Training loop for SIEVE models.

This module implements the training infrastructure including:
- Training and validation loops
- Model checkpointing
- Early stopping
- Learning rate scheduling
- Metrics tracking

Author: Lescai Lab
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

from .loss import SIEVELoss


class Trainer:
    """
    Trainer for SIEVE models.

    Handles training loop, validation, checkpointing, and early stopping.

    Args:
        model: SIEVE model to train
        optimizer: PyTorch optimizer
        loss_fn: Loss function (SIEVELoss instance)
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        scheduler: Optional learning rate scheduler
        early_stopping_patience: Number of epochs without improvement before stopping (default: 10)
        gradient_clip_value: Maximum gradient norm for clipping (default: None)

    Attributes:
        model: The model being trained
        optimizer: The optimizer
        loss_fn: Loss function
        device: Training device
        checkpoint_dir: Path to checkpoint directory
        scheduler: Learning rate scheduler
        early_stopping_patience: Early stopping patience
        gradient_clip_value: Gradient clipping value
        history: Training history (losses, metrics per epoch)
        best_val_auc: Best validation AUC achieved
        epochs_without_improvement: Counter for early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: SIEVELoss,
        device: str,
        checkpoint_dir: Path,
        scheduler: Optional[_LRScheduler] = None,
        early_stopping_patience: int = 10,
        gradient_clip_value: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_value = gradient_clip_value

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_classification_loss': [],
            'train_attribution_loss': [],
            'train_auc': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_classification_loss': [],
            'val_attribution_loss': [],
            'val_auc': [],
            'val_accuracy': [],
            'learning_rate': [],
        }
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        self.current_epoch = 0

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics (loss, AUC, accuracy)
        """
        self.model.train()

        all_losses = []
        all_classification_losses = []
        all_attribution_losses = []
        all_preds = []
        all_labels = []

        for batch in train_loader:
            # Move batch to device
            features = batch['features'].to(self.device)
            positions = batch['positions'].to(self.device)
            gene_ids = batch['gene_ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.loss_fn.lambda_attr > 0:
                # Need variant embeddings for attribution loss
                logits, intermediates = self.model(
                    features, positions, gene_ids, mask,
                    return_intermediate=True
                )
                variant_embeddings = intermediates['variant_embeddings']
                loss_dict = self.loss_fn(
                    logits=logits,
                    labels=labels,
                    variant_embeddings=variant_embeddings,
                    mask=mask,
                )
            else:
                # Standard forward pass
                logits, _ = self.model(features, positions, gene_ids, mask)
                loss_dict = self.loss_fn(logits=logits, labels=labels)

            loss = loss_dict['total']

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )

            self.optimizer.step()

            # Track metrics
            all_losses.append(loss.item())
            all_classification_losses.append(loss_dict['classification'].item())
            all_attribution_losses.append(loss_dict['attribution_sparsity'].item())

            # Store predictions and labels for AUC
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        # Compute epoch metrics
        train_loss = np.mean(all_losses)
        train_classification_loss = np.mean(all_classification_losses)
        train_attribution_loss = np.mean(all_attribution_losses)
        train_auc = roc_auc_score(all_labels, all_preds)
        train_accuracy = accuracy_score(all_labels, np.array(all_preds) > 0.5)

        return {
            'loss': train_loss,
            'classification_loss': train_classification_loss,
            'attribution_loss': train_attribution_loss,
            'auc': train_auc,
            'accuracy': train_accuracy,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics (loss, AUC, accuracy)
        """
        self.model.eval()

        all_losses = []
        all_classification_losses = []
        all_attribution_losses = []
        all_preds = []
        all_labels = []

        for batch in val_loader:
            # Move batch to device
            features = batch['features'].to(self.device)
            positions = batch['positions'].to(self.device)
            gene_ids = batch['gene_ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            if self.loss_fn.lambda_attr > 0:
                logits, intermediates = self.model(
                    features, positions, gene_ids, mask,
                    return_intermediate=True
                )
                variant_embeddings = intermediates['variant_embeddings']
                loss_dict = self.loss_fn(
                    logits=logits,
                    labels=labels,
                    variant_embeddings=variant_embeddings,
                    mask=mask,
                )
            else:
                logits, _ = self.model(features, positions, gene_ids, mask)
                loss_dict = self.loss_fn(logits=logits, labels=labels)

            loss = loss_dict['total']

            # Track metrics
            all_losses.append(loss.item())
            all_classification_losses.append(loss_dict['classification'].item())
            all_attribution_losses.append(loss_dict['attribution_sparsity'].item())

            # Store predictions and labels for AUC
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        val_loss = np.mean(all_losses)
        val_classification_loss = np.mean(all_classification_losses)
        val_attribution_loss = np.mean(all_attribution_losses)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_accuracy = accuracy_score(all_labels, np.array(all_preds) > 0.5)

        return {
            'loss': val_loss,
            'classification_loss': val_classification_loss,
            'attribution_loss': val_attribution_loss,
            'auc': val_auc,
            'accuracy': val_accuracy,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            verbose: Whether to print progress (default: True)

        Returns:
            Training history dictionary

        Raises:
            KeyboardInterrupt: If training is interrupted by user
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_classification_loss'].append(train_metrics['classification_loss'])
            self.history['train_attribution_loss'].append(train_metrics['attribution_loss'])
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])

            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_classification_loss'].append(val_metrics['classification_loss'])
            self.history['val_attribution_loss'].append(val_metrics['attribution_loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                      f"AUC: {train_metrics['auc']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"AUC: {val_metrics['auc']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}")
                if self.loss_fn.lambda_attr > 0:
                    print(f"  Attribution Loss - Train: {train_metrics['attribution_loss']:.4f}, "
                          f"Val: {val_metrics['attribution_loss']:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['auc'])

            # Checkpointing
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt', val_metrics)
                if verbose:
                    print(f"  → New best model (AUC: {self.best_val_auc:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Save last checkpoint
            self.save_checkpoint('last_model.pt', val_metrics)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping after {epoch+1} epochs "
                          f"({self.early_stopping_patience} epochs without improvement)")
                break

        return self.history

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            metrics: Current metrics to save with checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_auc': self.best_val_auc,
            'history': self.history,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename: str) -> Dict[str, float]:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            Metrics from the loaded checkpoint

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False  # We save metrics and history, not just weights
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.history = checkpoint['history']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['metrics']
