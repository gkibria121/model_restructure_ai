"""
AI vs Real Voice Detection System
===================================
A production-ready pipeline for detecting AI-generated voices using deep learning.

Usage:
    # Training
    detector = VoiceDetector(model_name='efficientnet_b0')
    detector.prepare_dataset('path/to/ai_folder', 'path/to/real_folder')
    detector.train(epochs=12)
    
    # Inference
    result = detector.predict('path/to/audio.wav')
    print(f"Prediction: {result['label']} (confidence: {result['confidence']:.2%})")
"""

import os
import sys
import json
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import soundfile as sf
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 160
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 7980.0  # sample_rate/2 - 20
    
    # Preprocessing
    pre_emphasis: float = 0.97
    hpf_cutoff: float = 40.0
    vad_frame_ms: int = 20
    vad_threshold: float = 0.5
    vad_pad_ms: int = 100
    
    # Denoising
    denoise_method: str = "auto"  # "auto", "spectral", "demucs"
    denoise_win: int = 1024
    denoise_hop: int = 256
    denoise_noise_frames: int = 20
    denoise_thresh_db: float = 6.0
    denoise_atten_db: float = 20.0


@dataclass
class ModelConfig:
    """Model training configuration"""
    model_name: str = "efficientnet_b0"  # efficientnet_b0, resnet18, convnext_tiny, etc.
    img_size: int = 224
    batch_size: int = 64
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 3
    
    # Data split
    test_size: float = 0.2
    val_from_test: float = 0.5
    random_seed: int = 42
    
    # Training
    num_workers: int = 0  # 0 for Windows/Jupyter stability
    pin_memory: bool = True
    use_amp: bool = True  # Automatic Mixed Precision


@dataclass
class PredictionResult:
    """Prediction result container"""
    file_path: str
    prediction: str  # "AI" or "Real"
    confidence: float
    prob_ai: float
    prob_real: float
    threshold: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (f"File: {Path(self.file_path).name}\n"
                f"Prediction: {self.prediction}\n"
                f"Confidence: {self.confidence:.2%}\n"
                f"AI Probability: {self.prob_ai:.4f}\n"
                f"Real Probability: {self.prob_real:.4f}")


# ============================================================================
# Audio Processing
# ============================================================================

class AudioProcessor:
    """Handles audio loading, preprocessing, and feature extraction"""
    
    def __init__(self, config: AudioConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        
        # Initialize transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0
        ).to(device)
        
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        ).to(device)
        
        # Denoising setup
        self._setup_denoising()
    
    def _setup_denoising(self):
        """Initialize denoising modules"""
        self._window = torch.hann_window(
            self.config.denoise_win, device=self.device
        )
        
        # Try to load Demucs for high-quality denoising
        self.demucs_model = None
        try:
            from demucs.pretrained import get_model as demucs_get_model
            self.demucs_model = demucs_get_model("htdemucs")
            self.demucs_model.to(self.device).eval()
            print("✓ Demucs loaded for high-quality denoising")
        except Exception:
            print("⚠ Demucs not available, using spectral gating only")
    
    @torch.no_grad()
    def load_audio(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load audio from file with fallback methods
        Returns: (1, T) tensor on device, normalized
        """
        path = str(path)
        
        # Try torchaudio
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.config.sample_rate:
                wav = torchaudio.functional.resample(
                    wav, sr, self.config.sample_rate
                )
            wav = self._normalize(wav)
            return wav.to(self.device)
        except Exception:
            pass
        
        # Try soundfile
        try:
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            if y.ndim == 2:
                y = y.mean(axis=1)
            wav = torch.from_numpy(y).unsqueeze(0)
            if sr != self.config.sample_rate:
                wav = torchaudio.functional.resample(
                    wav, sr, self.config.sample_rate
                )
            wav = self._normalize(wav)
            return wav.to(self.device)
        except Exception:
            pass
        
        # Try librosa
        try:
            y, sr = librosa.load(path, sr=self.config.sample_rate, mono=True)
            wav = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
            wav = self._normalize(wav)
            return wav.to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to load audio from {path}: {e}")
    
    @staticmethod
    def _normalize(wav: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1]"""
        mx = torch.amax(torch.abs(wav))
        if mx > 0:
            wav = wav / mx
        return wav
    
    @torch.no_grad()
    def denoise(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Denoise audio using specified method
        Args:
            wav: (1, T) tensor
        Returns:
            Denoised (1, T) tensor
        """
        method = self.config.denoise_method
        
        if method == "demucs" and self.demucs_model is not None:
            return self._denoise_demucs(wav)
        elif method == "spectral":
            return self._denoise_spectral(wav)
        else:  # auto
            if self.demucs_model is not None:
                try:
                    return self._denoise_demucs(wav)
                except Exception:
                    pass
            return self._denoise_spectral(wav)
    
    @torch.no_grad()
    def _denoise_spectral(self, wav: torch.Tensor) -> torch.Tensor:
        """Spectral gating denoising"""
        stft = torch.stft(
            wav, 
            n_fft=self.config.denoise_win,
            hop_length=self.config.denoise_hop,
            win_length=self.config.denoise_win,
            window=self._window,
            center=True,
            return_complex=True
        )
        
        mag = torch.abs(stft) + 1e-8
        phase = stft / mag
        
        # Estimate noise from first few frames
        N = min(self.config.denoise_noise_frames, mag.shape[-1])
        noise = mag[..., :N].median(dim=-1, keepdim=True).values
        
        mag_db = 20.0 * torch.log10(mag)
        noise_db = 20.0 * torch.log10(noise + 1e-8)
        
        # Soft gating
        keep = (mag_db - noise_db) >= self.config.denoise_thresh_db
        atten = 10 ** (-self.config.denoise_atten_db / 20.0)
        mag_clean = torch.where(keep, mag, mag * atten)
        
        # Reconstruct
        stft_clean = mag_clean * phase
        wav_clean = torch.istft(
            stft_clean,
            n_fft=self.config.denoise_win,
            hop_length=self.config.denoise_hop,
            win_length=self.config.denoise_win,
            window=self._window,
            center=True,
            length=wav.shape[-1]
        ).unsqueeze(0)
        
        return self._normalize(wav_clean)
    
    @torch.no_grad()
    def _denoise_demucs(self, wav: torch.Tensor) -> torch.Tensor:
        """Demucs-based denoising (vocal separation)"""
        x = wav.unsqueeze(0)  # (1, 1, T)
        out = self.demucs_model(x)
        
        if isinstance(out, (list, tuple)):
            out = out[0]
        
        # Extract vocals
        if out.dim() == 4:
            sources = getattr(
                self.demucs_model, 'sources', 
                ['drums', 'bass', 'other', 'vocals']
            )
            vocals_idx = sources.index('vocals') if 'vocals' in sources else -1
            vocals = out[:, vocals_idx, 0, :]
        elif out.dim() == 3:
            vocals = out[-1, 0, :].unsqueeze(0)
        else:
            return wav
        
        return self._normalize(vocals)
    
    @torch.no_grad()
    def vad_trim(self, wav: torch.Tensor) -> torch.Tensor:
        """Voice Activity Detection trimming"""
        T = wav.shape[-1]
        hop = int(self.config.sample_rate * self.config.vad_frame_ms / 1000.0)
        win = hop
        
        if T < win:
            return wav
        
        frames = wav.unfold(dimension=-1, size=win, step=hop)
        energy = (frames ** 2).mean(dim=-1).squeeze(0)
        
        thr = torch.median(energy) * self.config.vad_threshold
        mask = energy > thr
        
        if not mask.any():
            return wav
        
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        start_f, end_f = idx[0].item(), idx[-1].item()
        
        pad = int(self.config.sample_rate * self.config.vad_pad_ms / 1000.0)
        start = max(0, start_f * hop - pad)
        end = min(T, (end_f + 1) * hop + pad)
        
        return wav[:, start:end] if end > start else wav
    
    @torch.no_grad()
    def preprocess(self, wav: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Full preprocessing pipeline
        Returns None if audio too short after processing
        """
        # VAD trimming
        wav = self.vad_trim(wav)
        
        # High-pass filter
        if self.config.hpf_cutoff > 0:
            wav = torchaudio.functional.highpass_biquad(
                wav, self.config.sample_rate, self.config.hpf_cutoff
            )
        
        # Pre-emphasis
        if self.config.pre_emphasis > 0:
            x_shift = torch.zeros_like(wav)
            x_shift[..., 1:] = wav[..., :-1]
            wav = wav - self.config.pre_emphasis * x_shift
        
        wav = self._normalize(wav)
        
        # Check minimum length
        min_samples = self.config.hop_length * 6
        if wav.shape[-1] < min_samples:
            return None
        
        return wav
    
    @torch.no_grad()
    def wav_to_mel_spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram image tensor
        Returns: (1, H, W) tensor normalized to [0, 1]
        """
        S = self.mel_transform(wav)  # (1, n_mels, frames)
        S_db = self.amp_to_db(S).squeeze(0)  # (n_mels, frames)
        
        # Min-max normalization
        S_norm = (S_db - S_db.amin()) / (S_db.amax() - S_db.amin() + 1e-8)
        
        return S_norm.unsqueeze(0)  # (1, n_mels, frames)
    
    @torch.no_grad()
    def audio_to_model_input(
        self, 
        wav: torch.Tensor, 
        img_size: int = 224
    ) -> torch.Tensor:
        """
        Full pipeline: waveform -> model-ready tensor
        Returns: (1, 1, img_size, img_size) normalized tensor
        """
        # Get mel spectrogram
        spec = self.wav_to_mel_spectrogram(wav)  # (1, H, W)
        
        # Convert to PIL for resizing
        spec_np = (spec.squeeze(0) * 255).clamp(0, 255).to(torch.uint8)
        spec_np = spec_np.detach().cpu().numpy()
        pil_img = Image.fromarray(spec_np, mode='L')
        pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
        
        # Back to tensor and normalize (mean=0.5, std=0.5 as in training)
        img_tensor = torch.from_numpy(np.array(pil_img)).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, H, W)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor.to(self.device)


# ============================================================================
# Dataset
# ============================================================================

class SpectrogramDataset(Dataset):
    """Dataset for loading spectrogram images"""
    
    def __init__(
        self, 
        csv_path: Union[str, Path],
        img_size: int = 224,
        augment: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.augment = augment
        
        import torchvision.transforms as T
        
        if augment:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=5, 
                        translate=(0.02, 0.02),
                        scale=(0.98, 1.02)
                    )
                ], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        try:
            img = Image.open(row['path']).convert('L')
            x = self.transform(img)
        except Exception:
            # Fallback: return black image
            x = torch.zeros(1, self.img_size, self.img_size)
        
        return x, int(row['label'])


# ============================================================================
# Model Factory
# ============================================================================

class ModelFactory:
    """Factory for creating models"""
    
    @staticmethod
    def create(
        name: str, 
        num_classes: int = 2,
        pretrained: bool = False
    ) -> nn.Module:
        """Create model by name"""
        
        from torchvision.models import (
            resnet18, efficientnet_b0, convnext_tiny,
            densenet121, mobilenet_v3_small, vit_b_16, swin_t
        )
        
        name = name.lower()
        
        if name == "resnet18":
            model = resnet18(weights='DEFAULT' if pretrained else None)
            model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        elif name == "efficientnet_b0":
            model = efficientnet_b0(weights='DEFAULT' if pretrained else None)
            first = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                1, first.out_channels, 
                kernel_size=first.kernel_size,
                stride=first.stride, padding=first.padding, 
                bias=(first.bias is not None)
            )
            in_feats = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feats, num_classes)
        
        elif name == "convnext_tiny":
            model = convnext_tiny(weights='DEFAULT' if pretrained else None)
            model.features[0][0] = nn.Conv2d(1, 96, 4, stride=4, bias=True)
            model.classifier[2] = nn.Linear(
                model.classifier[2].in_features, num_classes
            )
        
        elif name == "densenet121":
            model = densenet121(weights='DEFAULT' if pretrained else None)
            first = model.features.conv0
            model.features.conv0 = nn.Conv2d(
                1, first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride, padding=first.padding,
                bias=(first.bias is not None)
            )
            model.classifier = nn.Linear(
                model.classifier.in_features, num_classes
            )
        
        elif name == "mobilenet_v3_small":
            model = mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
            first = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                1, first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride, padding=first.padding,
                bias=(first.bias is not None)
            )
            in_feats = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feats, num_classes)
        
        elif name == "vit_b_16":
            model = vit_b_16(weights='DEFAULT' if pretrained else None)
            embed_dim = model.conv_proj.out_channels
            model.conv_proj = nn.Conv2d(1, embed_dim, 16, stride=16, bias=True)
            try:
                in_features = model.heads.head.in_features
                model.heads.head = nn.Linear(in_features, num_classes)
            except:
                last = list(model.heads.children())[-1]
                model.heads = nn.Sequential(nn.Linear(last.in_features, num_classes))
        
        elif name == "swin_t":
            model = swin_t(weights='DEFAULT' if pretrained else None)
            old = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                1, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding,
                bias=(old.bias is not None)
            )
            model.head = nn.Linear(model.head.in_features, num_classes)
        
        else:
            raise ValueError(f"Unknown model: {name}")
        
        return model


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Model training and evaluation"""
    
    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=(device == "cuda" and config.use_amp)
        )
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'train_acc': [], 
                       'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for x, y in pbar:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(
                device_type='cuda', 
                enabled=(self.device == "cuda" and self.config.use_amp)
            ):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
            pbar.set_postfix({
                'loss': total_loss / total,
                'acc': correct / total
            })
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Validate model"""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_probs, all_targets = [], []
        
        for x, y in tqdm(loader, desc="Validating", leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with torch.amp.autocast(
                device_type='cuda',
                enabled=(self.device == "cuda" and self.config.use_amp)
            ):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            
            probs = F.softmax(logits, dim=1)[:, 1]
            
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
        
        avg_loss = total_loss / total
        accuracy = correct / total
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        
        return avg_loss, accuracy, probs, targets
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Union[str, Path]
    ):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Training {self.config.model_name}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch:02d}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Saved best model (acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered (patience: {self.config.patience})")
                break
        
        print(f"\nTraining complete! Best val acc: {self.best_val_acc:.4f}")
    
    def plot_history(self, save_path: Optional[Union[str, Path]] = None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# ============================================================================
# Main Interface
# ============================================================================

class VoiceDetector:
    """
    Main interface for AI voice detection
    
    Example:
        # Training
        detector = VoiceDetector(model_name='efficientnet_b0')
        detector.prepare_dataset('ai_folder', 'real_folder', 'output_dir')
        detector.train()
        
        # Inference
        result = detector.predict('audio.wav')
        print(result)
    """
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        device: Optional[str] = None,
        audio_config: Optional[AudioConfig] = None,
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize detector
        
        Args:
            model_name: Model architecture name
            device: 'cuda' or 'cpu', auto-detected if None
            audio_config: Audio processing config
            model_config: Model training config
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.audio_config = audio_config or AudioConfig()
        self.model_config = model_config or ModelConfig(model_name=model_name)
        
        self.audio_processor = AudioProcessor(self.audio_config, self.device)
        self.model = None
        self.threshold = 0.5
        
        self.output_dir = None
        self.splits_dir = None
        self.models_dir = None
    
    def prepare_dataset(
        self,
        ai_folder: Union[str, Path],
        real_folder: Union[str, Path],
        output_dir: Union[str, Path],
        denoise: bool = True,
        force_rebuild: bool = False
    ):
        """
        Prepare dataset from raw audio files
        
        Args:
            ai_folder: Path to AI voice samples
            real_folder: Path to real voice samples
            output_dir: Output directory for processed data
            denoise: Whether to denoise audio
            force_rebuild: Force rebuild even if exists
        """
        print("\n" + "="*60)
        print("DATASET PREPARATION")
        print("="*60 + "\n")
        
        self.output_dir = Path(output_dir)
        clean_audio_dir = self.output_dir / "clean_audio"
        spectrogram_dir = self.output_dir / "spectrograms"
        self.splits_dir = self.output_dir / "splits"
        self.models_dir = self.output_dir / "models"
        
        for d in [clean_audio_dir, spectrogram_dir, self.splits_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        ai_clean_dir = clean_audio_dir / "ai"
        real_clean_dir = clean_audio_dir / "real"
        ai_spec_dir = spectrogram_dir / "ai"
        real_spec_dir = spectrogram_dir / "real"
        
        for d in [ai_clean_dir, real_clean_dir, ai_spec_dir, real_spec_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Process audio files
        if force_rebuild or not list(ai_clean_dir.glob("*.wav")):
            print("Step 1: Processing audio files...")
            self._process_audio_folder(
                ai_folder, ai_clean_dir, denoise, "AI"
            )
            self._process_audio_folder(
                real_folder, real_clean_dir, denoise, "Real"
            )
        else:
            print("Step 1: Using existing audio files")
        
        # Step 2: Create spectrograms
        if force_rebuild or not list(ai_spec_dir.glob("*.png")):
            print("\nStep 2: Creating spectrograms...")
            self._create_spectrograms(ai_clean_dir, ai_spec_dir, "AI")
            self._create_spectrograms(real_clean_dir, real_spec_dir, "Real")
        else:
            print("\nStep 2: Using existing spectrograms")
        
        # Step 3: Create splits
        print("\nStep 3: Creating train/val/test splits...")
        self._create_splits(ai_spec_dir, real_spec_dir)
        
        print("\n✓ Dataset preparation complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Splits saved to: {self.splits_dir}")
    
    def _process_audio_folder(
        self,
        input_dir: Union[str, Path],
        output_dir: Path,
        denoise: bool,
        label: str
    ):
        """Process all audio files in a folder"""
        audio_files = self._list_audio_files(input_dir)
        print(f"  Processing {len(audio_files)} {label} files...")
        
        skipped = 0
        for audio_path in tqdm(audio_files, desc=f"  {label}"):
            output_path = output_dir / f"{Path(audio_path).stem}.wav"
            
            if output_path.exists():
                continue
            
            try:
                # Load and process
                wav = self.audio_processor.load_audio(audio_path)
                
                if denoise:
                    wav = self.audio_processor.denoise(wav)
                
                wav = self.audio_processor.preprocess(wav)
                
                if wav is None:
                    skipped += 1
                    continue
                
                # Save
                wav_np = wav.squeeze(0).cpu().numpy()
                sf.write(
                    str(output_path),
                    wav_np,
                    self.audio_config.sample_rate,
                    subtype='PCM_16'
                )
            except Exception as e:
                skipped += 1
        
        print(f"    Processed: {len(audio_files) - skipped}, Skipped: {skipped}")
    
    def _create_spectrograms(
        self,
        input_dir: Path,
        output_dir: Path,
        label: str
    ):
        """Create spectrogram images from audio files"""
        wav_files = list(input_dir.glob("*.wav"))
        print(f"  Creating {len(wav_files)} {label} spectrograms...")
        
        skipped = 0
        for wav_path in tqdm(wav_files, desc=f"  {label}"):
            output_path = output_dir / f"{wav_path.stem}.png"
            
            if output_path.exists():
                continue
            
            try:
                wav = self.audio_processor.load_audio(wav_path)
                spec = self.audio_processor.wav_to_mel_spectrogram(wav)
                
                # Convert to uint8 image
                spec_np = (spec.squeeze(0) * 255).clamp(0, 255)
                spec_np = spec_np.to(torch.uint8).cpu().numpy()
                
                img = Image.fromarray(spec_np, mode='L')
                img.save(output_path, 'PNG')
            except Exception:
                skipped += 1
        
        print(f"    Created: {len(wav_files) - skipped}, Skipped: {skipped}")
    
    def _create_splits(self, ai_dir: Path, real_dir: Path):
        """Create train/val/test splits"""
        ai_files = sorted(ai_dir.glob("*.png"))
        real_files = sorted(real_dir.glob("*.png"))
        
        paths = [str(f) for f in ai_files + real_files]
        labels = [1] * len(ai_files) + [0] * len(real_files)
        
        # 80/10/10 split
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels,
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_seed,
            stratify=labels
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=self.model_config.val_from_test,
            random_state=self.model_config.random_seed,
            stratify=temp_labels
        )
        
        # Save CSVs
        for name, p, l in [
            ('train', train_paths, train_labels),
            ('val', val_paths, val_labels),
            ('test', test_paths, test_labels)
        ]:
            df = pd.DataFrame({'path': p, 'label': l})
            df.to_csv(self.splits_dir / f"{name}.csv", index=False)
        
        print(f"    Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    
    @staticmethod
    def _list_audio_files(folder: Union[str, Path]) -> List[Path]:
        """List all audio files in folder"""
        folder = Path(folder)
        exts = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma']
        files = []
        for ext in exts:
            files.extend(folder.rglob(f"*{ext}"))
            files.extend(folder.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    
    def train(self, resume: bool = False):
        """
        Train the model
        
        Args:
            resume: Resume from checkpoint if exists
        """
        if self.splits_dir is None:
            raise ValueError("Call prepare_dataset() first")
        
        print("\n" + "="*60)
        print(f"TRAINING {self.model_config.model_name.upper()}")
        print("="*60 + "\n")
        
        # Create dataloaders
        train_ds = SpectrogramDataset(
            self.splits_dir / "train.csv",
            img_size=self.model_config.img_size,
            augment=True
        )
        val_ds = SpectrogramDataset(
            self.splits_dir / "val.csv",
            img_size=self.model_config.img_size,
            augment=False
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.model_config.batch_size,
            shuffle=True,
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory and (self.device == "cuda")
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=self.model_config.num_workers,
            pin_memory=self.model_config.pin_memory and (self.device == "cuda")
        )
        
        # Create model
        self.model = ModelFactory.create(self.model_config.model_name)
        
        model_path = self.models_dir / f"best_{self.model_config.model_name}.pt"
        
        if resume and model_path.exists():
            print(f"Resuming from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Train
        trainer = Trainer(self.model, self.model_config, self.device)
        trainer.fit(train_loader, val_loader, model_path)
        
        # Plot history
        history_plot = self.models_dir / f"{self.model_config.model_name}_history.png"
        trainer.plot_history(history_plot)
        
        # Load best model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"\n✓ Model saved to: {model_path}")
    
    def tune_threshold(self) -> float:
        """
        Tune decision threshold on validation set using Youden's J statistic
        
        Returns:
            Optimal threshold
        """
        if self.model is None:
            raise ValueError("Train or load model first")
        
        print("\nTuning threshold on validation set...")
        
        val_ds = SpectrogramDataset(
            self.splits_dir / "val.csv",
            img_size=self.model_config.img_size,
            augment=False
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Collect predictions
        all_probs, all_targets = [], []
        self.model.eval()
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Evaluating"):
                x = x.to(self.device)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)[:, 1]
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y.numpy())
        
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        
        # Find optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(targets, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        self.threshold = float(thresholds[best_idx])
        
        print(f"✓ Optimal threshold: {self.threshold:.4f}")
        print(f"  TPR: {tpr[best_idx]:.4f}, FPR: {fpr[best_idx]:.4f}")
        
        return self.threshold
    
    def evaluate(self, plot: bool = True) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            plot: Whether to plot ROC and confusion matrix
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Train or load model first")
        
        print("\n" + "="*60)
        print("EVALUATION ON TEST SET")
        print("="*60 + "\n")
        
        test_ds = SpectrogramDataset(
            self.splits_dir / "test.csv",
            img_size=self.model_config.img_size,
            augment=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Collect predictions
        all_probs, all_targets = [], []
        self.model.eval()
        
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Testing"):
                x = x.to(self.device)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)[:, 1]
                
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y.numpy())
        
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        preds = (probs >= self.threshold).astype(int)
        
        # Calculate metrics
        accuracy = (preds == targets).mean()
        roc_auc = roc_auc_score(targets, probs)
        
        print(f"Threshold: {self.threshold:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.6f}")
        print("\n" + classification_report(
            targets, preds, 
            target_names=['Real', 'AI']
        ))
        
        cm = confusion_matrix(targets, preds)
        print("Confusion Matrix:")
        print(f"              Predicted")
        print(f"           Real    AI")
        print(f"Actual Real  {cm[0,0]:<6} {cm[0,1]:<6}")
        print(f"       AI    {cm[1,0]:<6} {cm[1,1]:<6}")
        
        # Plot
        if plot:
            self._plot_evaluation(targets, probs, preds)
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'threshold': self.threshold
        }
    
    def _plot_evaluation(self, targets, probs, preds):
        """Plot ROC curve and confusion matrix"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
        ax1.plot([0, 1], [0, 1], 'k--', lw=1)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {self.model_config.model_name}')
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(targets, preds)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'AI'],
            yticklabels=['Real', 'AI'],
            ax=ax2
        )
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        ax2.set_title(f'Confusion Matrix (threshold={self.threshold:.3f})')
        
        plt.tight_layout()
        
        eval_plot = self.models_dir / f"{self.model_config.model_name}_evaluation.png"
        plt.savefig(eval_plot, dpi=150, bbox_inches='tight')
        print(f"\n✓ Evaluation plot saved to: {eval_plot}")
        plt.show()
    
    def predict(
        self,
        audio_path: Union[str, Path],
        return_probs: bool = False
    ) -> Union[PredictionResult, Dict]:
        """
        Predict if audio is AI or Real
        
        Args:
            audio_path: Path to audio file
            return_probs: Return full probability distribution
            
        Returns:
            PredictionResult object or dict with probabilities
        """
        if self.model is None:
            raise ValueError("Train or load model first")
        
        self.model.eval()
        
        # Process audio
        wav = self.audio_processor.load_audio(audio_path)
        wav = self.audio_processor.preprocess(wav)
        
        if wav is None:
            raise ValueError("Audio too short after preprocessing")
        
        x = self.audio_processor.audio_to_model_input(
            wav, self.model_config.img_size
        )
        
        # Predict
        with torch.no_grad():
            with torch.amp.autocast(
                device_type='cuda',
                enabled=(self.device == "cuda")
            ):
                logits = self.model(x)
                probs = F.softmax(logits, dim=1).squeeze(0)
        
        prob_real = probs[0].item()
        prob_ai = probs[1].item()
        
        prediction = "AI" if prob_ai >= self.threshold else "Real"
        confidence = max(prob_ai, prob_real)
        
        result = PredictionResult(
            file_path=str(audio_path),
            prediction=prediction,
            confidence=confidence,
            prob_ai=prob_ai,
            prob_real=prob_real,
            threshold=self.threshold
        )
        
        return result.to_dict() if return_probs else result
    
    def predict_batch(
        self,
        audio_paths: List[Union[str, Path]],
        batch_size: int = 32
    ) -> List[PredictionResult]:
        """
        Predict multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for audio_path in tqdm(audio_paths, desc="Predicting"):
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append(None)
        
        return results
    
    def save(self, path: Union[str, Path]):
        """
        Save model and configuration
        
        Args:
            path: Path to save directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model is not None:
            torch.save(
                self.model.state_dict(),
                path / "model.pt"
            )
        
        # Save config
        config = {
            'model_name': self.model_config.model_name,
            'threshold': self.threshold,
            'audio_config': asdict(self.audio_config),
            'model_config': asdict(self.model_config)
        }
        
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Export to TorchScript
        if self.model is not None:
            example = torch.randn(
                1, 1, 
                self.model_config.img_size,
                self.model_config.img_size,
                device=self.device
            )
            self.model.eval()
            ts_model = torch.jit.trace(self.model, example)
            ts_model.save(str(path / "model_torchscript.pt"))
        
        print(f"✓ Model saved to: {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None):
        """
        Load model and configuration
        
        Args:
            path: Path to saved model directory
            device: Device to load model on
            
        Returns:
            VoiceDetector instance
        """
        path = Path(path)
        
        # Load config
        with open(path / "config.json", 'r') as f:
            config = json.load(f)
        
        # Create instance
        detector = cls(
            model_name=config['model_name'],
            device=device,
            audio_config=AudioConfig(**config['audio_config']),
            model_config=ModelConfig(**config['model_config'])
        )
        
        detector.threshold = config['threshold']
        
        # Load model
        model_path = path / "model.pt"
        if model_path.exists():
            detector.model = ModelFactory.create(config['model_name'])
            detector.model.load_state_dict(
                torch.load(model_path, map_location=detector.device)
            )
            detector.model.to(detector.device)
            detector.model.eval()
            print(f"✓ Model loaded from: {path}")
        
        return detector


# ============================================================================
# CLI Interface (Optional)
# ============================================================================

def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Voice Detector - Train and predict"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--ai-folder', required=True, help='AI audio folder')
    train_parser.add_argument('--real-folder', required=True, help='Real audio folder')
    train_parser.add_argument('--output-dir', required=True, help='Output directory')
    train_parser.add_argument('--model', default='efficientnet_b0', help='Model name')
    train_parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')
    train_parser.add_argument('--no-denoise', action='store_true', help='Skip denoising')
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict audio file')
    pred_parser.add_argument('--model-path', required=True, help='Path to saved model')
    pred_parser.add_argument('--audio', required=True, help='Audio file path')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        detector = VoiceDetector(model_name=args.model)
        detector.prepare_dataset(
            args.ai_folder,
            args.real_folder,
            args.output_dir,
            denoise=not args.no_denoise
        )
        detector.train()
        detector.tune_threshold()
        detector.evaluate()
        
        # Save model
        save_path = Path(args.output_dir) / "final_model"
        detector.save(save_path)
    
    elif args.command == 'predict':
        detector = VoiceDetector.load(args.model_path)
        result = detector.predict(args.audio)
        print("\n" + "="*60)
        print(result)
        print("="*60)


if __name__ == "__main__":
    main()


# ============================================================================
# Example Usage
# ============================================================================

"""
# Example 1: Quick Training
detector = VoiceDetector(model_name='efficientnet_b0')
detector.prepare_dataset(
    ai_folder='data/ai_voices',
    real_folder='data/real_voices',
    output_dir='output'
)
detector.train()
detector.tune_threshold()
detector.evaluate()
detector.save('models/voice_detector')

# Example 2: Load and Predict
detector = VoiceDetector.load('models/voice_detector')
result = detector.predict('test_audio.wav')
print(result)

# Example 3: Batch Prediction
audio_files = ['audio1.wav', 'audio2.wav', 'audio3.wav']
results = detector.predict_batch(audio_files)
for result in results:
    if result:
        print(f"{result.file_path}: {result.prediction} ({result.confidence:.2%})")

# Example 4: Custom Configuration
from pathlib import Path

audio_cfg = AudioConfig(
    sample_rate=16000,
    n_mels=128,
    denoise_method='auto'
)

model_cfg = ModelConfig(
    model_name='resnet18',
    batch_size=32,
    epochs=20,
    learning_rate=1e-4
)

detector = VoiceDetector(
    model_name='resnet18',
    audio_config=audio_cfg,
    model_config=model_cfg
)

# Example 5: CLI Usage
# python script.py train --ai-folder data/ai --real-folder data/real --output-dir output --model efficientnet_b0
# python script.py predict --model-path models/voice_detector --audio test.wav
"""