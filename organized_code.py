"""
AI vs Real Voice Detection System
Organized with proper OOP structure
"""

import os 
import random 
import glob
import hashlib
import warnings
from typing import List, Dict, Tuple, Optional 
  
import numpy as np
import pandas as pd
import librosa 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, auc
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import torchvision.transforms as T
from torchvision.models import (
    resnet18, efficientnet_b0, efficientnet_b1,
    densenet121, densenet169, mobilenet_v3_small,
    convnext_tiny, vit_b_16, swin_t
)

# Critical stability settings
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '0'
# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the entire pipeline"""
    
    # Paths
    COMBINED_DIR = r"./dataset"
    AI_DIR = os.path.join(COMBINED_DIR, "ai")
    REAL_DIR = os.path.join(COMBINED_DIR, "real")
    CLEAN_DIR = os.path.join(COMBINED_DIR, "clean_audio_unique")
    IMG_DIR = os.path.join(COMBINED_DIR, "spectrogram_images_clean_unique")
    SPLIT_DIR = os.path.join(IMG_DIR, "_splits")
    
    # Audio settings
    SR = 16000
    N_FFT = 1024
    HOP = 160
    N_MELS = 128
    F_MIN = 20.0
    F_MAX = SR / 2 - 20.0
    
    # Denoising settings
    DENOISE_WIN = 1024
    DENOISE_HOP = 256
    DENOISE_NOISE_FRAMES = 20
    DENOISE_THRESH_DB = 6.0
    DENOISE_ATTEN_DB = 20.0
    
    # VAD settings
    VAD_FRAME_MS = 20
    VAD_THR_RATIO = 0.5
    VAD_PAD_MS = 100
    
    # Preprocessing
    PRE_EMPH = 0.97
    HPF_CUTOFF = 40.0
    MIN_SAMPLES = HOP * 6
    
    # Training settings
    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_WORKERS = 0  # Windows/Jupyter compatibility
    EPOCHS = 12
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 3
    
    # System
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Audio extensions
    AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma",
                  ".WAV", ".MP3", ".FLAC", ".M4A", ".OGG", ".AAC", ".WMA")
    
    @classmethod
    def setup_environment(cls):
        """Initialize environment settings"""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.set_num_threads(4)
        if cls.DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Create directories
        for d in [cls.CLEAN_DIR, cls.IMG_DIR, cls.SPLIT_DIR]:
            os.makedirs(d, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class FileUtils:
    """File and path utilities"""
    
    @staticmethod
    def list_audio_files(folder: str, exts: tuple = Config.AUDIO_EXTS) -> List[str]:
        """Recursively list all audio files in folder"""
        files = []
        
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))

        return files
    @staticmethod
    def list_image_files(folder: str, exts: tuple = (".png", ".jpg", ".jpeg")) -> List[str]:
        """Recursively list all image files in a folder"""
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
        return files
    
    @staticmethod
    def unique_output_name(src_path: str) -> str:
        """Generate unique output name: <basename>__<ext>__<hash>.wav"""
        base, ext = os.path.splitext(os.path.basename(src_path))
        h = hashlib.sha1(os.path.abspath(src_path).encode("utf-8")).hexdigest()[:10]
        return f"{base}__{ext.lstrip('.').lower()}__{h}.wav"


# ============================================================================
# AUDIO PROCESSING
# ============================================================================
import torch
import librosa
import soundfile as sf
import numpy as np 
from typing import Optional
import warnings 

class AudioLoader:
    """Load any audio file robustly using multiple fallback methods"""

    def __init__(self, target_sr: int = 16000, device: str = "cpu"):
        self.target_sr = target_sr
        self.device = device
 
    @torch.no_grad()
    def load(self, path: str) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
            if sr != self.target_sr:  wav = torchaudio.functional.resample(wav, sr,   self.target_sr)
            mx = torch.amax(torch.abs(wav)); 
            if mx > 0: wav = wav / mx
            return wav.to(self.device)
        except Exception  :
         
            pass
        # 2) soundfile
        try:
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            wav = torch.from_numpy(y).unsqueeze(0)
            if sr !=  self.target_sr: wav = torchaudio.functional.resample(wav, sr,  self.target_sr)
            mx = torch.amax(torch.abs(wav)); 
            if mx > 0: wav = wav / mx
            return wav.to(self.device)
        except Exception:
            pass
        # 3) librosa
        try:
            y, sr = librosa.load(path, sr= self.target_sr, mono=True)
            wav = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(self.device)
            mx = torch.amax(torch.abs(wav)); 
            if mx > 0: wav = wav / mx
            return wav
        except Exception:
            return None
    @staticmethod
    def _normalize( wav):
        mx = torch.amax(torch.abs(wav))
        return wav / (mx + 1e-8)

class AudioDenoiser:
    """Audio denoising with spectral gating and optional Demucs"""
    
    def __init__(self, device: str = Config.DEVICE):
        self.device = device
        self.window = torch.hann_window(Config.DENOISE_WIN, device=device)
        self.demucs_model = self._init_demucs()
    
    def _init_demucs(self):
        """Initialize Demucs model if available"""
        try:
            from demucs.pretrained import get_model as demucs_get_model
            model = demucs_get_model("htdemucs")
            return model.to(self.device).eval()
        except Exception:
            try:
                model = torch.hub.load("facebookresearch/demucs:main", "htdemucs")
                return model.to(self.device).eval()
            except Exception:
                return None
    
    @torch.no_grad()
    def denoise(self, wav: torch.Tensor, method: str = "auto") -> torch.Tensor:
        """
        Denoise audio
        Args:
            wav: (1, T) tensor on GPU
            method: "spectral", "demucs", or "auto"
        """
        if method == "demucs":
            result = self._demucs_denoise(wav)
            if result is None:
                raise RuntimeError("Demucs failed")
            return result
        elif method == "spectral":
         
            return self._spectral_gate(wav)
        else:  # auto
            result = self._demucs_denoise(wav)
            if result is not None:
                return result
            return self._spectral_gate(wav)
    
    @torch.no_grad()
    def _spectral_gate(self, wav: torch.Tensor) -> torch.Tensor:
        """Spectral gating denoising"""
        stft = torch.stft(
            wav, n_fft=Config.DENOISE_WIN, hop_length=Config.DENOISE_HOP,
            win_length=Config.DENOISE_WIN, window=self.window,
            center=True, return_complex=True
        )
        
        mag = torch.abs(stft) + 1e-8
        phase = stft / mag
        
        # Estimate noise from first frames
        N = min(Config.DENOISE_NOISE_FRAMES, mag.shape[-1])
        noise = mag[..., :N].median(dim=-1, keepdim=True).values
        
        # Apply gate
        mag_db = 20.0 * torch.log10(mag)
        noise_db = 20.0 * torch.log10(noise + 1e-8)
        keep = (mag_db - noise_db) >= Config.DENOISE_THRESH_DB
        atten = 10 ** (-Config.DENOISE_ATTEN_DB / 20.0)
        mag_dn = torch.where(keep, mag, mag * atten)
        
        # Reconstruct
        stft_dn = mag_dn * phase
        wav_out = torch.istft(
            stft_dn, n_fft=Config.DENOISE_WIN, hop_length=Config.DENOISE_HOP,
            win_length=Config.DENOISE_WIN, window=self.window,
            center=True, length=wav.shape[-1]
        ).unsqueeze(0)
        
        return AudioLoader._normalize(wav_out)
    
    @torch.no_grad()
    def _demucs_denoise(self, wav: torch.Tensor) -> Optional[torch.Tensor]:
        """Demucs vocal separation"""
        if self.demucs_model is None:
            return None
        
        try:
            x = wav.unsqueeze(0)  # (1, 1, T)
            out = self.demucs_model(x)
            
            if isinstance(out, (list, tuple)):
                out = out[0]
            
            if out.dim() == 4:
                sources = getattr(self.demucs_model, "sources", ['drums','bass','other','vocals'])
                vocals_idx = sources.index('vocals') if 'vocals' in sources else -1
                vocals = out[:, vocals_idx, 0, :]
            elif out.dim() == 3:
                vocals = out[-1, 0, :].unsqueeze(0)
            else:
                return None
            
            return   AudioLoader._normalize(vocals)
        except Exception:
            return None


class AudioPreprocessor:
    """Audio preprocessing: VAD, HPF, pre-emphasis"""
    
    def __init__(self, sr: int = Config.SR, device: str = Config.DEVICE):
        self.sr = sr
        self.device = device
    
    @torch.no_grad()
    def process(self, wav: torch.Tensor) -> Optional[torch.Tensor]:
        """Full preprocessing pipeline"""
        # VAD trimming
        wav = self._vad_trim(wav)
        
        # High-pass filter
        if Config.HPF_CUTOFF > 0:
            wav = torchaudio.functional.highpass_biquad(wav, self.sr, Config.HPF_CUTOFF)
        
        # Pre-emphasis
        if Config.PRE_EMPH > 0:
            wav = self._pre_emphasize(wav)
        
        # Normalize
        wav = AudioLoader._normalize(wav)
        
        # Check length
        if wav.shape[-1] < Config.MIN_SAMPLES:
            return None
        
        return wav
    
    @torch.no_grad()
    def _vad_trim(self, wav: torch.Tensor) -> torch.Tensor:
        """Simple VAD-based trimming"""
        T = wav.shape[-1]
        hop = int(self.sr * Config.VAD_FRAME_MS / 1000.0)
        win = hop
        
        if T < win:
            return wav
        
        frames = wav.unfold(dimension=-1, size=win, step=hop)
        energy = (frames ** 2).mean(dim=-1).squeeze(0)
        thr = torch.median(energy) * Config.VAD_THR_RATIO
        mask = energy > thr
        
        if not mask.any():
            return wav
        
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        start_f, end_f = idx[0].item(), idx[-1].item()
        pad = int(self.sr * Config.VAD_PAD_MS / 1000.0)
        start = max(0, start_f * hop - pad)
        end = min(T, (end_f + 1) * hop + pad)
        
        return wav[:, start:end] if end > start else wav
    
    @staticmethod
    def _pre_emphasize(wav: torch.Tensor) -> torch.Tensor:
        """Apply pre-emphasis filter"""
        x_shift = torch.zeros_like(wav)
        x_shift[..., 1:] = wav[..., :-1]
        return wav - Config.PRE_EMPH * x_shift


class SpectrogramGenerator:
    """Generate mel spectrograms from audio"""
    
    def __init__(self, device: str = Config.DEVICE):
        self.device = device
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR, n_fft=Config.N_FFT, hop_length=Config.HOP,
            n_mels=Config.N_MELS, f_min=Config.F_MIN, f_max=Config.F_MAX,
            power=2.0
        ).to(device)
        self.amp2db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        ).to(device)
    
    @torch.no_grad()
    def generate(self, wav: torch.Tensor) -> np.ndarray:
        """Generate mel spectrogram image (uint8 numpy array)"""
        S = self.mel_transform(wav)
        S_db = self.amp2db(S).squeeze(0)
        
        # Normalize to [0, 255]
        a = (S_db - S_db.amin()) / (S_db.amax() - S_db.amin() + 1e-8)
        a = (a * 255.0).clamp(0, 255).to(torch.uint8)
        
        return a.detach().cpu().numpy()
    
    @torch.no_grad()
    def wav_to_tensor(self, wav: torch.Tensor, img_size: int = Config.IMG_SIZE) -> torch.Tensor:
        """Generate normalized tensor for model input (1, 1, H, W)"""
        img_array = self.generate(wav)
        pil_img = Image.fromarray(img_array).resize(
            (img_size, img_size), Image.BILINEAR
        )
        x = torch.from_numpy(np.array(pil_img)).float()
        x = x.unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, H, W)
        x = (x - 0.5) / 0.5  # Normalize like training
        return x.to(self.device)


# ============================================================================
# DATASET AND DATA PROCESSING
# ============================================================================

class SpectrogramDataset(Dataset):
    """Dataset for spectrogram images"""
    
    def __init__(self, csv_path: str, augment: bool = False, 
                 img_size: int = Config.IMG_SIZE):
        df = pd.read_csv(csv_path)
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.img_size = img_size
        
        if augment:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomApply([
                    T.RandomAffine(degrees=5, translate=(0.02, 0.02), 
                                  scale=(0.98, 1.02))
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
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            img = Image.open(self.paths[idx]).convert("L")
            x = self.transform(img)
        except Exception:
            # Fallback: return zeros to avoid blocking
            x = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
        
        return x, self.labels[idx]


class DatasetBuilder:
    """Build and manage datasets"""
    
    def __init__(self, base_dir: str = Config.IMG_DIR):
        self.base_dir = base_dir
        self.split_dir = os.path.join(base_dir, "_splits")
        os.makedirs(self.split_dir, exist_ok=True)
    
    def create_splits(self, test_size: float = 0.2, val_size: float = 0.5):
        """Create train/val/test splits"""
        ai_files = FileUtils.list_image_files(os.path.join(self.base_dir, "ai") )
        real_files = FileUtils.list_image_files(os.path.join(self.base_dir, "real"))
        
 
        
        paths = ai_files + real_files
        labels = [1] * len(ai_files) + [0] * len(real_files)
    
        # Stratified split
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, test_size=test_size, random_state=Config.SEED,
            stratify=labels
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=val_size,
            random_state=Config.SEED, stratify=temp_labels
        )
        
        # Save CSVs
        self._save_split("train", train_paths, train_labels)
        self._save_split("val", val_paths, val_labels)
        self._save_split("test", test_paths, test_labels)
        
        print(f"Splits created: Train={len(train_paths)}, "
              f"Val={len(val_paths)}, Test={len(test_paths)}")
    
    def _save_split(self, name: str, paths: List[str], labels: List[int]):
        """Save split to CSV"""
        df = pd.DataFrame({"path": paths, "label": labels})
        df.to_csv(os.path.join(self.split_dir, f"{name}.csv"), index=False)
    
    def get_dataloaders(self, batch_size: int = Config.BATCH_SIZE,
                       num_workers: int = Config.NUM_WORKERS) -> Dict[str, DataLoader]:
        """Get train/val/test dataloaders"""
        train_ds = SpectrogramDataset(
            os.path.join(self.split_dir, "train.csv"), augment=True
        )
        val_ds = SpectrogramDataset(
            os.path.join(self.split_dir, "val.csv"), augment=False
        )
        test_ds = SpectrogramDataset(
            os.path.join(self.split_dir, "test.csv"), augment=False
        )
        
        pin_memory = Config.DEVICE == "cuda"
        
        return {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory),
        }


# ============================================================================
# MODELS
# ============================================================================

class ModelFactory:
    """Factory for creating different model architectures"""
    
    @staticmethod
    def create_model(name: str) -> nn.Module:
        """Create model by name"""
        name = name.lower()
        
        if name == "resnet18":
            return ModelFactory._create_resnet18()
        elif name == "efficientnet_b0":
            return ModelFactory._create_efficientnet_b0()
        elif name == "efficientnet_b1":
            return ModelFactory._create_efficientnet_b1()
        elif name == "densenet121":
            return ModelFactory._create_densenet121()
        elif name == "densenet169":
            return ModelFactory._create_densenet169()
        elif name == "mobilenet_v3_small":
            return ModelFactory._create_mobilenet_v3_small()
        elif name == "convnext_tiny":
            return ModelFactory._create_convnext_tiny()
        elif name == "vit_b16":
            return ModelFactory._create_vit_b16()
        elif name == "swin_tiny":
            return ModelFactory._create_swin_tiny()
        else:
            raise ValueError(f"Unknown model: {name}")
    
    @staticmethod
    def _create_resnet18() -> nn.Module:
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model
    
    @staticmethod
    def _create_efficientnet_b0() -> nn.Module:
        model = efficientnet_b0(weights=None)
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, first.out_channels, kernel_size=first.kernel_size,
            stride=first.stride, padding=first.padding, bias=(first.bias is not None)
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    
    @staticmethod
    def _create_efficientnet_b1() -> nn.Module:
        model = efficientnet_b1(weights=None)
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, first.out_channels, kernel_size=first.kernel_size,
            stride=first.stride, padding=first.padding, bias=(first.bias is not None)
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    
    @staticmethod
    def _create_densenet121() -> nn.Module:
        model = densenet121(weights=None)
        first = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            1, first.out_channels, kernel_size=first.kernel_size,
            stride=first.stride, padding=first.padding, bias=(first.bias is not None)
        )
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model
    
    @staticmethod
    def _create_densenet169() -> nn.Module:
        model = densenet169(weights=None)
        first = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            1, first.out_channels, kernel_size=first.kernel_size,
            stride=first.stride, padding=first.padding, bias=(first.bias is not None)
        )
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model
    
    @staticmethod
    def _create_mobilenet_v3_small() -> nn.Module:
        model = mobilenet_v3_small(weights=None)
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, first.out_channels, kernel_size=first.kernel_size,
            stride=first.stride, padding=first.padding, bias=(first.bias is not None)
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model
    
    @staticmethod
    def _create_convnext_tiny() -> nn.Module:
        model = convnext_tiny(weights=None)
        model.features[0][0] = nn.Conv2d(1, 96, kernel_size=4, stride=4, padding=0, bias=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        return model
    
    @staticmethod
    def _create_vit_b16() -> nn.Module:
        model = vit_b_16(weights=None)
        embed_dim = model.conv_proj.out_channels
        model.conv_proj = nn.Conv2d(1, embed_dim, kernel_size=16, stride=16, bias=True)
        
        try:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, 2)
        except Exception:
            last = list(model.heads.children())[-1]
            in_features = last.in_features
            model.heads = nn.Sequential(nn.Linear(in_features, 2))
        
        return model
    
    @staticmethod
    def _create_swin_tiny() -> nn.Module:
        model = swin_t(weights=None)
        old = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=(old.bias is not None)
        )
        model.head = nn.Linear(model.head.in_features, 2)
        return model


class EnsembleModel(nn.Module):
    """Ensemble of multiple models"""
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.fc = nn.Linear(len(models) * 2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = []
            for model in self.models:
                logits = model(x)
                probs.append(F.softmax(logits, dim=1))
        
        combined = torch.cat(probs, dim=1)
        return self.fc(combined)


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    """Model trainer with early stopping and AMP"""
    
    def __init__(self, model: nn.Module, device: str = Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train(self, dataloaders: Dict[str, DataLoader], epochs: int = Config.EPOCHS,
              patience: int = Config.PATIENCE, save_path: Optional[str] = None):
        """Train model with early stopping"""
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc, _, _ = self._run_epoch(
                dataloaders["train"], train=True
            )
            
            # Validate
            val_loss, val_acc, val_logits, val_targets = self._run_epoch(
                dataloaders["val"], train=False
            )
            
            print(f"Epoch {epoch:02d} | "
                  f"Train {train_loss:.4f}/{train_acc:.3f} | "
                  f"Val {val_loss:.4f}/{val_acc:.3f}")
            
            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  -> saved: {save_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print("Early stopping.")
                    break
        
        return self.best_val_acc
    
    def _run_epoch(self, loader: DataLoader, train: bool = False):
        """Run one epoch"""
        self.model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        all_logits, all_targets = [], []
        
        pbar = tqdm(loader, leave=False)
        for x, y in pbar:
            x = x.to(self.device, non_blocking=True)
            y = torch.as_tensor(y, device=self.device)
            
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', 
                                       enabled=(self.device == "cuda")):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with torch.no_grad(), torch.amp.autocast(device_type='cuda',
                                                         enabled=(self.device == "cuda")):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
            
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_logits.append(logits.detach().float().cpu())
            all_targets.append(y.detach().cpu())
            
            mode = 'Train' if train else 'Eval'
            pbar.set_description(
                f"{mode} loss {loss_sum/max(total,1):.4f} "
                f"acc {correct/max(total,1):.3f}"
            )
        
        avg_loss = loss_sum / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc, torch.cat(all_logits), torch.cat(all_targets)
    
    def evaluate(self, loader: DataLoader) -> Dict:
        """Evaluate model and return metrics"""
        self.model.eval()
        _, acc, logits, targets = self._run_epoch(loader, train=False)
        
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)
        tgts = targets.numpy()
        
        return {
            "accuracy": acc,
            "roc_auc": roc_auc_score(tgts, probs),
            "predictions": preds,
            "probabilities": probs,
            "targets": tgts,
            "logits": logits
        }


class ThresholdTuner:
    """Tune classification threshold using validation set"""
    
    def __init__(self, model: nn.Module, device: str = Config.DEVICE):
        self.model = model.to(device).eval()
        self.device = device
    
    @torch.no_grad()
    def find_optimal_threshold(self, val_loader: DataLoader) -> float:
        """Find optimal threshold using Youden's J statistic"""
        # Collect predictions
        all_probs, all_targets = [], []
        
        with torch.amp.autocast(device_type='cuda', 
                               enabled=(self.device == "cuda")):
            for x, y in tqdm(val_loader, desc="Tuning threshold"):
                x = x.to(self.device, non_blocking=True)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.detach().float().cpu().numpy())
                all_targets.append(y.numpy())
        
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        
        # Find threshold maximizing Youden's J
        fpr, tpr, thresholds = roc_curve(targets, probs)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(thresholds[best_idx])
        
        print(f"Optimal threshold (Youden's J): {best_threshold:.6f}")
        return best_threshold


# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

class Evaluator:
    """Model evaluation and metrics"""
    
    @staticmethod
    def evaluate_model(model: nn.Module, test_loader: DataLoader,
                      threshold: float = 0.5,
                      device: str = Config.DEVICE) -> Dict:
        """Complete evaluation with all metrics"""
        model.eval()
        all_probs, all_targets = [], []
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda',
                                                 enabled=(device == "cuda")):
            for x, y in tqdm(test_loader, desc="Evaluating"):
                x = x.to(device, non_blocking=True)
                logits = model(x)
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.detach().float().cpu().numpy())
                all_targets.append(y.numpy())
        
        probs = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)
        preds = (probs >= threshold).astype(int)
        
        # Calculate metrics
        acc = (preds == targets).mean()
        roc_auc = roc_auc_score(targets, probs)
        
        print(f"\n{'='*60}")
        print(f"TEST RESULTS (threshold={threshold:.4f})")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc_auc:.6f}")
        print(f"\n{classification_report(targets, preds, target_names=['Real(0)', 'AI(1)'])}")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(targets, preds))
        
        return {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "predictions": preds,
            "probabilities": probs,
            "targets": targets,
            "threshold": threshold
        }
    
    @staticmethod
    def plot_results(results: Dict, title: str = "Model Performance"):
        """Plot ROC curve and confusion matrix"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(results["targets"], results["probabilities"])
        roc_auc = auc(fpr, tpr)
        
        axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="navy")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(f"ROC Curve — {title}")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)
        
        # Confusion Matrix
        cm = confusion_matrix(results["targets"], results["predictions"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                   xticklabels=["Real(0)", "AI(1)"],
                   yticklabels=["Real(0)", "AI(1)"])
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        axes[1].set_title(f"Confusion Matrix — {title}")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_models(results_dict: Dict[str, Dict]):
        """Compare multiple models"""
        data = []
        for model_name, results in results_dict.items():
            # Extract F1 score from predictions
            from sklearn.metrics import f1_score
            f1 = f1_score(results["targets"], results["predictions"])
            data.append({
                "Model": model_name,
                "ACC": results["accuracy"],
                "ROC_AUC": results["roc_auc"],
                "F1": f1
            })
        
        df = pd.DataFrame(data)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        df_melt = df.melt(id_vars="Model", value_vars=["ACC", "ROC_AUC", "F1"],
                         var_name="Metric", value_name="Score")
        sns.barplot(x="Model", y="Score", hue="Metric", data=df_melt, ax=axes[0])
        axes[0].set_ylim(0.9, 1.01)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
        axes[0].set_title("Model Comparison — Bar Chart")
        axes[0].legend(loc="lower right")
        axes[0].grid(axis="y", ls="--", alpha=0.6)
        
        # Line plot
        axes[1].plot(df["Model"], df["ACC"], marker="o", label="Accuracy")
        axes[1].plot(df["Model"], df["ROC_AUC"], marker="s", label="ROC-AUC")
        axes[1].plot(df["Model"], df["F1"], marker="^", label="F1-score")
        axes[1].set_xticklabels(df["Model"], rotation=45, ha="right")
        axes[1].set_ylim(0.9, 1.01)
        axes[1].set_ylabel("Score")
        axes[1].set_title("Model Comparison — Line Chart")
        axes[1].legend()
        axes[1].grid(ls="--", alpha=0.6)
        
        plt.tight_layout()
        plt.show()
        
        return df


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class AudioProcessingPipeline:
    """End-to-end audio processing pipeline"""
    
    def __init__(self):
        self.loader = AudioLoader()
        self.denoiser = AudioDenoiser()
        self.preprocessor = AudioPreprocessor()
        self.spec_gen = SpectrogramGenerator()
    
    def process_folder(self, src_dir: str, dst_dir: str, 
                      denoise: bool = True, method: str = "auto"):
        """Process all audio files in a folder"""
        os.makedirs(dst_dir, exist_ok=True)
        files = FileUtils.list_audio_files(src_dir)
        skipped = 0
       
        for fp in tqdm(files, desc=f"Processing {os.path.basename(src_dir)}"):
            out_name = FileUtils.unique_output_name(fp)
            out_path = os.path.join(dst_dir, out_name)
           
            if os.path.exists(out_path):
                continue
      
            try:
                # Load
                wav = self.loader.load(fp)
             
                if wav is None:
                    skipped += 1
                    continue
                
                # Denoise
                if denoise:
                    wav = self.denoiser.denoise(wav, method=method)
                
                # Save
                self._save_wav(wav, out_path)
            except Exception as e:
           
                skipped += 1
        
        print(f"{src_dir} -> {dst_dir} | total={len(files)} | skipped={skipped}")
    
    def create_spectrograms(self, audio_dir: str, output_dir: str):
        """Create spectrogram images from audio files"""
        os.makedirs(output_dir, exist_ok=True)
        files = glob.glob(os.path.join(audio_dir, "*.wav"))
        skipped = 0
      
        for fp in tqdm(files, desc=f"Generating spectrograms"):
            base_name = os.path.splitext(os.path.basename(fp))[0]
            out_path = os.path.join(output_dir, f"{base_name}.png")
           
            if os.path.exists(out_path):
                continue
          
            
            try:
                wav = self.loader.load(fp)
                
                if wav is None:
                    skipped += 1
                    continue
                
             
                # Preprocess
                wav = self.preprocessor.process(wav)
                if wav is None:
                    skipped += 1
                    continue
                
                # Generate spectrogram
                img_array = self.spec_gen.generate(wav)
            
                Image.fromarray(img_array).save(out_path, format="PNG")
            except Exception as e:
                skipped += 1
     
       
        print(f"{audio_dir} -> {output_dir} | total={len(files)} | skipped={skipped}")
 
    @staticmethod
    def _save_wav(tensor_gpu: torch.Tensor, path: str, sr: int = Config.SR):
        """Save tensor as WAV file"""
        y = tensor_gpu.detach().cpu().squeeze(0).numpy()
        sf.write(path, y, sr, subtype="PCM_16")


# ============================================================================
# INFERENCE
# ============================================================================

import os
import torch
from torch import nn

class VoiceDetector:
    """High-level interface for AI voice detection"""
    
    def __init__(self, model_path: str, model_name: str = None, threshold: float = 0.5,
                 device: str = "cpu"):
        self.device = device
        self.threshold = threshold
        self.model_path = model_path

        # --- Auto-detect model name if not provided ---
        if model_name is None:
            model_name = self._infer_model_name(model_path)
        print(f"🧠 Detected model architecture: {model_name}")

        # --- Build model using factory ---
        self.model = ModelFactory.create_model(model_name)

        # --- Load state dict safely ---
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)

        # --- Prepare model for inference ---
        self.model.to(device)
        self.model.eval()

        # --- Initialize helper components ---
        self.loader = AudioLoader(device=device)
        self.preprocessor = AudioPreprocessor(device=device)
        self.spec_gen = SpectrogramGenerator(device=device)

        print(f"✅ VoiceDetector ready with {model_name}")

    # --------------------------------------------------------------------------
    def _infer_model_name(self, path: str) -> str:
        """Infer model name from filename (e.g. best_efficientnet_b0.pt → efficientnet_b0)"""
        name = os.path.basename(path).lower()
        candidates = [
            "resnet18", "efficientnet_b0", "efficientnet_b1",
            "densenet121", "densenet169", "mobilenet_v3_small",
            "convnext_tiny", "vit_b16", "swin_tiny"
        ]
        for candidate in candidates:
            if candidate in name:
                return candidate
        raise ValueError(f"❌ Could not infer model name from: {name}")

    
    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict:
        """
        Predict if audio is AI-generated or real
        Returns: dict with path, probability, label, threshold
        """
        # Load and preprocess
        wav = self.loader.load(audio_path)
        if wav is None:
            raise ValueError(f"Could not load audio: {audio_path}")
        
        # Convert to model input
        x = self.spec_gen.wav_to_tensor(wav)  # (1, 1, 224, 224)
        
        # Predict
        with torch.amp.autocast(device_type='cuda', 
                               enabled=(self.device == "cuda")):
            logits = self.model(x)
            prob_ai = F.softmax(logits, dim=1)[0, 1].item()
        
        label = "AI voice" if prob_ai >= self.threshold else "Real voice"
        
        result = {
            "path": audio_path,
            "prob_ai": prob_ai,
            "threshold": self.threshold,
            "label": label
        }
        
        print(f"\nFile: {audio_path}")
        print(f"Prob(AI) = {prob_ai:.6f} | Threshold = {self.threshold:.3f} -> {label}")
        
        return result
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """Predict multiple audio files"""
        results = []
        for path in tqdm(audio_paths, desc="Predicting"):
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append({
                    "path": path,
                    "error": str(e)
                })
        return results
    
    def export_torchscript(self, output_path: str):
        """Export model to TorchScript"""
        example = torch.randn(1, 1, Config.IMG_SIZE, Config.IMG_SIZE, 
                             device=self.device)
        self.model.eval()
        ts = torch.jit.trace(self.model, example)
        ts.save(output_path)
        print(f"TorchScript model saved: {output_path}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

class WorkflowManager:
    """Manage complete training workflow"""
    
    def __init__(self):
        Config.setup_environment()
        self.pipeline = AudioProcessingPipeline()
        self.dataset_builder = DatasetBuilder()
    
    def prepare_data(self):
        """Prepare data: denoise and create spectrograms"""
        print("\n" + "="*60)
        print("STEP 1: Audio Denoising")
        print("="*60)
        
        # Process AI audio
        self.pipeline.process_folder(
            Config.AI_DIR,
            os.path.join(Config.CLEAN_DIR, "ai"),
            denoise=True, method="auto"
        )
        
        # Process Real audio
        self.pipeline.process_folder(
            Config.REAL_DIR,
            os.path.join(Config.CLEAN_DIR, "real"),
            denoise=True, method="auto"
        )
        
        print("\n" + "="*60)
        print("STEP 2: Spectrogram Generation")
        print("="*60)
        
        # Create spectrograms for AI
        self.pipeline.create_spectrograms(
            os.path.join(Config.CLEAN_DIR, "ai"),
            os.path.join(Config.IMG_DIR, "ai")
        )
        
        # Create spectrograms for Real
        self.pipeline.create_spectrograms(
            os.path.join(Config.CLEAN_DIR, "real"),
            os.path.join(Config.IMG_DIR, "real")
        )
        
        print("\n" + "="*60)
        print("STEP 3: Creating Train/Val/Test Splits")
        print("="*60)
   
        self.dataset_builder.create_splits()
    
    def train_model(self, model_name: str = "efficientnet_b0") -> str:
        """Train a single model"""
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print("="*60)
        
        # Create model
        model = ModelFactory.create_model(model_name)
        
        # Get dataloaders
        dataloaders = self.dataset_builder.get_dataloaders()
        
        # Train
        save_path = os.path.join(Config.IMG_DIR, f"best_{model_name}_ai_vs_real.pt")
        trainer = Trainer(model)
        trainer.train(dataloaders, save_path=save_path)
        
        # Evaluate
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()} on Test Set")
        print("="*60)
        
        model.load_state_dict(torch.load(save_path, map_location=Config.DEVICE))
        results = Evaluator.evaluate_model(
            model, dataloaders["test"], threshold=0.5
        )
        Evaluator.plot_results(results, title=model_name.upper())
        
        return save_path
    
    def tune_and_evaluate(self, model_path: str, model_name: str):
        """Tune threshold and evaluate with optimal threshold"""
        print(f"\n{'='*60}")
        print(f"Threshold Tuning for {model_name.upper()}")
        print("="*60)
        
        # Load model
        model = ModelFactory.create_model(model_name)
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        
        # Get dataloaders
        dataloaders = self.dataset_builder.get_dataloaders()
        
        # Tune threshold
        tuner = ThresholdTuner(model)
        best_threshold = tuner.find_optimal_threshold(dataloaders["val"])
        
        # Evaluate with tuned threshold
        results = Evaluator.evaluate_model(
            model, dataloaders["test"], threshold=best_threshold
        )
        Evaluator.plot_results(results, 
                              title=f"{model_name.upper()} (Tuned)")
        
        return results, best_threshold


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main_training_workflow():
    """Complete training workflow example"""
    # Initialize
    workflow = WorkflowManager()
    
    # Step 1: Prepare data (uncomment if needed)
    workflow.prepare_data()
    
    # Step 2: Train model
    model_path = workflow.train_model("efficientnet_b0")
    
    # Step 3: Tune threshold and evaluate
    results, threshold = workflow.tune_and_evaluate(model_path, "efficientnet_b0")
 
    # Step 4: Export for inference
    detector = VoiceDetector(model_path, threshold=threshold)
    export_path = os.path.join(Config.IMG_DIR, "efficientnet_b0_ai_vs_real.torchscript.pt")
    detector.export_torchscript(export_path)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Model: {model_path}")
    print(f"TorchScript: {export_path}")
    print(f"Optimal Threshold: {threshold:.6f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test ROC-AUC: {results['roc_auc']:.6f}")
    print("="*60)


def main_inference_example():
    """Inference example"""
    # Initialize detector
    model_path = os.path.join(
        Config.IMG_DIR, 
        "efficientnet_b0_ai_vs_real.torchscript.pt"
    )
    detector = VoiceDetector(model_path, threshold=0.9167826771736145)
    
    # Predict single file
    audio_path = r"C:\path\to\your\audio.wav"
    result = detector.predict(audio_path)
    
    # Predict multiple files
    audio_files = [
        r"C:\path\to\audio1.wav",
        r"C:\path\to\audio2.wav",
    ]
    results = detector.predict_batch(audio_files)
    
    # Print summary
    for r in results:
        if "error" not in r:
            print(f"{r['label']}: {r['prob_ai']:.4f} - {r['path']}")


if __name__ == "__main__":
    # Run training workflow
    main_training_workflow()
    
    # Or run inference
    # main_inference_example()