"""
Phase 0.4(d): SI-FID as a reference-free signal for non-GT tests.
"""

import numpy as np
import cv2

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from scipy.linalg import sqrtm
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_inception_model = None

def _get_inception_model():
    global _inception_model
    if _inception_model is None:
        _inception_model = models.inception_v3(pretrained=True, transform_input=False)
        _inception_model.fc = torch.nn.Identity()
        _inception_model.eval()
        if torch.cuda.is_available():
            _inception_model = _inception_model.cuda()
    return _inception_model

def _extract_patches(img: np.ndarray, patch_size: int = 299, stride: int = 150) -> np.ndarray:
    h, w = img.shape[:2]
    patches = []
    for y in range(0, max(1, h - patch_size + 1), stride):
        for x in range(0, max(1, w - patch_size + 1), stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = max(0, patch_size - patch.shape[0])
                pad_w = max(0, patch_size - patch.shape[1])
                patch = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            patches.append(patch)
    return np.array(patches)

def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            covmean = covmean.real
        else:
            covmean = covmean.real
            
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def compute_si_fid(img: np.ndarray, patch_size: int = 299) -> float:
    if img is None or not _HAS_TORCH:
        return float("nan")
    
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = _extract_patches(rgb, patch_size=patch_size, stride=patch_size//2)
        
        if len(patches) == 0:
            return float("nan")

        model = _get_inception_model()
        device = next(model.parameters()).device
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        features_list = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack([transform(p) for p in patches[i:i+batch_size]]).to(device)
                features = model(batch)
                features_list.append(features.cpu().numpy())
                
        features = np.concatenate(features_list, axis=0)
        
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        mu_ref = np.zeros_like(mu)
        sigma_ref = np.eye(sigma.shape[0])
        
        score = _calculate_frechet_distance(mu, sigma, mu_ref, sigma_ref)
        return float(score)
    except Exception:
        return float("nan")
