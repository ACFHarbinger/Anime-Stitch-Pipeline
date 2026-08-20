import os
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import asp_backend_benchmark.generate_samples as gen_samples
except ImportError:
    import backend.benchmark.generate_samples as gen_samples

def test_generator_runs(tmp_path):
    """Test that the generator script runs without errors and creates dirs."""
    gen_samples.create_gradient_sample(tmp_path / "grad", num_frames=3)
    gen_samples.create_pattern_sample(tmp_path / "pat", num_frames=3)
    gen_samples.create_layered_pan_sample(tmp_path / "layer", num_frames=3)
    
    assert (tmp_path / "grad").exists()
    assert (tmp_path / "pat").exists()
    assert (tmp_path / "layer").exists()
    assert (tmp_path / "layer" / "manifest.json").exists()

def test_output_png_dimensions(tmp_path):
    """Test that the output PNGs have expected dimensions."""
    gen_samples.create_gradient_sample(tmp_path / "grad", num_frames=2, width=400, height=600)
    
    img_path = tmp_path / "grad" / "frame_000.png"
    assert img_path.exists()
    
    with Image.open(img_path) as img:
        assert img.size == (400, 600)
        assert img.mode == 'RGB'

def test_data_samples_directory_structure():
    """Test that the data/samples directory has the expected structure."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/samples'))
    
    grad_dir = os.path.join(base_dir, 'test_scroll_gradient')
    pat_dir = os.path.join(base_dir, 'test_scroll_pattern')
    
    assert os.path.isdir(grad_dir), f"Directory missing: {grad_dir}"
    assert os.path.isdir(pat_dir), f"Directory missing: {pat_dir}"
    
    grad_files = [f for f in os.listdir(grad_dir) if f.endswith('.png')]
    assert len(grad_files) == 6, f"Expected 6 frames in grad_dir, got {len(grad_files)}"
    
    pat_files = [f for f in os.listdir(pat_dir) if f.endswith('.png')]
    assert len(pat_files) == 6, f"Expected 6 frames in pat_dir, got {len(pat_files)}"
