import argparse
import numpy as np
import torch
import pydicom
import nrrd
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn

def get_tag_value_as_float(ds, tag_name, alt_tag_hex=None):
    """
    Helper to safely extract a DICOM tag value as float.
    Mimics the C++ logic of trying primary tag then secondary.
    """
    val_str = "0.0"
    
    # Try attribute name first (e.g., 'AcquisitionTime')
    if hasattr(ds, tag_name):
        val = getattr(ds, tag_name)
        if val:
            val_str = str(val)
    # Try alternative hex tag if provided (e.g., 0x0008, 0x002a)
    elif alt_tag_hex and alt_tag_hex in ds:
        val_str = str(ds[alt_tag_hex].value)
        
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def calculate_tr_from_dicom(dicoms):
    """
    Calculates the acquisition rate (TR) based on the first two available DICOMs.
    Ported from C++ tlDataIO::getGDCMAcquisitionRate.
    """
    if len(dicoms) < 2:
        print("Warning: Not enough DICOMs to calculate TR. Defaulting to 0.3s")
        return 0.3

    d1 = dicoms[0]
    d2 = dicoms[1]

    # Extract Time Stamps
    # Primary: AcquisitionTime (0008,0032), Secondary: AcquisitionDateTime (0008,002a)
    t1 = get_tag_value_as_float(d1, 'AcquisitionTime', 0x0008002a)
    t2 = get_tag_value_as_float(d2, 'AcquisitionTime', 0x0008002a)

    # Extract Seconds part using the C++ logic:
    # (Val/100 - floor(Val/100)) * 100 extracts the SS.frac part 
    # Works for HHMMSS.frac and YYYYMMDDHHMMSS.frac
    t1_sec = ( (t1/100.0) - math.floor(t1/100.0) ) * 100.0
    t2_sec = ( (t2/100.0) - math.floor(t2/100.0) ) * 100.0

    # Extract Instance Numbers
    inst1 = float(d1.InstanceNumber) if 'InstanceNumber' in d1 else 1.0
    inst2 = float(d2.InstanceNumber) if 'InstanceNumber' in d2 else 2.0

    # Calculate Time Difference
    if t1_sec == t2_sec:
        print("Warning: Timestamps identical. Defaulting to 0.3s")
        return 0.3
    
    if t2_sec > t1_sec:
        diff_sec = t2_sec - t1_sec
    else:
        diff_sec = (60.0 - t1_sec) + t2_sec
    
    if diff_sec < 0.0:
        diff_sec += 1.0
        
    inst_diff = inst2 - inst1
    if inst_diff == 0: 
        inst_diff = 1.0

    # Final TR
    tr_calc = diff_sec / inst_diff
    
    if tr_calc <= 0.0 or tr_calc > 5.0:
        print(f"Warning: Calculated TR {tr_calc:.4f}s seems invalid. Defaulting to 0.3s")
        return 0.3

    return tr_calc

def load_dicoms(dicom_dir):
    dicom_files = [f for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')]
    if not dicom_files:
        raise ValueError("No DICOM files found in directory.")
    
    # Load and sort by InstanceNumber
    dicoms = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in dicom_files]
    dicoms.sort(key=lambda x: int(x.InstanceNumber))
    
    # Calculate TR dynamically
    tr_val = calculate_tr_from_dicom(dicoms)
    print(f"Detected TR: {tr_val:.4f} s")

    height, width = dicoms[0].pixel_array.shape
    time_steps = len(dicoms)
    data = np.zeros((height, width, time_steps), dtype=np.float32)
    
    for t, ds in enumerate(dicoms):
        data[:, :, t] = ds.pixel_array.astype(np.float32)
    
    print(f"Loaded {time_steps} time steps, shape {data.shape}")
    return data, tr_val

def run_inference(data, tr_val, model_path, output_path, window_min, window_max):
    
    height, width, time_steps = data.shape
    model_input_len = 190
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load JIT model
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    batch_size = 4096 if device.type == 'cuda' else 512
    
    print("Padding data for 3x3 extraction...")
    padded_data = np.pad(data, ((1,1), (1,1), (0,0)), mode='edge')
        
    # Maps
    vamp = np.zeros((width, height), dtype=np.float32)
    qamp = np.zeros((width, height), dtype=np.float32)
    vtime = np.zeros((width, height), dtype=np.float32)
    qtime = np.zeros((width, height), dtype=np.float32)
    baseline_map = np.zeros((width, height), dtype=np.float32) 
    
    batch_data = []
    pixel_coords = []
    
    stabilizer = 50.0

    print("Processing voxels (Spatial Mode)...")
    
    for x in tqdm(range(width)):
        for y in range(height):
            
            # Extract 3x3 Patch
            patch_3x3 = padded_data[x:x+3, y:y+3, :] # Shape (3, 3, Time)
            
            # Flatten spatial dims -> (9, Time)
            neighbors = patch_3x3.reshape(9, -1) 
            
            # --- PREPROCESSING ---
            means = np.mean(neighbors, axis=1, keepdims=True)
            means[means < 1e-6] = 1.0 
            
            # Soft Normalization
            norm_neighbors = (neighbors - means) / (means + stabilizer)
            
            # --- INPUT TENSOR CONSTRUCTION ---
            input_sample = np.zeros((10, model_input_len), dtype=np.float32)
            limit = min(time_steps, model_input_len)
            
            pad_total = model_input_len - limit
            pad_left = pad_total // 2
            
            # Fill Channels 0-8 (Signals)
            input_sample[0:9, pad_left : pad_left + limit] = norm_neighbors[:, :limit]            
            
            # Fill Channel 9 (TR)
            input_sample[9, :] = tr_val
            
            batch_data.append(input_sample)
            pixel_coords.append((x, y))
            
            # --- BATCH EXECUTION ---
            if len(pixel_coords) >= batch_size or (x == width-1 and y == height-1):
                input_tensor = torch.tensor(np.array(batch_data)).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor).cpu().numpy()
                
                for i, (px, py) in enumerate(pixel_coords):
                    
                    local_base = np.mean(data[px, py, :])
                    baseline_map[px, py] = local_base
                    
                    out_v = max(output[i, 0], 0.0)
                    out_q = max(output[i, 1], 0.0)

                    # Scale back to Absolute Amplitude
                    vamp[px, py] = (out_v * (local_base + stabilizer)) 
                    qamp[px, py] = (out_q * (local_base + stabilizer))                        
                    
                    vtime[px, py] = np.arctan2(output[i, 2], output[i, 3]) 
                    qtime[px, py] = np.arctan2(output[i, 4], output[i, 5]) - np.pi/2
                
                batch_data = []
                pixel_coords = []

    # --- FRACTIONAL VENTILATION ---
    print("Calculating Fractional Ventilation Map...")
    eps = 1e-5
    fv_map = vamp / (baseline_map + (vamp / 2.0) + eps)
    fv_map[baseline_map < 10] = 0.0 

    # Visualization
    if window_min is None: window_min = 0
    if window_max is None: window_max = np.percentile(vamp, 95)
    
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(vamp, cmap='gray', vmin=window_min, vmax=window_max)
    ax1.set_title(f'Ventilation Amp (Abs) [TR={tr_val:.3f}s]') 
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(fv_map, cmap='gray', vmin=0, vmax=0.3) 
    ax2.set_title('Fractional Ventilation (FV)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(qamp, cmap='gray', vmin=window_min, vmax=window_max*1.1)
    ax3.set_title('Perfusion Amp (Abs)')
    ax3.axis('off')

    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(vtime, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax4.set_title('Ventilation Phase')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04) 

    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(qtime, cmap='twilight_shifted', vmin=-np.pi, vmax=np.pi)
    ax5.set_title('Perfusion Phase')
    ax5.axis('off')

    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(baseline_map, cmap='gray')
    ax6.set_title('Baseline Image')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.show()

    # Save NRRD
    maps = np.stack([
        np.rot90(vamp), 
        np.rot90(qamp), 
        np.rot90(vtime), 
        np.rot90(qtime),
        np.rot90(fv_map)
    ], axis=-1)
    
    header = {'encoding': 'gzip', 'space': 'left-posterior-superior'}
    nrrd.write(output_path, maps, header)
    print(f"Saved maps to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-Wave Inference Script for DICOM Data")
    parser.add_argument("-d", "--dicom_dir", required=True, help="Path to directory containing DICOM time-series")
    parser.add_argument("-m", "--model", default="VQWaveNetwork_V13.pt", help="Path to the trained model (.pt)")
    parser.add_argument("-o", "--output", default="output_maps.nrrd", help="Filename for the output NRRD file")
    parser.add_argument("--window_min", type=float, default=None, help="Min value for plotting")
    parser.add_argument("--window_max", type=float, default=None, help="Max value for plotting")
    args = parser.parse_args()
    
    data, tr_val = load_dicoms(args.dicom_dir)
    run_inference(data, tr_val, args.model, args.output, args.window_min, args.window_max)
    
