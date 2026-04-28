#!/usr/bin/env python3

__author__      = "Grzegorz Baumann"
__contact__     = "g.baumann@unibas.ch"

import argparse
import numpy as np
import torch
import pydicom
from pydicom.dataset import Dataset, FileDataset
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import datetime

def get_tag_value_as_float(ds, tag_name, alt_tag_hex=None):
    """
    Helper to safely extract a DICOM tag value as float.
    Mimics the C++ logic of trying primary tag then secondary.
    """
    val_str = "0.0"
    
    if hasattr(ds, tag_name):
        val = getattr(ds, tag_name)
        if val:
            val_str = str(val)

    elif alt_tag_hex and alt_tag_hex in ds:
        val_str = str(ds[alt_tag_hex].value)
        
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def calculate_tr_from_dicom(dicoms):
    """
    Calculates the acquisition rate (TR) based on the first two available DICOMs.
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

    t1_sec = ( (t1/100.0) - math.floor(t1/100.0) ) * 100.0
    t2_sec = ( (t2/100.0) - math.floor(t2/100.0) ) * 100.0

    inst1 = float(d1.InstanceNumber) if 'InstanceNumber' in d1 else 1.0
    inst2 = float(d2.InstanceNumber) if 'InstanceNumber' in d2 else 2.0

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
    """
    Loads a 2D+t DICOM series
    """
    dicom_files = [f for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')]
    if not dicom_files:
        raise ValueError("No DICOM files found.")
    dicoms = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in dicom_files]
    dicoms.sort(key=lambda x: int(x.InstanceNumber))
    tr_val = calculate_tr_from_dicom(dicoms)
    time_steps = len(dicoms)
    print(f"Detected acquisition rate: {1.0/tr_val:.2f} images/second")
    h, w = dicoms[0].pixel_array.shape
    data = np.zeros((h, w, time_steps), dtype=np.float32)
    for t, ds in enumerate(dicoms):
        data[:, :, t] = ds.pixel_array.astype(np.float32)
        
    print(f"Loaded {time_steps} time steps, shape {data.shape}")
    return data, tr_val, dicoms[0]

def save_as_dicom(data_map, reference_ds, output_filename, series_description, is_phase=False):
    """
    Saves functional maps with auto-calculated Windowing and proper 
    handling of signed phase values using Slope/Intercept.
    """
    ds = reference_ds.copy()
    ds.SeriesDescription = series_description
    ds.SeriesNumber = int(ds.SeriesNumber) + 1000 if hasattr(ds, 'SeriesNumber') else 1000
    ds.InstanceNumber = 1
    ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.ContentTime = datetime.datetime.now().strftime('%H%M%S')
    
    # --- PHYSICAL SCALING ---
    if is_phase:
        # Phase is roughly [-pi, pi]. Offset by 4 to make all values positive for uint16 storage
        # and use a 0.0001 slope for high precision.
        rescale_intercept = -4.0
        rescale_slope = 0.0001
    else:
        # Amplitudes are positive
        rescale_intercept = 0.0
        rescale_slope = 0.01

    # Map: PixelValue = (StoredValue * Slope) + Intercept 
    # => StoredValue = (PixelValue - Intercept) / Slope
    stored_data = ((data_map - rescale_intercept) / rescale_slope).astype(np.uint16)
    
    # --- AUTO WINDOWING (WW/WL) ---
    # Calculate based on stored values for viewer compatibility
    vmin, vmax = np.percentile(data_map, [1, 99])
    window_width = vmax - vmin
    window_center = vmin + (window_width / 2)
    
    ds.WindowCenter = f"{window_center:.4f}"
    ds.WindowWidth = f"{window_width:.4f}"
    
    ds.RescaleSlope = f"{rescale_slope:.6f}"
    ds.RescaleIntercept = str(int(rescale_intercept))
    
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 0 # Unsigned uint16 (Offset handled by RescaleIntercept)
    ds.PixelData = stored_data.tobytes()
    ds.save_as(output_filename)
    print(f"DICOM saved: {output_filename} ({series_description})")

def run_inference(data, tr_val, model_path, output_dir, window_min, window_max, ref_dicom): 
    """
    Run the model and generate output data as PNG and DICOM
    """
    height, width, time_steps = data.shape
    model_input_len = 190
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
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
            
            # PREPROCESSING
            means = np.mean(neighbors, axis=1, keepdims=True)
            means[means < 1e-6] = 1.0 
            
            # Soft Normalization
            norm_neighbors = (neighbors - means) / (means + stabilizer)
            
            # INPUT TENSOR CONSTRUCTION
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
            
            # BATCH EXECUTION
            if len(pixel_coords) >= batch_size or (x == width-1 and y == height-1):
                input_tensor = torch.tensor(np.array(batch_data)).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor).cpu().numpy()
                
                for i, (px, py) in enumerate(pixel_coords):
                    
                    local_base = np.mean(data[px, py, :])
                    baseline_map[px, py] = local_base
                    
                    out_v = max(output[i, 0], 0.0)
                    out_q = max(output[i, 1], 0.0)

                    # Scale back to absolute amplitude
                    vamp[px, py] = (out_v * (local_base + stabilizer)) 
                    qamp[px, py] = (out_q * (local_base + stabilizer))                        
                    
                    vtime[px, py] = np.arctan2(output[i, 2], output[i, 3]) 
                    qtime[px, py] = np.arctan2(output[i, 4], output[i, 5]) - np.pi/2
                
                batch_data = []
                pixel_coords = []

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
    ax1.set_title('Ventilation Amp (Abs)') 
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(fv_map, cmap='gray', vmin=0, vmax=0.3) 
    ax2.set_title('Fractional Ventilation (FV)')
    ax2.axis('off')

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
    plt.savefig(os.path.join(output_dir, "VQWave_Summary.png"), dpi=300)
    plt.show()
    print(f"Summary PNG saved to {output_dir}")    

    save_as_dicom(vamp, ref_dicom, os.path.join(output_dir, "VQ_Vent_Amp.dcm"), "VQ-Wave Ventilation Amp")
    save_as_dicom(qamp, ref_dicom, os.path.join(output_dir, "VQ_Perf_Amp.dcm"), "VQ-Wave Perfusion Amp")
    save_as_dicom(fv_map, ref_dicom, os.path.join(output_dir, "VQ_FV.dcm"), "VQ-Wave Fractional Ventilation")
    save_as_dicom(vtime, ref_dicom, os.path.join(output_dir, "VQ_Vent_Phase.dcm"), "VQ-Wave Vent Phase", is_phase=True)
    save_as_dicom(qtime, ref_dicom, os.path.join(output_dir, "VQ_Perf_Phase.dcm"), "VQ-Wave Perf Phase", is_phase=True)    
    print(f"Saved maps to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-Wave Inference Script for DICOM Data")
    parser.add_argument("-d", "--dicom_dir", required=True, help="Path to directory containing DICOM time-series")
    parser.add_argument("-m", "--model", default="VQWaveNetwork_V11.pt", help="Path to the trained model (.pt)")
    parser.add_argument("-o", "--output", default="results", help="Output directory")
    parser.add_argument("--window_min", type=float, default=None, help="Min value for plotting")
    parser.add_argument("--window_max", type=float, default=None, help="Max value for plotting")
    args = parser.parse_args()
    
    data, tr_val, ref_ds = load_dicoms(args.dicom_dir)
    run_inference(data, tr_val, args.model, args.output, args.window_min, args.window_max, ref_ds)
    
