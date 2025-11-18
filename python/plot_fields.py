import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    filepath = 'data/ez_final.bin'
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return
        
    size = os.path.getsize(filepath)
    nx = ny = int(np.sqrt(size // 4))
    
    data = np.fromfile(filepath, dtype=np.float32).reshape((nx, ny))
    
    plt.figure(figsize=(8, 6))
    vmax = np.max(np.abs(data))
    if vmax == 0: vmax = 1.0
    
    plt.imshow(data, cmap='seismic', vmin=-vmax, vmax=vmax, origin='lower')
    plt.colorbar(label='Ez (V/m)')
    
    tfsf_low = 50
    tfsf_high = nx - 50
    plt.plot([tfsf_low, tfsf_high, tfsf_high, tfsf_low, tfsf_low],
             [tfsf_low, tfsf_low, tfsf_high, tfsf_high, tfsf_low],
             'w--', linewidth=1.5, label='TF/SF Boundary')
             
    plt.title("Ez field — 2D FDTD, GPU, t = 2000 × dt")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('data/ez_field.png', dpi=300)
    print("Saved data/ez_field.png")

if __name__ == '__main__':
    main()