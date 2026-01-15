# /// script
# dependencies = ["xarray", "numpy", "matplotlib", "seaborn", "netCDF4"]
# ///

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Paths
    input_file = "usecases/Granger_causality/gc_result.nc"
    output_file = "usecases/Granger_causality/gc_heatmap.png"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Load data
    print(f"Loading {input_file}...")
    gc = xr.load_dataarray(input_file)
    
    # Check structure
    print("Dimensions:", gc.dims)
    print("Coordinates:", gc.coords)

    # Average over trials and times to get a robust estimate of connectivity strength
    # Dimensions are ('trials', 'roi', 'times', 'direction')
    gc_mean = gc.mean(dim=['trials', 'times'])

    # Extract connectivity values
    # We expect 'direction' to contain 'x->y' and 'y->x'
    try:
        val_xy = gc_mean.sel(direction='x->y').item()
        val_yx = gc_mean.sel(direction='y->x').item()
    except KeyError as e:
        print(f"Error selecting direction: {e}")
        print("Available directions:", gc.direction.values)
        return

    # Construct the connectivity matrix (Source -> Target)
    # Rows = Source, Cols = Target
    #    x    y
    # x  0   x->y
    # y y->x  0
    matrix = np.array([[0, val_xy], [val_yx, 0]])
    labels = ['x', 'y']

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='viridis')
    plt.title("Mean Directed Granger Causality")
    plt.xlabel("Target")
    plt.ylabel("Source")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Heatmap saved to {output_file}")

if __name__ == "__main__":
    main()
