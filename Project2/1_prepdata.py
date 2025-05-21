import torch, glob
import pandas as pd, numpy as np
if __name__=="__main__":
    # Each .npy file in "imagedata" contains a single NumPy array in a binary format, making them faster to load and save
    # After processing the data files in the directory "imagedata", 
    # there is a generated pickle file "urothelial_cell_toy_data.pkl" containing dictionary of images and segmentation masks. 
    # This will serve as the STANDARD INPUT for subsequent analyses.
    urothelial_cells=dict(X=torch.stack([torch.tensor(np.load(f"imagedata/X/{i}.npy")) for i in range(len(glob.glob("imagedata/X/*.npy")))],axis=0),
                            y=np.stack([np.load(f"imagedata/y/{i}.npy") for i in range(len(glob.glob("imagedata/y/*.npy")))],axis=0))
    pd.to_pickle(urothelial_cells,"urothelial_cell_toy_data.pkl")