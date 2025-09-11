import numpy as np
import pandas as pd
import xarray as xr
import joblib, os
import argparse
from tqdm import tqdm 

parser = argparse.ArgumentParser(description = 'Used to extract crop from nc file')


def CropNc2Np(nc_path,var:str='rainfall',slide_T:int=6,slide_H:int = 32,slide_W:int = 32,size:int= 256):
    """
    Crops windows from an xarray DataArray and returns them as a list of numpy arrays.
    
    Args:
        nc_path (str): Path to the NetCDF file.
        var (str): The variable name to extract.
        slide_T (int): Time dimension slide distance.
        slide_H (int): Height dimension slide distance.
        slide_W (int): Width dimension slide distance.
        size (int): The size of the height and width dimensions for the crop window.
    
    Returns:
        list: A list of cropped numpy arrays.
    """
    da = xr.open_dataset(nc_path)[var]
    da = da.clip(min =0.0,max = 128.0)
    da = da.fillna(0.0)
    results =[]
    
    T,H,W = da.shape[0],da.shape[1],da.shape[2]
    
    # Correctly calculate the number of crops
    num_t_crops = len(range(0, T - 29 + 1, slide_T))
    num_h_crops = len(range(0, H - size + 1, slide_H))
    num_w_crops = len(range(0, W - size + 1, slide_W))
    
    total_crops = num_t_crops * num_h_crops * num_w_crops
    print(f'{total_crops} crops can be extracted.')
    
    # Use a list of tuples for h and w ranges, which is more readable
    h_coords = range(0, H - size + 1, slide_H)
    w_coords = range(0, W - size + 1, slide_W)
    
    np_da = da.to_numpy()

    # Add a tqdm progress bar to the outer loop
    for t in tqdm(range(0, T - 29 + 1, slide_T), desc="Cropping Time Slices"):
        for h in h_coords:
            for w in w_coords:
                crop = np_da[t:t + 29, h:h + size, w:w + size]
                results.append(crop)
    return results

if __name__ == "__main__":
    path = r"C:\NowcastNet\data\dataset\xrain\raw_nc\202207172200_XRAIN_10min.nc"
    
    joblib.dump(np.array(CropNc2Np(nc_path = path,
                                   slide_T = 20,
                                   slide_H = 64,
                                   slide_W = 64,)),r'C:\NowcastNet\data\dataset\xrain\figure\xrain_test.joblib')