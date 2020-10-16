#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 12:16:49 2020

@author: jkreula
"""

import os
import tarfile
import urllib
import numpy as np
import pandas as pd
import pyprind # For progress bar
from time import time # For timing

def fetch_data(data_save_path: str, data_url: str, verbose: bool = True) -> None:
    '''
    Fetch data from data_url and save it to data_save_path.

    Parameters
    ----------
    data_save_path : str
        Path to folder for saving data.
    data_url : str
        URL to extract data from.
    verbose : bool, optional
        Boolean argument to control verbosity. The default is True.

    Returns
    -------
    None
    '''
    time_start = time()
    
    # Create folder if needed
    if not os.path.isdir(data_save_path):
        os.makedirs(data_save_path)
        
    # Filename from URL
    file_to_be_downloaded = DATA_URL[DATA_URL.rindex("/")+1:]
    file_path = os.path.join(data_save_path,file_to_be_downloaded)
    
    # Retrieve data
    if verbose: print("Starting to fetch data...")
    urllib.request.urlretrieve(data_url,file_path)
    
    # Extract contents
    with tarfile.open(file_path, 'r:gz') as tar:
        try:
            tar.extractall(path=data_save_path)
            if verbose: print(f"Data fetch successful. It took {time()-time_start:.2f}s.")
        except:
            print("Error in extracting data.")
    
def find_num_files(data_path: str) -> int:
    num_files = 0
    for split in ('test','train'):
        for label in ('pos','neg'):
            filepath = os.path.join(data_path,split,label)
            num_files += len(os.listdir(filepath))
    return num_files
    

def save_data_as_csv(data_path: str, save_filename: str, verbose: bool = True) -> None:
    '''
    Preprocess data by reading them into a single pandas DataFrame, labelling the reviews
    as 1 (positive) or 0 (negative) by mapping the 'pos' and 'neg' labels in the data.
    The rows of the resulting df are then shuffled with train_test_split in mind.
    The shuffled df is saved as csv for future use.
    
    Parameters
    ----------
    data_path : str
        Path to data folder.
    save_filename : str
        Name of csv for saving
    verbose : bool, optional
        Boolean argument to control verbosity. The default is True.

    Returns
    -------
    None
    '''
    # Initialise progress bar
    num_files = find_num_files(data_files)
    progress_bar = pyprind.ProgBar(num_files)
    # Dictionary to map labels
    label_dict = {'pos': 1, 'neg': 0}
    # Initialise DataFrame
    df = pd.DataFrame()
    if verbose: print("Starting to preprocess data...")
    # Loop over all files and add contents to df
    for split in ('test','train'):
        for label in ('pos','neg'):
            filepath = os.path.join(data_path,split,label)
            for file in os.listdir(filepath):
                with open(os.path.join(filepath,file),"r",encoding="utf-8") as f:
                    text = f.read()
                df = df.append([[label_dict[label],text]],ignore_index=True)
                # Update progress bar
                progress_bar.update()
    # Column names
    df.columns = ["Label","Review_text"]
    # Set seed for shuffling index
    np.random.seed(0)
    # Create a new DataFrame by shuffling index. Required for convenience in ¨¨
    # manual train test split.
    df=df.reindex(np.random.permutation(df.index))
    if verbose: 
        print("Data processing complete.")
        print("Saving data to csv...")
    try:
        df.to_csv(os.path.join(DATA_PATH,save_filename),index=False,encoding="utf-8")
    except:
        print("Error in saving data.")
    else:
        print("Data saved to csv.")

if __name__ == "__main__":
    
    # Current working directory
    CWD = os.getcwd()
    
    # Path for saving data
    data_folder = "Data"
    DATA_PATH = os.path.join(CWD,data_folder)
    
    # URL to download data from
    DOWNLOAD_ROOT = r"http://ai.stanford.edu/~amaas/data/"
    DATA_URL = DOWNLOAD_ROOT + r"sentiment/aclImdb_v1.tar.gz"
    
    # Fetch data    
    fetch_data(DATA_PATH, DATA_URL)
    
    data_files = os.path.join(DATA_PATH,"aclImdb")
    save_data_as_csv(data_files,"imdb_data.csv")