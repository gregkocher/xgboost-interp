"""
Utilities for loading and processing data.
"""

import glob
import itertools
import pandas as pd
from typing import List, Optional


class DataLoader:
    """Utility class for loading and processing data files."""
    
    @staticmethod
    def load_parquet_files(data_dir_path: str, 
                          cols_to_load: Optional[List[str]] = None,
                          num_files_to_read: int = 1000) -> pd.DataFrame:
        """
        Load data from multiple parquet files.
        
        Args:
            data_dir_path: Directory containing parquet files
            cols_to_load: List of columns to load (None for all columns)
            num_files_to_read: Maximum number of files to read
            
        Returns:
            Combined DataFrame from all loaded files
        """
        file_pattern = f"{data_dir_path}/*.parquet"
        all_files = glob.glob(file_pattern)
        
        if not all_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir_path}")
        
        print(f"Loading data from: {data_dir_path}")
        files = itertools.islice(all_files, num_files_to_read)
        
        if cols_to_load:
            df = pd.concat(
                pd.read_parquet(parquet_file, columns=cols_to_load)
                for parquet_file in files
            )
        else:
            df = pd.concat(
                pd.read_parquet(parquet_file)
                for parquet_file in files
            )
        
        print(f"[OK] Loaded {len(df)} rows")
        if cols_to_load:
            print(f"   Columns: {cols_to_load}")
        
        return df
