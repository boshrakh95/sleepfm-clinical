"""
STAGES Data Preparation for SleepFM
====================================

This package contains scripts to convert CogPSGFormerPP preprocessed STAGES data
into the HDF5 format required by SleepFM.

Directory Structure:
- convert_to_hdf5.py: Main conversion script (NumPy → HDF5)
- prepare_labels.py: Prepare cognitive targets and demographics
- create_splits.py: Generate train/val/test splits
- validate_data.py: Validation utilities
- config_stages_conversion.yaml: Configuration file
"""

__version__ = "1.0.0"
