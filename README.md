# convert-to-safetensors

## Overview
This Python utility is designed to convert PyTorch model weights from '.bin' to '.safetensors' format.

## System Requirements
Before running the script, ensure you have the following installed:

- Python `3.10.x` or `3.11.x` must be installed on the system.
- Required Python packages can be installed by running `pip install -r requirements.txt`

## Steps to run the script
1. Clone this repository by running `git clone https://github.com/IBM/convert-to-safetensors.git`
2. Navigate to the project directory by running `cd convert-to-safetensors` and run `pip install -r requirements.txt` to install required dependencies.
3. Run `python convert_to_safetensor.py` and provide the path to the directory which contains the `pytorch_model.bin` file. Also, provide a path to a directory where you want the converted files to be saved. If the provided directory doesn't already exist, a new directory will be created. If no directory is provided, a new directory containing the converted files will be created with `_safetensors` as suffix.

## Usage 
```
usage: convert_to_safetensor.py [-h] [--source_dir SOURCE_DIR] [--destination_dir DESTINATION_DIR]

Python utililty to convert weights in `bin` format to `safetensors` format.

options:
  -h, --help  show this help message and exit
  --source_dir SOURCE_DIR
                        Path to the directory which contains the `pytorch_model.bin` file
  --destination_dir DESTINATION_DIR
                        Path to the directory where the model in safetensors format and related JSON files will be stored
```