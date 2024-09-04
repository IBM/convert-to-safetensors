import argparse
import os
import shutil
import torch
from collections import defaultdict
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from typing import Dict


def copy_remaining_files(source_folder: str, destination_folder: str) -> None:
    """
    Copy remaining files like config.json and other JSON files from source directory to the destination directory
    """
    for file in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file)
        if os.path.isfile(file_path) and not (file.endswith('.bin') or file.endswith('.py')):
            shutil.copy(file_path, destination_folder)

def get_shared_weights(tensors: Dict[str, torch.Tensor]) -> list:
    """
    Identify shared weights among tensors
    """
    tmp = defaultdict(list)
    for k, v in tensors.items():
        tmp[v.data_ptr()].append(k)
    return [names for names in tmp.values() if len(names) > 1]

def check_file_size(sf_filename: str, pt_filename: str) -> None:
    """
    Compare file sizes of converted and original files and raise an error if the size difference is more than 1%
    """

    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
            - {sf_filename}: {sf_size}
            - {pt_filename}: {pt_size}
            """
        )

def convert_file(pt_filename: str, sf_filename: str, copy_add_data: bool=True) -> None:
    src_dir = os.path.dirname(pt_filename)
    dest_dir = os.path.dirname(sf_filename)

    # Note: When 'weights_only' is set to True, the unpickler is limited to loading only tensors, primitive types, and dictionaries. 
    # This helps mitigate the risk of executing arbitrary code that could be embedded in malicious pickle data. 
    # By default, if 'weights_only' is set to False, the function will use Python's pickle module without restrictions, 
    # which can be unsafe if the source of the data is not trusted. Reference: https://pytorch.org/docs/stable/generated/torch.load.html
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared_ = get_shared_weights(loaded)

    # Iterate through the identified shared weights and remove any redundant references
    # from the loaded dictionary to ensure that only the primary shared tensors are retained
    for shared_weights in shared_:
        for name in shared_weights[1:]:
            loaded.pop(name)
    
    # The 'half' method used below converts the tensor to half-precision floating point (FP16) format 
    # instead of single precision (FP32) format to keep it compatible with the original model format
    # Reference: https://pytorch.org/docs/stable/generated/torch.Tensor.half.html
    loaded = {k: v.contiguous().half() for k, v in loaded.items()}

    # Save safetensors file with other JSON files in the destination directory
    os.makedirs(dest_dir, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    if copy_add_data:
        copy_remaining_files(src_dir, dest_dir)

    # Validate that the tensors in the converted file match the tensors in the original file
    # and raise an error with the tensor key wherever there's a mismatch
    reloaded = load_file(sf_filename)
    for k, v in loaded.items():
        if not torch.equal(v, reloaded[k]):
            raise RuntimeError(f"Mismatch in tensors for key {k}.")

def main():
    DESCRIPTION = """
    Simple utility tool to convert weights in `bin` format to `safetensors` format.
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--src_directory",
        type=str,
        help="Path to the directory which contains the `pytorch_model.bin` file",
    )
    parser.add_argument(
        "--dest_directory",
        type=str,
        help="Path to the directory where the model in safetensors format and related JSON files will be stored",
    )

    args = parser.parse_args()
    src_dir = args.src_directory.strip()
    dest_dir = args.dest_directory.strip()
    
    # If the destination directory is not provided, create a new directory with '_safetensors' suffix 
    # in the same path as source directory to keep the converted files
    if not dest_dir:
        model_name = os.path.basename(os.path.normpath(src_dir))
        dest_dir = os.path.join(src_dir, model_name + "_safetensors")

    if "pytorch_model.bin" not in os.listdir(src_dir):
        raise RuntimeError("pytorch_model.bin file not found. Please ensure the correct source directory is specified.")
    else:
        convert_file(os.path.join(src_dir, "pytorch_model.bin"), os.path.join(dest_dir, "model.safetensors"), copy_add_data=True)

if __name__ == "__main__":
    main()
