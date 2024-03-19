import sys
import os
# Get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the parent directory to sys.path
print(current_dir)
print(parent_dir)
sys.path.append(parent_dir)
print(parent_dir)

from image_dataset.torch_datasets import PdfImages