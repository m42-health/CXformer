import os
import multiprocessing
import pydicom
from PIL import Image
import pandas as pd
from convert_to_jpeg import read_dicom
from glob import iglob, glob
import time
from tqdm import tqdm
import tempfile
import psutil
from pathlib import Path
import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument(
        "--num_processes", type=int, default=1, help="Number of processes to use"
    )
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--chunksize", type=int, default=10000, help="Chunk size")
    parser.add_argument(
        "--input_extension",
        choices=["dcm", "dicom", "png"],
        help="Extension of the input files",
    )

    return parser.parse_args()


def convert_file(input_file, output_save_path, fail_fname="failed_to_convert.txt"):
    """
    Convert a single DICOM file to a JPEG image and save it to the output directory.
    Returns a dictionary containing the DICOM metadata for the file.
    """
    # Load the DICOM file using the pydicom library
    file_extension = os.path.splitext(input_file)[1]
    metadata = {}
    if file_extension in {".dcm", ".dicom"}:
        try:
            img, metadata = read_dicom(input_file, return_metadata=True)
            image = Image.fromarray(img)  # convert to PIL
        except Exception as e:
            print("=" * 50)
            print(f"Error reading dcm image for {input_file}: {e}")
            print("=" * 50)

            with open(fail_fname, "a+") as f:
                f.write(f"{input_file}]\n")

            metadata = {}
            metadata["jpg_fpath"] = ""
            metadata["input_fpath"] = input_file
            return metadata
    else:
        assert (
            file_extension == ".png"
        ), f"Unsupported input file extension {file_extension}"
        try:
            image = Image.open(input_file)
            image = image.convert("L")
            image = np.array(image)
            image = (
                (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            ).astype(np.uint8)
            image = Image.fromarray(image)
        except:
            print("=" * 50)
            print(f"Error reading png image for {input_file}")
            print("=" * 50)

            with open(fail_fname, "a+") as f:
                f.write(f"{input_file}]\n")

            metadata = {}
            metadata["jpg_fpath"] = ""
            metadata["input_fpath"] = input_file
            return metadata

    if not os.path.isfile(output_save_path):
        # ensure parent directory always exists
        path = Path(output_save_path)
        os.makedirs(str(path.parent.absolute()), exist_ok=True)
        image.save(output_save_path)

    # Return the metadata for the DICOM Image
    metadata["jpg_fpath"] = output_save_path
    metadata["input_fpath"] = input_file

    return metadata


def convert_files(
    input_dir, output_dir, num_processes=1, all_input_fpaths: list = None
):
    """
    Convert all DICOM files in the input directory to JPEG images and save them to the output directory,
    using the specified number of worker processes.
    Returns a Pandas DataFrame containing the DICOM metadata for all files.
    """
    start_time = time.time()

    batch_size = 5000
    chunks = [
        all_input_fpaths[i : i + batch_size]
        for i in range(0, len(all_input_fpaths), batch_size)
    ]

    pbar = tqdm(total=len(all_input_fpaths))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pool = multiprocessing.Pool(num_processes)

    manager = multiprocessing.Manager()
    num_completed = manager.Value("i", 0)

    results = []

    def update(*a):
        pbar.update(1)

    for chunk in tqdm(chunks, desc="Parsing chunks", total=len(chunks)):
        for f in chunk:
            file_extension = os.path.splitext(f)[1]
            if file_extension == ".dcm":
                output_save_path = f.replace(input_dir, output_dir).replace(
                    ".dcm", ".jpg"
                )
            elif file_extension == ".dicom":
                output_save_path = f.replace(input_dir, output_dir).replace(
                    ".dicom", ".jpg"
                )
            else:
                assert (
                    file_extension == ".png"
                ), f"Unsupported input extension {file_extension}"
                output_save_path = f.replace(input_dir, output_dir).replace(
                    ".png", ".jpg"
                )
            result = pool.apply_async(
                convert_file,
                args=(f, output_save_path, f"failed_to_convert.txt"),
                callback=update,
            )
            results.append(result)
    metadata = [r.get() for r in results]
    end_time = time.time()

    print("=" * 50)
    print("Time taken to run file conversion: ", (end_time - start_time) / 60, " mins")
    print("=" * 50)

    start_time = time.time()

    end_time = time.time()
    print("=" * 50)
    print("Time taken to write metadata: ", (end_time - start_time) / 60, " mins")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_arguments()

    # Access arguments
    num_processes = args.num_processes
    input_dir = args.input_dir
    output_dir = args.output_dir
    chunksize = args.chunksize
    input_extension = args.input_extension

    print(f"Number of processes: {num_processes}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunksize}")

    os.makedirs(output_dir, exist_ok=True)

    input_file_paths = glob(
        os.path.join(input_dir, "**", f"*.{input_extension}"), recursive=True
    )
    print(f"Files to convert: {len(input_file_paths)}")

    st = 0
    chunk_paths = []
    while st < len(input_file_paths):
        chunk_paths.append(input_file_paths[st : st + chunksize])
        st = st + chunksize

    n_chunks = len(chunk_paths)

    for curr_chunk in tqdm(chunk_paths, desc="Parsing chunks"):
        convert_files(input_dir, output_dir, num_processes, input_file_paths)

    print(f"Finished Successfully...")
