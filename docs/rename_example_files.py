from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pathlib import Path
from shutil import copy

def main():
    parser = ArgumentParser(description="Rename example files for upload to CMS server", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("directory", type=Path, help="Directory containing example files")
    parser.add_argument("--output", type=Path, default="CMS", help="Output directory")
    args = parser.parse_args()

    input_dir = args.directory
    assert input_dir.exists(), f"Directory {input_dir} does not exist"
    output = args.output
    output.mkdir(exist_ok=True, parents=True)

    for file in input_dir.glob("*"):
        if not file.is_file():
            continue
        new_name = f"optimap-{file.name}_.webm"
        copy(file, output / new_name)
        print(f"Renamed {file.name} to {new_name}")

if __name__ == "__main__":
    main()