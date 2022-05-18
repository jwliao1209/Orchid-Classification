import argparse, os, zipfile
import gdown
import pandas as pd

def get_working_list(data_root):
    gdrive_id = [
        "1NGV8O6CAmxioeCAh0Q5-72xp4osaH0uI",
        "1rhSA9xNmLfOA7LS_rGFrxhaLe5Hf-FgI",
        "1B4C7IW4J4k8x43Ne1vr4r59teL-Kexc6",
        "1cy-s8R-w4jJ8jWAO3vXADB70bdlByUVT"
    ]
    output_path = [
        os.path.join(data_root, "train.csv"),
        os.path.join(data_root, "val.csv"),
        os.path.join(data_root, "test.csv"),
        os.path.join(data_root, "images.zip")
    ]
    return zip(gdrive_id, output_path)

def create_directory_by_permission(directory):
    permission = input(
        f"Not found directory {directory}. "
        "Create a new directory. (y/n)"
    )

    if permission == "y":
        os.makedirs(directory)

    elif permission == "n":
        raise RuntimeError("Reject creating a new directory.")

    else:
        print("Reply must be y or n.")
        create_directory_by_permission(directory)

def main(data_root):
    working_list = get_working_list(data_root)
    image_dir    = os.path.join(data_root, "image")

    if not os.path.exists(data_root):
        create_directory_by_permission(data_root)

    if not os.path.exists(image_dir):
        create_directory_by_permission(image_dir)

    for gdrive_id, output in working_list:
        if not os.path.exists(output):
            gdown.download(id=gdrive_id, output=output)

        if output.endswith(".zip"):
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(image_dir)
        else:
            df = pd.read_csv(output, names=["filename", "category"])
            df.to_csv(output, index=False)

    os.rename(
        os.path.join(image_dir, "label.csv"),
        os.path.join(data_root, "label.csv")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--data_root",
        type=str,
        default="data",
        help="root directory for downloaded data."
    )
    args = parser.parse_args()
    main(args.data_root)
