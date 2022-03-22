import pandas as pd
from os import listdir, remove, path

data_frames = []
INPUT_DIR = "dataset"
OUTPUT_FILE_NAME = "final-merged-dataset.csv"


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    return [
        path.join(path_to_dir, filename)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def read_data(files):
    for file in files:
        df = pd.read_csv(file, index_col=None, header=0)
        data_frames.append(df)


def merge_data():
    return pd.concat(data_frames, axis=0, ignore_index=True)


def write_csv(df):
    df.to_csv(OUTPUT_FILE_NAME, encoding="utf-8", index=False)


if __name__ == "__main__":
    try:
        remove(OUTPUT_FILE_NAME)
        print("Creating new file.")
    except OSError:
        pass

    csv_files = find_csv_filenames(INPUT_DIR)
    print("CSV files: ", *csv_files)

    read_data(csv_files)
    print("Read all the csv.")

    final_data_frame = merge_data()
    print("Merged all the CSV")

    write_csv(final_data_frame)
    print("Merged dataset CSV: ", OUTPUT_FILE_NAME)
