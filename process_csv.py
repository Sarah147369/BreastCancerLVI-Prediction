import csv

import pandas as pd


def merge_csv(file1, file2, output_file):
    df1 = pd.read_csv(file1)
    df2 = pd.read_excel(file2)

    merged_df = pd.merge(df1, df2, on='patient_id', how='inner')

    merged_df.to_csv(output_file, index=False)

def mergeCSV(csv1_file_path, csv2_file_path, final_file_path):
    df1 = pd.read_csv(csv1_file_path)
    df2 = pd.read_csv(csv2_file_path)

    merged_df = pd.merge(df1, df2, on='id', how='left')

    merged_df.to_csv(final_file_path, index=False)


def read_csv_header(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)

    print(header)
    return header

def get_csv_dimensions(file_path):
    try:
        df = pd.read_csv(file_path)
        num_rows, num_cols = df.shape
        return num_rows, num_cols
    except Exception as e:
        return None, str(e)