import os
import shutil
from datetime import datetime

import pandas as pd


RAW_DATA = "data/raw/hour.csv"
NEW_BATCH_FOLDER = "data/new_batches"
LOG_FILE = "artifacts/ingestion_log.csv"


def ingest_new_batches():

    if not os.path.exists(NEW_BATCH_FOLDER):
        print("No new batch folder found.")
        return

    batch_files = [
        f for f in os.listdir(NEW_BATCH_FOLDER)
        if f.endswith(".csv")
    ]

    if len(batch_files) == 0:
        print("No new CSV files to ingest.")
        return

    master_df = pd.read_csv(RAW_DATA)

    log_entries = []

    for file in batch_files:

        file_path = os.path.join(
            NEW_BATCH_FOLDER,
            file
        )

        new_df = pd.read_csv(file_path)

        rows = len(new_df)

        master_df = pd.concat(
            [master_df, new_df],
            ignore_index=True
        )

        log_entries.append({
            "file": file,
            "rows": rows,
            "ingested_at": datetime.now()
        })

        processed_folder = os.path.join(
            NEW_BATCH_FOLDER,
            "processed"
        )

        os.makedirs(
            processed_folder,
            exist_ok=True
        )

        shutil.move(
            file_path,
            os.path.join(
                processed_folder,
                file
            )
        )

    master_df.drop_duplicates(inplace=True)

    master_df.to_csv(
        RAW_DATA,
        index=False
    )

    log_df = pd.DataFrame(log_entries)

    if os.path.exists(LOG_FILE):

        existing = pd.read_csv(LOG_FILE)

        log_df = pd.concat(
            [existing, log_df],
            ignore_index=True
        )

    log_df.to_csv(
        LOG_FILE,
        index=False
    )

    print("Batch ingestion completed.")

    print(f"Rows after merge: {len(master_df)}")


if __name__ == "__main__":
    ingest_new_batches()