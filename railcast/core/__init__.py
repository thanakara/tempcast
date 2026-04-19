DATASET_URL = "https://github.com/ageron/data/raw/main/ridership.tgz"
DATASETS_DIR = "railcast/datasets/"
DATAPATH = (
    DATASETS_DIR
    + "ridership_extracted/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv"
)
CONFIG_PATH = "conf/config.yaml"


__all__ = ["DATASET_URL", "DATAPATH", "CONFIG_PATH"]
