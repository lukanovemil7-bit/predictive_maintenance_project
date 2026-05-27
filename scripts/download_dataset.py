from pathlib import Path
import urllib.request

UCI_CSV_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "predictive_maintenance.csv"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from: {UCI_CSV_URL}")
    urllib.request.urlretrieve(UCI_CSV_URL, OUTPUT_PATH)
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
