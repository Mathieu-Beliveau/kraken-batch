import concurrent.futures
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import sys

CORPUS_LOCATION = '/home/strav/Dev/shared-corpus/'
LOCAL_BIN = '/home/strav/.local/bin/'
PROCESSED_FILENAME = "processed"
EXCEPTION_FILENAME = "exception"


def process_ocr(text_path):
    pdf_path = get_pdf(text_path)
    output = Path(f"{pdf_path.parent}/{pdf_path.stem}.txt")
    try:
        subprocess.call([f"{LOCAL_BIN}kraken", '-d', 'cuda:0', '-f', 'pdf', '-i', str(pdf_path), str(output),
                         'segment', '-bl', 'ocr', '-m', 'Gallicorpora%2B_best.mlmodel'])
        with open(f"{text_path}/{PROCESSED_FILENAME}_{datetime.now()}", "w"):
            print(f"{text_path} has been processed")
    except Exception:
        with open(f"{text_path}/{EXCEPTION_FILENAME}_{datetime.now()}", "w"):
            print(f"{text_path} has encountered an exception")


def is_processed(input_path: Path) -> bool:
    processed_lst = list(input_path.rglob("processed*"))
    return len(processed_lst) == 1


def get_pdf(input_path: Path) -> Path:
    pdf_list = list(input_path.rglob("*.pdf"))
    if len(pdf_list) == 1:
        return pdf_list[0]


def process_dirs_for_ocr(corpus_path: Path):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(lambda path: process_ocr(path), get_folders_to_process(corpus_path))


def get_folders_to_process(corpus_path: Path) -> list[Path]:
    return list(filter(lambda sub_path: sub_path.is_dir() and not is_processed(sub_path), corpus_path.iterdir()))


if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument("-c", "--corpus", default='/home/strav/Dev/shared-corpus/')
    args = argparse.parse_args()
    base_path = Path(CORPUS_LOCATION)
    process_dirs_for_ocr(base_path)

