import concurrent.futures
import os
import re
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import sys

CORPUS_LOCATION = '/home/strav/Dev/shared-corpus/'
LOCAL_BIN = '/home/strav/.local/bin/'
PROCESSED_FILENAME = "processed"
EXCEPTION_FILENAME = "exception"


def process_ocr(text_path: Path):
    pdf_path = get_pdf(text_path)
    output = Path(f"{pdf_path.parent}/{pdf_path.stem}")
    try:
        subprocess.call([f"{LOCAL_BIN}kraken", '-d', 'cuda', '-f', 'pdf', '-i', str(pdf_path), str(output),
                         'segment', '-bl', 'ocr', '-m', 'Gallicorpora%2B_best.mlmodel'])
        post_process_results(pdf_path.parent, pdf_path.stem)
        with open(f"{text_path}/{PROCESSED_FILENAME}_{datetime.now()}", "w"):
            print(f"{text_path} has been processed")
    except Exception:
        with open(f"{text_path}/{EXCEPTION_FILENAME}_{datetime.now()}", "w"):
            print(f"{text_path} has encountered an exception")


def post_process_results(output_path: Path, output_pattern: str):
    result_files = output_path.rglob(f"{output_pattern}.pdf_*")
    for file in result_files:
        with open(file, 'r') as file_obj:
            file_data = file_obj.read()
            file_data = file_data.replace('¬\n', '')
        with open(file, 'w') as file_obj:
            file_obj.write(file_data)


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


def recompose_folders():
    base_path = Path(CORPUS_LOCATION)
    result_dirs = base_path.glob("*_*-*")
    range_match_expression = r"(.*)_(\d*)-(\d*)"
    file_page_match_expression = f"(.*)_(\d*-\d*).pdf_(\d*)"
    for dir in result_dirs:
        range_match = re.search(range_match_expression, dir.name).groups()
        source_dir = range_match[0]
        start_range = int(range_match[1])
        dir_files = dir.glob("*_*")
        for file in dir_files:
            file_matches = re.search(file_page_match_expression, file.name)
            if file_matches is None:
                continue
            file_group_matches = file_matches.groups()
            page = int(file_group_matches[2])
            prefix = file_group_matches[0]
            absolute_page = page + start_range - 1
            new_page_name = f"{prefix}.pdf_{str(absolute_page).zfill(6)}"
            os.rename(str(file.absolute()), f"{str(dir.parent)}/{source_dir}/{new_page_name}")

if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument("-c", "--corpus", default='/home/strav/Dev/shared-corpus/')
    argparse.add_argument("-r", "--recompose", default=False, type=bool)
    args = argparse.parse_args()
    if args.recompose is True:
        recompose_folders()
    base_path = Path(CORPUS_LOCATION)
    process_dirs_for_ocr(base_path)

