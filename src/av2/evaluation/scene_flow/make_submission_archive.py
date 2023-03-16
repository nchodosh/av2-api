from zipfile import ZipFile
import json
from pathlib import Path
import argparse
import pandas as pd
from rich.progress import track
from typing import Dict

def validate(submission_root: Path, fmt: Dict[str, int]):
    for filename in track(fmt.keys(), description = 'Validating...'):
        input_file = submission_root / filename
        if not input_file.exists():
            raise FileNotFoundError(f'{str(input_file)} not found in submission directory')
        pred = pd.read_feather(input_file)

        if ('flow_tx_m' not in pred.columns or
            'flow_ty_m' not in pred.columns or
            'flow_tz_m' not in pred.columns):
            raise ValueError(f'{str(input_file)} does not contain the correct columns')

        if len(pred.columns) > 3:
            raise ValueError(f'{str(input_file)} contains extra columns')

        if len(pred) != fmt[filename]:
            raise ValueError(f'{str(input_file)} has {len(pred)} rows but it should have {fmt[filename]}')

def zip(submission_root: Path, fmt: Dict[str, int], output_file: Path):
    with ZipFile(output_file, 'w') as myzip:
        for filename in track(fmt.keys(), description = 'Zipping...'):
            input_file = submission_root / filename
            myzip.write(input_file, arcname=filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('make_submission_archive',
                                     description='Validate a set of submission files and then '
                                     'package them into a zip file. Files should be in the format '
                                     'specified in SUBMISSION_FORMAT.md')

    parser.add_argument('submission-root', type=str, help='location of the submission files')
    parser.add_argument('--output-file', type=str, default='submission.zip',
                        help='name of output archive file')
    args = parser.parse_args()

    with open('test_submission_format.json', 'r') as f:
        fmt = json.load(f)

    submission_root = Path(args.submission_root)
    output_file = Path(args.output_file)
    try:
        validate(submission_root, fmt)
    except Exception as e:
        print('Input validation failed with:')
        print(e)

    zip(submission_root, fmt, output_file)
        
        
