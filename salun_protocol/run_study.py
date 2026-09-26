"""Whole study, resumable: pilot tuning on seed 1, then ten test trials (seeds 2-11) per ratio.

python run_study.py              # tune, then trials with the freshly selected settings
python run_study.py --recorded   # skip tuning; trials use the settings in selected/
"""

import subprocess
import sys
from pathlib import Path

from classification import WORK

HERE = Path(__file__).resolve().parent


def run(*args):
    subprocess.run(
        [sys.executable, '-u', str(HERE / 'classification.py'), *args], check=True, cwd=HERE
    )


def main():
    recorded = '--recorded' in sys.argv
    for ratio in ('.1', '.5'):
        tag = f'forget{int(float(ratio)*100)}'
        settings = HERE / 'selected' / f'{tag}.json'
        if not recorded:
            settings = WORK / 'seed1' / tag / 'tuning' / 'selected.json'
            if not settings.exists():
                run('--seed', '1', '--ratio', ratio, '--tune')
        for seed in range(2, 12):
            run('--seed', str(seed), '--ratio', ratio, '--settings', str(settings))
    subprocess.run([sys.executable, str(HERE / 'summarize.py')], check=True)


if __name__ == '__main__':
    main()
