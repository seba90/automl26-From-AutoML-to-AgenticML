import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_documented_python_entrypoints_accept_help():
    for script in ('preprocess_criteo.py', 'experiment.py', 'bayes_search.py',
                   'train.py', 'analyze.py'):
        result = subprocess.run(
            [sys.executable, str(ROOT / 'src' / script), '--help'],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_documented_download_script_has_valid_shell_syntax():
    result = subprocess.run(
        ['bash', '-n', str(ROOT / 'scripts' / 'download_data.sh')],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
