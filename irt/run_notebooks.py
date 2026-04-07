"""
IRT Notebook Executor (portable version)
=========================================
Minimal Jupyter notebook executor — works without jupyter/nbconvert installed.
Runs each code cell with exec(), captures stdout/stderr/matplotlib figures,
saves outputs back into the .ipynb file (nbformat v4 format).

Usage:
    python run_notebooks.py            # run all notebooks in current directory
    python run_notebooks.py IRT_P2_PCM.ipynb   # run one specific notebook
"""

import json, io, sys, os, base64, traceback, glob, signal, builtins, re, tempfile
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm


# ── Korean font auto-detection ────────────────────────────────────────────
def _setup_korean_font():
    """Find and register a Korean-capable font, then configure matplotlib."""
    ko_candidates = [
        # Linux / Debian-based
        '/usr/share/fonts-droid-fallback/truetype/DroidSansFallback.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        # macOS
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/Library/Fonts/AppleGothic.ttf',
        # Windows (via WSL or native path lookup)
        'C:/Windows/Fonts/malgun.ttf',
        'C:/Windows/Fonts/gulim.ttc',
    ]
    for path in ko_candidates:
        if os.path.exists(path):
            # Remove any conflicting registration with same name but wrong file
            existing_name = None
            try:
                prop = fm.FontProperties(fname=path)
                existing_name = prop.get_name()
                fm.fontManager.ttflist = [
                    f for f in fm.fontManager.ttflist
                    if not (f.name == existing_name and f.fname != path)
                ]
                fm.fontManager.addfont(path)
            except Exception:
                pass
            mpl.rcParams['font.family'] = ['DejaVu Sans', existing_name or 'sans-serif']
            mpl.rcParams['axes.unicode_minus'] = False
            return path
    # Fallback: just suppress unicode_minus warning
    mpl.rcParams['axes.unicode_minus'] = False
    return None

_ko_font_path = _setup_korean_font()


NB_DIR = os.path.dirname(os.path.abspath(__file__))


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Cell timed out")


def _make_stream(name, text):
    return {'output_type': 'stream', 'name': name, 'text': text}


def _make_display(png_b64):
    return {
        'output_type': 'display_data',
        'metadata': {},
        'data': {'image/png': png_b64, 'text/plain': ['<Figure>']},
    }


def _make_error(etype, evalue, tb):
    return {
        'output_type': 'error',
        'ename': etype,
        'evalue': evalue,
        'traceback': tb.splitlines(),
    }


def execute_notebook(nb_path, cell_timeout=120, verbose=True):
    nb_name = os.path.basename(nb_path)
    if verbose:
        print(f'\n{"="*60}')
        print(f'  {nb_name}')
        print(f'{"="*60}')

    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    cells   = nb.get('cells', [])
    tmpdir  = tempfile.mkdtemp()
    g       = {
        '__builtins__': builtins,
        '__name__':     '__main__',
        'tmpdir':       tmpdir,   # pre-defined so plot-saving cells work
    }

    # Intercept plt.show() to capture figures as base64 PNG
    captured_figs = []

    def _capture_show(*args, **kwargs):
        for fig_num in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(fig_num).savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            captured_figs.append(base64.b64encode(buf.read()).decode())
        plt.close('all')

    plt.show = _capture_show

    total_errors = 0
    exec_count   = 0

    for cell_idx, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue

        src = ''.join(cell.get('source', []))
        if not src.strip():
            cell['outputs'] = []
            cell['execution_count'] = None
            continue

        # Strip IPython magic / shell commands
        cleaned = re.sub(r'^\s*%.*$',  '', src,     flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*!.*$',  '', cleaned, flags=re.MULTILINE)

        captured_figs.clear()
        outputs = []
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        exc_info = None
        if hasattr(signal, 'SIGALRM'):          # SIGALRM not available on Windows
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(cell_timeout)
        try:
            exec(compile(cleaned, f'<{nb_name}[{cell_idx}]>', 'exec'), g)
        except TimeoutError:
            exc_info = ('TimeoutError', f'Cell {cell_idx} timed out after {cell_timeout}s', '')
        except SystemExit:
            pass
        except Exception:
            tb       = traceback.format_exc()
            exc_type = type(sys.exc_info()[1]).__name__
            exc_val  = str(sys.exc_info()[1])
            exc_info = (exc_type, exc_val, tb)
            total_errors += 1
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        stdout_val = sys.stdout.getvalue()
        stderr_val = sys.stderr.getvalue()
        sys.stdout = old_out
        sys.stderr = old_err
        exec_count += 1

        if stdout_val:
            outputs.append(_make_stream('stdout', stdout_val))
        if stderr_val:
            # Suppress common harmless warnings from output
            noisy = ['UserWarning', 'DeprecationWarning', 'FutureWarning', 'RuntimeWarning']
            filtered = '\n'.join(
                l for l in stderr_val.splitlines()
                if not any(n in l for n in noisy)
            ).strip()
            if filtered:
                outputs.append(_make_stream('stderr', stderr_val))
        for png in captured_figs:
            outputs.append(_make_display(png))
        if exc_info:
            outputs.append(_make_error(*exc_info))
            if verbose:
                print(f'  [cell {cell_idx:02d}] ❌ {exc_info[0]}: {exc_info[1][:100]}')
        else:
            if verbose and (stdout_val.strip() or captured_figs):
                n_figs  = len(captured_figs)
                preview = stdout_val.strip().splitlines()
                print(f'  [cell {cell_idx:02d}] ✅'
                      + (f'  figs={n_figs}' if n_figs else '')
                      + (f'  {preview[0][:70]}' if preview else ''))

        cell['outputs']         = outputs
        cell['execution_count'] = exec_count

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    if verbose:
        status = '✅  0 errors' if total_errors == 0 else f'⚠️  {total_errors} errors'
        print(f'  → Saved  ({status})')
    return total_errors


def main():
    parser = argparse.ArgumentParser(
        description='Execute IRT Jupyter notebooks without Jupyter installed.'
    )
    parser.add_argument('notebooks', nargs='*',
                        help='Specific .ipynb files to run (default: all in this directory)')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Per-cell timeout in seconds (default: 120)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-cell output')
    args = parser.parse_args()

    if args.notebooks:
        notebooks = [os.path.abspath(p) for p in args.notebooks]
    else:
        notebooks = sorted(glob.glob(os.path.join(NB_DIR, '*.ipynb')))

    if not notebooks:
        print('No notebooks found.')
        return

    total_errors = 0
    summary      = []

    for nb_path in notebooks:
        try:
            nerr = execute_notebook(nb_path, cell_timeout=args.timeout,
                                    verbose=not args.quiet)
            summary.append((os.path.basename(nb_path), nerr))
            total_errors += nerr
        except Exception as exc:
            print(f'FATAL: {nb_path}: {exc}')
            summary.append((os.path.basename(nb_path), -1))

    print(f'\n{"="*60}')
    print('  SUMMARY')
    print(f'{"="*60}')
    for name, nerr in summary:
        if nerr == 0:
            icon = '✅'
        elif nerr < 0:
            icon = '💥 FATAL'
        else:
            icon = f'⚠️  ({nerr} cell errors)'
        print(f'  {icon}  {name}')
    print(f'\nTotal cell errors: {total_errors}')


if __name__ == '__main__':
    main()
