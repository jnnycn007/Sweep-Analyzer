**Project Layout**
- `gui.py`: Main GUI and analysis logic.
- `eegbin.py`: SW3/eegbin parsing and data models.
- `util.py`: Shared helpers and optional imgui utilities.
- `docs/`: Human‑readable documentation.
- `hooks/`: PyInstaller runtime hooks.
- `.github/workflows/build.yml`: CI build pipeline.

**Running Locally**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python gui.py
```

**Packaging and CI**
- CI builds for macOS, Windows, and Linux when tags are pushed.
- The workflow uses PyInstaller and bundles Tk and Pillow runtime dependencies.
- The macOS app is unsigned and will trigger Gatekeeper warnings.

**Development Conventions**
- Prefer explicit units in variable names when ambiguous (e.g., `timestamp_s`).
- Keep analysis logic in `gui.py` and file format logic in `eegbin.py`.
- `eegbin.py` is treated as the single source of truth for file parsing.

**Testing**
- Run unit/smoke tests:
```bash
./venv/bin/python -m unittest discover -s tests -v
```
- Use the Open Excimer dataset tuning script to validate long-run behavior:
```bash
./venv/bin/python scripts/tune_open_excimer.py --dataset-dir "OSLUV Data/OSLUV Experiments/Open Excimer"
```
