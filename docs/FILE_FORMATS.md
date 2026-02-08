**SW3/EEGBIN Overview**
SW3/eegbin files store optical irradiance measurements captured by a goniometer. The GUI loads two related on‑disk formats: a legacy binary (v1) and a JSON+binary hybrid (v2/v3). Parsing and writing logic lives in `eegbin.py`.

**Version 1 (Legacy Binary)**
- Stream of fixed‑layout records.
- Each record contains azimuth/elevation, distance, wavelength range/step, and an integral irradiance value plus spectral samples.
- Parsed via `load_legacy_eegbin1`.

**Version 2/4/5 (JSON + Base64)**
- File structure: UTF‑8 JSON metadata, then separator `\n\nB64 FOLLOWS\n\n`, then base64 row payload.
- Metadata includes scan‑level fields and optional per‑row notes.
- Each row is a packed struct plus an optional spectral array.

**Version 3 (EEGBIN3 / v6)**
- File structure: magic `EEGBIN3\n`, then UTF‑8 JSON metadata, then separator `\n\nGZIP FOLLOWS\n\n`, then raw or gzip‑compressed spectral arrays.
- Metadata includes full row objects plus phase definitions.
- Spectral arrays are emitted in row order for rows that include spectral data.

**Row Fields (v2/v3)**
- `yaw_deg`, `roll_deg`, `lin_mm`: Goniometer coordinates and distance.
- `timestamp`: Epoch seconds.
- `integral_result`: Irradiance (W/m^2).
- `spectral_result`: Spectral irradiance per wavelength (W/m^2/nm).
- `flags`: Bitwise flags in `Flags` enum (e.g., ignore, no spectral).

**Units and Conventions**
- Integral irradiance is stored in W/m^2.
- Spectral irradiance is stored in W/m^2/nm.
- Wavelengths are in nm.
- The GUI reports intensity in µW/cm^2 (1 W/m^2 = 100 µW/cm^2).

**Power CSV**
- Must contain a timestamp column and a power column.
- Defaults for detection are configured in the app (Settings → Power Columns…).
- The app renames detected columns to `Timestamp` and `W_Active` internally.

**Session JSON**
- Saved via the GUI and captures files, groups, associations, trims, and control values.
- Stored as a single JSON file; see `on_save_session` in `gui.py` for field layout.
