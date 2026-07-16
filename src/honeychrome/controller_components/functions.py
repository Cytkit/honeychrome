import numpy as np
import struct
from flowio.exceptions import FCSParsingError
from flowkit import Sample, QuadrantDivider, Dimension, gates
from PySide6.QtCore import QSettings
from queue import Empty
from time import perf_counter
from functools import wraps
from pathlib import Path
from honeychrome.controller_components.transform import Transform
from honeychrome.settings import linear_a, logicle_w, logicle_m, logicle_a, log_m

q_settings = QSettings("honeychrome", "ExperimentSelector")

import logging
logger = logging.getLogger(__name__)

def _parse_fcs_keywords(txt, delim):
    """Split FCS TEXT segment on delim, return keyword dict."""
    kv = txt.split(delim)
    if kv and kv[0] == '':
        kv = kv[1:]
    if len(kv) % 2 != 0:
        kv.append('')
    return dict(zip(kv[0::2], kv[1::2]))


def _detect_fcs_inner_delim(txt):
    """Return the most-frequent punctuation char in the first 500 chars of txt."""
    probe = txt[:500]
    counts = {}
    for ch in probe:
        if not (ch.isalnum() or ch in ' $.,;:_\\-+()[]'):
            counts[ch] = counts.get(ch, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _load_fcs_with_repaired_delimiter(path):
    """
    Last-resort loader for FCS files whose TEXT delimiter byte is wrong
    (e.g. FACSDiscover files that write space 0x20 but use pipe | internally).
    Reads the TEXT segment, detects the true delimiter, parses keywords,
    then reads the data segment manually and wraps it as an fk.Sample.
    Returns an fk.Sample or raises ValueError if repair fails.
    """
    path = str(path)
    with open(path, 'rb') as f:
        header = f.read(58).decode('latin-1')
        txt_st = int(header[10:18].strip())
        txt_en = int(header[18:26].strip())

        f.seek(txt_st)
        raw = f.read(txt_en - txt_st + 1)
        txt = raw.replace(b'\x00', b'').decode('latin-1')

        stated_delim = txt[0]
        keywords = _parse_fcs_keywords(txt, stated_delim)

        # If critical keys missing, try detected inner delimiter
        if '$TOT' not in keywords or '$PAR' not in keywords:
            inner = _detect_fcs_inner_delim(txt)
            if inner and inner != stated_delim:
                keywords = _parse_fcs_keywords(txt, inner)

        if '$TOT' not in keywords or '$PAR' not in keywords:
            raise ValueError(f'Cannot repair FCS TEXT segment in {path}: '
                             f'$TOT/$PAR missing after delimiter detection')

        total_events = int(keywords['$TOT'])
        n_par        = int(keywords['$PAR'])
        data_st      = int(keywords['$BEGINDATA'])
        byteord      = keywords.get('$BYTEORD', '1,2,3,4')
        endian       = '<' if byteord == '1,2,3,4' else '>'

        f.seek(data_st)
        n_vals = total_events * n_par
        raw_data = f.read(n_vals * 4)

    fmt = f'{endian}{n_vals}f'
    flat = struct.unpack(fmt, raw_data)
    # FCS data is row-major (event, channel)
    arr = np.array(flat, dtype=np.float64).reshape(total_events, n_par)

    col_names = []
    for i in range(1, n_par + 1):
        col_names.append(keywords.get(f'$P{i}N', f'Channel_{i}'))

    sample = Sample(arr, channel_labels=col_names, sample_id=str(Path(path).name))
    # Attach keywords so get_metadata() callers see them
    sample._flowdata_object.text.update(keywords)
    return sample


def sample_from_fcs(path, bus=None):
    try:
        if bus:
            bus.statusMessage.emit(f'Loading sample {path}...')
        sample = Sample(path)

    except KeyError as e:
        logging.warning(
            f'Controller: FlowIO reports FCS file does not conform to standards. '
            f'Missing {e}. Attempting to load with use_header_offsets and ignore_offset_error set')
        try:
            sample = Sample(path, use_header_offsets=True, ignore_offset_error=True)
        except Exception:
            logging.warning(f'Controller: use_header_offsets fallback also failed for {path}. '
                            f'Attempting delimiter repair.')
            sample = _load_fcs_with_repaired_delimiter(path)

    except FCSParsingError as e:
        logging.warning(
            f'Controller: FlowIO reports a data offset that is off by 1. '
            f'{e}. Attempting to load with ignore_offset_error=True')
        try:
            sample = Sample(path, use_header_offsets=True, ignore_offset_error=True)
        except Exception:
            logging.warning(f'Controller: use_header_offsets fallback also failed for {path}. '
                            f'Attempting delimiter repair.')
            sample = _load_fcs_with_repaired_delimiter(path)

    except Exception as e:
        logging.warning(f'Controller: Unexpected error loading {path}: {e}. '
                        f'Attempting delimiter repair.')
        sample = _load_fcs_with_repaired_delimiter(path)

    # Replace literal 'NA' keyword values (some FACSDiscover files write these).
    meta = sample.get_metadata()
    if meta:
        na_keys = [k for k, v in meta.items() if str(v).strip().upper() == 'NA']
        for k in na_keys:
            meta[k] = ''
        if na_keys:
            logger.debug('sample_from_fcs: NA keywords zeroed in %s: %s', path, na_keys)

    return sample

def define_process_plots(fluorescence_channels_x, fluorescence_channels_y, source_gate):
    process_plots = [{'type': 'hist2d', 'channel_x': x, 'channel_y': y, 'source_gate': source_gate, 'child_gates': []} if x != y
                     else {'type': 'hist1d', 'channel_x': x, 'source_gate': source_gate, 'child_gates': []}
                     for x in fluorescence_channels_x for y in fluorescence_channels_y]
    return process_plots

def all_same(lst):
    if not lst:  # Handle empty list
        return True  # or False, depending on your needs
    return all(x == lst[0] for x in lst)

def empty_queue_nowait(q):
    """Empty queue using get_nowait()"""
    items_removed = 0
    while True:
        try:
            q.get_nowait()
            items_removed += 1
        except Empty:
            break
    return items_removed

def add_recent_file(path):
    path = str(path)
    recent = q_settings.value("recent_files", [])
    if isinstance(recent, str):
        recent = [recent]
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    q_settings.setValue("recent_files", recent)  # store full history

def export_unmixed_sample(
    sample_name: str,
    unmixed_folder: 'Path | str',
    export_event_data: np.ndarray,
    export_pnn: list,
    spillover: np.ndarray,
    raw_keywords: 'dict[str, str]',
    spectral_model: list,
    unmixed_settings: dict,
    raw_settings: dict,
    af_spectra: 'np.ndarray | None',
    unmixing_spectra: 'np.ndarray | None',
    version: str,
    subsample: 'int | None' = None,
    extra_null_channels: 'list | None' = None,
    unmixing_method: str = 'OLS',
    unmixing_weights: 'np.ndarray | None' = None,
    extra_whitelist: 'frozenset | set' = frozenset(),
) -> None:
    """
    Write an unmixed FCS file with full FCS 3.1 compliant metadata.

    Parameters
    ----------
    sample_name       : base name without extension (used for $FIL and filename).
    unmixed_folder    : destination directory (must exist).
    export_event_data : (n_events, n_channels) float array in output column order.
    export_pnn        : ordered list of output channel names, matching export_event_data columns.
    spillover         : (n_fluor x n_fluor) fine-tuning spillover; written to $SPILLOVER.
    raw_keywords      : TEXT keywords from the source FCS (sample.get_metadata()['text']).
    spectral_model    : controller.experiment.process['spectral_model'].
    unmixed_settings  : controller.experiment.settings['unmixed'].
    raw_settings      : controller.experiment.settings['raw'].
    af_spectra        : stacked AF spectra (n_profiles x n_detectors) or None.
    unmixing_spectra  : fluorophore spectra matrix (n_fluor x n_raw_detectors) or None.
                        Pass controller._build_fluor_spectra() output.
    version           : honeychrome.__version__.
    subsample         : if set, randomly subsample to this many events before writing.
    extra_null_channels: channel names (e.g. 'AF Index') whose values should be zeroed.
    unmixing_method   : written to UNMIXINGMETHOD keyword.
    extra_whitelist   : additional channel names to carry through from the raw file
                        verbatim. Pass imaging_carry_through_set for FACSDiscover.
    """
    # Optional subsampling
    if subsample is not None and subsample < export_event_data.shape[0]:
        idx = np.sort(np.random.choice(export_event_data.shape[0], subsample, replace=False))
        export_event_data = export_event_data[idx]

    # Zero out null channels in a copy (event_id, AF Index when not meaningful, etc.)
    null_set = {'event_id'} | set(extra_null_channels or [])
    if null_set & set(export_pnn):
        export_event_data = export_event_data.copy()
        for col_name in null_set:
            if col_name in export_pnn:
                export_event_data[:, export_pnn.index(col_name)] = 0.0

    file_name = sample_name + ' (Unmixed).fcs'
    file_path = Path(unmixed_folder) / file_name

    keywords = define_fcs_keywords(
        raw_keywords=raw_keywords,
        pnn=export_pnn,
        event_data=export_event_data,
        spectral_model=spectral_model,
        unmixed_settings=unmixed_settings,
        raw_settings=raw_settings,
        spillover=spillover,
        af_spectra=af_spectra,
        unmixing_spectra=unmixing_spectra,
        file_name=file_name,
        version=version,
        unmixing_method=unmixing_method,
        unmixing_weights=unmixing_weights,
        extra_whitelist=extra_whitelist,
    )

    write_fcs(export_event_data, keywords, file_path)

def define_fcs_keywords(
    raw_keywords: 'dict[str, str]',
    pnn: list,
    event_data: np.ndarray,
    spectral_model: list,
    unmixed_settings: dict,
    raw_settings: dict,
    spillover: np.ndarray,
    af_spectra: 'np.ndarray | None',
    unmixing_spectra: 'np.ndarray | None',
    file_name: str,
    version: str,
    unmixing_method: str = 'OLS',
    unmixing_weights: 'np.ndarray | None' = None,
    extra_whitelist: 'frozenset | set' = frozenset(),
) -> dict:
    """
    Build a complete FCS 3.1 TEXT keyword dict for an unmixed export file.
    Mirrors AutoSpectral define.keywords().

    Parameters
    ----------
    raw_keywords      : dict returned by sample.get_metadata()['text'] on the raw FCS.
    pnn               : ordered list of output channel names (the export column order).
    event_data        : (n_events, n_channels) array — used only for $TOT.
    spectral_model    : controller.experiment.process['spectral_model'] list of dicts.
    unmixed_settings  : controller.experiment.settings['unmixed'].
    raw_settings      : controller.experiment.settings['raw'].
    spillover         : (n_fluor x n_fluor) fine-tuning spillover matrix.
    af_spectra        : (n_profiles x n_detectors) AF spectra, or None.
    unmixing_spectra  : (n_fluor x n_detectors) fluorophore spectra used for unmixing,
                        or None. Pass controller._build_fluor_spectra() output.
    file_name         : output filename string (written to $FIL).
    version           : honeychrome.__version__ string.
    unmixing_method   : written to custom keyword UNMIXINGMETHOD.
    extra_whitelist   : additional channel names to carry through from the raw file
                        verbatim (same treatment as scatter/time). Used for FACSDiscover
                        imaging channels: pass imaging_carry_through_set from the caller.
    """
    import re
    from datetime import datetime, timezone

    # Keywords Honeychrome recomputes fresh elsewhere in this function (or in
    # write_fcs). Raw values for these must never be carried through, or they
    # end up as duplicate keywords alongside the correct ones — since raw
    # keyword keys now come back lowercased/$-stripped from get_metadata(),
    # they no longer collide by exact key string with the ones we write below.
    _RECOMPUTED_KEYWORDS = {
        'FIL', 'PAR', 'TOT', 'DATATYPE', 'BYTEORD', 'MODE', 'NEXTDATA',
        'BEGINANALYSIS', 'ENDANALYSIS', 'BEGINSTEXT', 'ENDSTEXT',
        'BEGINDATA', 'ENDDATA', 'ORIGINALITY', 'LAST_MODIFIED', 'LAST_MODIFIER',
        'HONEYCHROME', 'UNMIXINGMETHOD', 'SPILLOVER',
        'SPECTRA', 'FLUOROCHROMES', 'AUTOFLUORESCENCE', 'WEIGHTS',
        'BDCHORUSDATARECORD',
    }

    # ---- 1. Carry-through non-parameter keywords from raw file ----
    # Guard against malformed keyword dicts produced by FlowIO when the FCS
    # TEXT delimiter byte is wrong (FACSDiscover export bug): corrupt entries
    # have pipe-delimited blobs as keys. Skip any key that contains '|', is
    # empty, or is implausibly long (>64 chars) for a real FCS keyword.
    non_param_keys = {
        k: v for k, v in raw_keywords.items()
        if k
        and '|' not in k
        and len(k) <= 64
        and not re.match(r'^\$?P\d+', k, re.IGNORECASE)
        and not re.match(r'^\$?CH\d+', k, re.IGNORECASE)
        and k.lstrip('$').upper() not in _RECOMPUTED_KEYWORDS
    }

    # ---- 2. Whitelist for raw param carry-through ----
    sc_ids = set(raw_settings.get('scatter_channel_ids', []))
    time_id = raw_settings.get('time_channel_id')
    raw_pnn = raw_settings.get('event_channels_pnn', [])

    whitelist = set()
    if time_id is not None and time_id < len(raw_pnn):
        whitelist.add(raw_pnn[time_id])
    for i in sc_ids:
        if i < len(raw_pnn):
            whitelist.add(raw_pnn[i])
    whitelist |= set(extra_whitelist)   # imaging channels (or empty on non-FACSDiscover)

    raw_param_lookup = _build_raw_param_lookup(raw_keywords, whitelist)

    # ---- 3. Antigen lookup from spectral model ----
    label_to_antigen = {
        c['label']: (c.get('antigen') or '')
        for c in (spectral_model or [])
    }

    # ---- 4. Channel-type index sets for the *output* pnn ----
    fl_ids_out = set(unmixed_settings.get('fluorescence_channel_ids', []))
    magnitude_ceiling = unmixed_settings.get('magnitude_ceiling', 262144)

    BIT_DEPTH = '32'
    param_keywords = {}

    for i, ch in enumerate(pnn, start=1):
        prefix = f'$P{i}'
        ch_safe = ch.replace(',', '_')   # FCS 3.1 §3.2.23

        if ch == 'AF Abundance':
            param_keywords.update({
                f'{prefix}N': 'AF Abundance',
                f'{prefix}S': 'Autofluorescence Abundance',
                f'{prefix}B': BIT_DEPTH,
                f'{prefix}E': '0,0',
                f'{prefix}R': str(magnitude_ceiling),
                f'{prefix}G': '1',
                f'{prefix}DISPLAY': 'LOG',
            })

        elif ch == 'AF Index':
            n_profiles = af_spectra.shape[0] if af_spectra is not None else 1
            param_keywords.update({
                f'{prefix}N': 'AF Index',
                f'{prefix}S': 'Autofluorescence Index',
                f'{prefix}B': BIT_DEPTH,
                f'{prefix}E': '0,0',
                f'{prefix}R': str(n_profiles),
                f'{prefix}G': '1',
                f'{prefix}DISPLAY': 'LIN',
            })

        elif ch in raw_param_lookup:
            # Scatter / Time: carry selected fields from raw file
            old = raw_param_lookup[ch]
            for field in ('N', 'S', 'B', 'E', 'R', 'G', 'V', 'DISPLAY', 'TYPE'):
                old_key = next(
                    (k for k in old if re.sub(r'^\$?P\d+', '', k, flags=re.IGNORECASE).upper() == field),
                    None
                )
                if old_key:
                    param_keywords[f'{prefix}{field}'] = old[old_key]
            # Guarantee mandatory fields
            param_keywords.setdefault(f'{prefix}N', ch_safe)
            param_keywords.setdefault(f'{prefix}E', '0,0')
            param_keywords.setdefault(f'{prefix}R', str(magnitude_ceiling))
            param_keywords.setdefault(f'{prefix}DISPLAY', 'LIN')
            param_keywords.setdefault(f'{prefix}B', BIT_DEPTH)

        else:
            # Unmixed fluorophore (or imaging channel not in raw_param_lookup)
            antigen = label_to_antigen.get(ch) or label_to_antigen.get(ch.removesuffix('-A'), '')
            is_fl = (i - 1) in fl_ids_out
            param_keywords.update({
                f'{prefix}N': ch_safe,
                f'{prefix}S': antigen if antigen else ch_safe,
                f'{prefix}B': BIT_DEPTH,
                f'{prefix}E': '0,0',
                f'{prefix}R': str(magnitude_ceiling),
                f'{prefix}G': '1',
                f'{prefix}DISPLAY': 'LOG' if is_fl else 'LIN',
            })

    # ---- 5. Spillover keyword ----
    fl_pnn_out = [pnn[i] for i in sorted(fl_ids_out) if i < len(pnn)]
    spillover_str = _format_spillover(fl_pnn_out, spillover)

    # ---- 6. Spectra matrix keywords ----
    def _fmt_matrix(m, row_names, col_names):
        vals = ','.join(f'{v:.8g}' for v in m.flatten())
        return f'{m.shape[0]},{m.shape[1]},{",".join(row_names)},{",".join(col_names)},{vals}'

    spectra_kw = {}
    # Use the channels actually in the unmixing matrix, not the raw unfiltered
    # list — keeps SPECTRA/AUTOFLUORESCENCE/WEIGHTS shape checks correct
    # regardless of fluorescence_channel_filter.
    det_names = unmixed_settings.get('fluorescence_channels') or [
        raw_pnn[i] for i in raw_settings.get('fluorescence_channel_ids', [])
    ]

    if unmixing_spectra is not None and unmixing_spectra.ndim == 2:
        fluor_names_short = [pnn[i].removesuffix('-A') for i in sorted(fl_ids_out) if i < len(pnn)]
        if unmixing_spectra.shape == (len(fluor_names_short), len(det_names)):
            spectra_kw['SPECTRA'] = _fmt_matrix(unmixing_spectra, fluor_names_short, det_names)
            spectra_kw['FLUOROCHROMES'] = ','.join(fluor_names_short)

    if af_spectra is not None and af_spectra.ndim == 2:
        af_row_names = [f'AF{i + 1}' for i in range(af_spectra.shape[0])]
        if af_spectra.shape[1] == len(det_names):
            spectra_kw['AUTOFLUORESCENCE'] = _fmt_matrix(af_spectra, af_row_names, det_names)

    if unmixing_weights is not None and unmixing_weights.ndim == 1:
        if len(unmixing_weights) == len(det_names):
            vals = ','.join(f'{v:.8g}' for v in unmixing_weights)
            spectra_kw['WEIGHTS'] = f'{len(det_names)},{",".join(det_names)},{vals}'

    # ---- 7. Provenance keywords ----
    now_str = datetime.now(timezone.utc).strftime('%d-%b-%Y %H:%M:%S').upper()
    provenance = {
        '$FIL':           file_name,
        '$PAR':           str(len(pnn)),
        '$TOT':           str(event_data.shape[0]),
        '$DATATYPE':      'F',
        '$BYTEORD':       '1,2,3,4',
        '$MODE':          'L',
        '$NEXTDATA':      '0',
        '$BEGINANALYSIS': '0',
        '$ENDANALYSIS':   '0',
        '$BEGINSTEXT':    '0',
        '$ENDSTEXT':      '0',
        '$ORIGINALITY':   'DataModified',
        '$LAST_MODIFIED': now_str,
        '$LAST_MODIFIER': f'Honeychrome_{version}',
        'HONEYCHROME':    version,
        'UNMIXINGMETHOD': unmixing_method,
    }
    if spillover_str:
        provenance['$SPILLOVER'] = spillover_str

    # ---- 8. Merge: raw non-param ← param ← provenance ← spectra ----
    keywords = {**non_param_keys, **param_keywords, **provenance, **spectra_kw}
    return keywords


def _build_raw_param_lookup(raw_keywords: dict, whitelist: set[str]) -> dict:
    """
    Extract per-parameter keywords from raw FCS TEXT for whitelisted channels.
    Returns {channel_name: {original_keyword: value, ...}}.
    """
    import re
    pN_keys = [k for k in raw_keywords if re.match(r'^\$?P\d+N$', k, re.IGNORECASE)]
    lookup = {}
    for key in pN_keys:
        ch_name = raw_keywords[key]
        if ch_name not in whitelist:
            continue
        idx = re.sub(r'^\$?P(\d+)N$', r'\1', key, flags=re.IGNORECASE)
        matches = {k: raw_keywords[k]
                   for k in raw_keywords
                   if re.match(rf'^\$?P{idx}[A-Z]+$', k, re.IGNORECASE)}
        lookup[ch_name] = matches
    return lookup


def _format_spillover(fl_pnn: list[str], spillover: np.ndarray) -> str:
    """
    Serialise the fine-tuning spillover matrix as an FCS 3.1 $SPILLOVER string.
    fl_pnn: ordered list of fluorophore $PnN names (must match spillover shape).
    spillover: square (n_fluor x n_fluor) matrix.
    Written row-major, untransposed: spillover[i][j] (row i spills into column j)
    matches FlowJo's own row/column convention directly now that Changes 1 and 2
    apply the same convention internally without an extra transpose.
    """
    n = len(fl_pnn)
    if spillover is None or spillover.shape != (n, n):
        return ''
    vals = ','.join(f'{v:.8g}' for v in spillover.flatten())
    names = ','.join(fl_pnn)
    return f'{n},{names},{vals}'

def write_fcs(
    event_data: np.ndarray,  # (n_events, n_channels), float32 written row-major
    keywords: dict[str, str],
    file_path: Path | str,
    chunk_rows: int = 131_072,
) -> None:
    """
    Write a minimal FCS 3.1 file (HEADER + TEXT + DATA).
    Data written as little-endian float32, row-major (one event per row).
    Mirrors writeFCS.R from AutoSpectral.
    """
    import struct

    DELIM = '|'
    file_path = Path(file_path)

    # Mandatory field overrides (ensure consistency)
    kw = dict(keywords)
    kw['$TOT']           = str(event_data.shape[0])
    kw['$PAR']           = str(event_data.shape[1])
    kw['$DATATYPE']      = 'F'
    kw['$BYTEORD']       = '1,2,3,4'
    kw['$NEXTDATA']      = '0'
    kw['$MODE']          = 'L'
    kw['$BEGINDATA']     = '0'   # placeholder; updated below
    kw['$ENDDATA']       = '0'
    kw['$BEGINSTEXT']    = '0'
    kw['$ENDSTEXT']      = '0'
    kw['$BEGINANALYSIS'] = '0'
    kw['$ENDANALYSIS']   = '0'

    TEXT_START = 58  # HEADER is always 58 bytes

    def _build_text(kw_dict):
        return DELIM + ''.join(
            f'{k}{DELIM}{v}{DELIM}' for k, v in kw_dict.items()
        )

    text = _build_text(kw)
    text_bytes = text.encode('utf-8')
    TEXT_END = TEXT_START + len(text_bytes) - 1
    data_bytes = event_data.shape[0] * event_data.shape[1] * 4  # float32

    # Iterative layout: grow TEXT_END until $BEGINDATA/$ENDDATA digit-lengths stabilise.
    # Seed with the actual encoded lengths of the placeholder values already in kw.
    kw_len_old = len(kw['$BEGINDATA']) + len(kw['$ENDDATA'])
    while True:
        DATA_START = TEXT_END + 1
        DATA_END   = DATA_START + data_bytes - 1
        kw_len_new = len(str(DATA_START)) + len(str(DATA_END))
        if kw_len_new > kw_len_old:
            TEXT_END  += kw_len_new - kw_len_old
            kw_len_old = kw_len_new
        else:
            break

    # Patch offsets and iterate until the encoded text length stops changing.
    # Each rebuild may change $BEGINDATA/$ENDDATA digit counts, which shifts
    # DATA_START and therefore DATA_END, which may change digit counts again.
    for _ in range(8):  # converges in ≤3 iterations in practice
        kw['$BEGINDATA'] = str(DATA_START)
        kw['$ENDDATA']   = str(DATA_END)
        text = _build_text(kw)
        text_bytes = text.encode('latin-1')
        new_TEXT_END = TEXT_START + len(text_bytes) - 1
        new_DATA_START = new_TEXT_END + 1
        new_DATA_END   = new_DATA_START + data_bytes - 1
        if new_DATA_START == DATA_START and new_DATA_END == DATA_END:
            TEXT_END = new_TEXT_END
            break
        TEXT_END = new_TEXT_END
        DATA_START = new_DATA_START
        DATA_END   = new_DATA_END
    else:
        raise RuntimeError(f'write_fcs: offset layout did not converge for {file_path}')

    # FCS 3.1 §3.1: header offset fields are 8 chars each.
    # If DATA_START or DATA_END exceed 8 digits, write 0 in the header —
    # compliant readers must use $BEGINDATA/$ENDDATA from TEXT in that case.
    def _h(n): return f'{n:>8d}' if n < 100_000_000 else '       0'
    header = (
        f'{"FCS3.1":<10}'
        + _h(TEXT_START) + _h(TEXT_END)
        + _h(DATA_START)  + _h(DATA_END)
        + '       0'      + '       0'   # STEXT always 0
    )

    with open(file_path, 'wb') as fh:
        fh.write(header.encode('ascii'))
        fh.write(text_bytes)
        # Write event data in chunks (row-major, float32 little-endian)
        rows_remaining = event_data.shape[0]
        offset = 0
        while rows_remaining > 0:
            rows = min(chunk_rows, rows_remaining)
            chunk = event_data[offset:offset + rows, :]
            fh.write(chunk.astype('<f4').tobytes())   # little-endian float32
            offset         += rows
            rows_remaining -= rows
        fh.write(b'00000000')  # CRC placeholder


# All subfolders recursively
def get_all_subfolders_recursive(path, experiment_dir):
    """Get all subfolders recursively using pathlib"""
    p = Path(path)
    return [p.relative_to(experiment_dir)] + [folder.relative_to(experiment_dir) for folder in p.rglob('*') if folder.is_dir()]

def timer(func):
    """Decorator to report execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' executed in {execution_time:0.6f} seconds")
        return result
    return wrapper

def assign_default_transforms(settings, channels=None):
    if channels is None:
        channels = settings['event_channels_pnn']
    transforms = {}

    ceiling = settings['magnitude_ceiling']
    transforms['ribbon'] = {'scale_t': ceiling, 'linear_a': linear_a, 'logicle_w': logicle_w, 'logicle_m': logicle_m,
                            'logicle_a': logicle_a, 'log_m': log_m, 'id': 1, 'limits': [0, 1]}

    for label in channels:
        index = settings['event_channels_pnn'].index(label)
        channel_pnr = settings.get('channel_pnr')
        if index in settings['scatter_channel_ids']:
            if label[-2:] == '-W' or label in settings['width_channels']:
                ceiling = settings['width_ceiling']
            elif channel_pnr:
                ceiling = float(channel_pnr[index])
            else:
                ceiling = settings['magnitude_ceiling']
        elif index in settings['fluorescence_channel_ids']:
            ceiling = settings['magnitude_ceiling']
        else:
            ceiling = settings['default_ceiling']
            if label[-2:] == '-W' or label in settings['width_channels']:
                ceiling = settings['width_ceiling']

        if index in settings['scatter_channel_ids']:
            id = 0  # linear
            display_ceiling_overrides = settings.get('scatter_display_ceiling', {})
            if label in display_ceiling_overrides and ceiling > 0:
                # Clamp the initial viewport to the override value in display space,
                # leaving scale_t (and therefore the transform) covering the full PNR.
                # The user can still zoom out beyond this limit.
                display_limit = min(display_ceiling_overrides[label] / ceiling, 1.0)
            else:
                display_limit = 1.0
            limits = [0, display_limit]
        elif index in settings['fluorescence_channel_ids']:
            id = 1  # logicle
            limits = [0, 1]
        else:
            id = 'default'
            limits = [0, 100]

        transforms[label] = {'scale_t': ceiling, 'linear_a': linear_a, 'logicle_w': logicle_w, 'logicle_m': logicle_m,
                             'logicle_a': logicle_a, 'log_m': log_m, 'id': id, 'limits': limits}

    return transforms


def update_transforms(transforms, transformations):
    for label in transformations:
        transformation = transformations[label]
        linear_a = transformation.linear_a
        logicle_w = transformation.logicle_w
        logicle_m = transformation.logicle_m
        logicle_a = transformation.logicle_a
        scale_t = transformation.scale_t
        id = transformation.id
        limits = transformation.limits
        if scale_t is not None:
            transforms[label]['scale_t'] = scale_t
        if linear_a is not None:
            transforms[label]['linear_a'] = linear_a
        if logicle_w is not None:
            transforms[label]['logicle_w'] = logicle_w
        if logicle_m is not None:
            transforms[label]['logicle_m'] = logicle_m
        if logicle_a is not None:
            transforms[label]['logicle_a'] = logicle_a
        if log_m is not None:
            transforms[label]['log_m'] = log_m
        if id is not None:
            transforms[label]['id'] = id
        if limits is not None:
            transforms[label]['limits'] = limits


def generate_transformations(transforms):
    transformations = {}
    for label in transforms:
        transform = transforms[label]
        transformations[label] = Transform(scale_t=transform['scale_t'], linear_a=transform['linear_a'],
                                           logicle_w=transform['logicle_w'], logicle_m=transform['logicle_m'],
                                           logicle_a=transform['logicle_a'], log_m=transform['log_m'])
        transformations[label].set_transform(id=transform['id'], limits=transform['limits'])

    return transformations


def apply_transfer_matrix(transfer_matrix, raw_event_data):
    '''
    called when:
        sample loaded
        fine-tuning matrix changes
        live data updated
        calculate stats
        export unmixed FCS
    '''
    return raw_event_data @ transfer_matrix


def define_quad_gates(x, y, channel_x, channel_y, transformations):
    # QuadrantDivider instances are similar to a Dimension, they take compensation_ref and tranformation_ref
    transformation_ref_x = channel_x if transformations[channel_x].xform else None
    transformation_ref_y = channel_y if transformations[channel_y].xform else None
    quad_div_x = QuadrantDivider('xdiv', channel_x, compensation_ref='uncompensated', transformation_ref=transformation_ref_x, values=[x])
    quad_div_y = QuadrantDivider('ydiv', channel_y, compensation_ref='uncompensated', transformation_ref=transformation_ref_y, values=[y])

    quad_divs = [quad_div_x, quad_div_y]

    # the 2 dividers above will be used to divide the space into 4 quadrants
    quad_pp = gates.Quadrant(quadrant_id=f'{channel_x}+ {channel_y}+', divider_refs=['xdiv', 'ydiv'],
        divider_ranges=[(x, None), (y, None)])
    quad_pn = gates.Quadrant(quadrant_id=f'{channel_x}+ {channel_y}-', divider_refs=['xdiv', 'ydiv'],
        divider_ranges=[(x, None), (None, y)])
    quad_np = gates.Quadrant(quadrant_id=f'{channel_x}- {channel_y}+', divider_refs=['xdiv', 'ydiv'],
        divider_ranges=[(None, x), (y, None)])
    quad_nn = gates.Quadrant(quadrant_id=f'{channel_x}- {channel_y}-', divider_refs=['xdiv', 'ydiv'],
        divider_ranges=[(None, x), (None, y)])
    quadrants = [quad_pp, quad_pn, quad_np, quad_nn]

    return quad_divs, quadrants

def define_range_gate(x1, x2, channel_x, transformations):
    transformation_ref = channel_x if transformations[channel_x].xform else None
    dim_x = Dimension(channel_x, compensation_ref='uncompensated', transformation_ref=transformation_ref, range_min=x1,
                         range_max=x2)
    return dim_x

def define_polygon_gate(points, channel_x, channel_y, transformations):
    # print(points)
    transformation_ref_x = channel_x if transformations[channel_x].xform else None
    transformation_ref_y = channel_y if transformations[channel_y].xform else None
    dim_x = Dimension(channel_x, compensation_ref='uncompensated', transformation_ref=transformation_ref_x, range_min=0, range_max=1)
    dim_y = Dimension(channel_y, compensation_ref='uncompensated', transformation_ref=transformation_ref_y, range_min=0, range_max=1)
    return points, dim_x, dim_y

def define_rectangle_gate(pos, size, channel_x, channel_y, transformations):
    x0, y0 = pos
    Dx, Dy = size
    # x0, y0 = np.array(pos)
    # Dx, Dy = np.array(size) / 2

    transformation_ref_x = channel_x if transformations[channel_x].xform else None
    transformation_ref_y = channel_y if transformations[channel_y].xform else None
    dim_x = Dimension(channel_x, compensation_ref='uncompensated', transformation_ref=transformation_ref_x, range_min=x0,
                         range_max=x0 + Dx)
    dim_y = Dimension(channel_y, compensation_ref='uncompensated', transformation_ref=transformation_ref_y, range_min=y0,
                         range_max=y0 + Dy)

    return dim_x, dim_y

def define_ellipse_gate(pos, size, angle, channel_x, channel_y, transformations):
    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coordinates = np.array(pos) + 0.5 * R @ np.array(size)

    w, h = np.array(size)

    # Covariance matrix
    D = np.diag([w, h])
    covariance_matrix = R @ D @ R.T
    distance_square = w * h

    transformation_ref_x = channel_x if transformations[channel_x].xform else None
    transformation_ref_y = channel_y if transformations[channel_y].xform else None

    dim_x = Dimension(channel_x, compensation_ref='uncompensated', transformation_ref=transformation_ref_x, range_min=0,
                         range_max=1)
    dim_y = Dimension(channel_y, compensation_ref='uncompensated', transformation_ref=transformation_ref_y, range_min=0,
                         range_max=1)

    # print(pos, size, angle)
    # print(coordinates)
    # print(w, h)
    # print(covariance_matrix)
    # print(distance_square)
    # print(transformations[channel_x].limits)
    # print(transformations[channel_y].limits)

    return dim_x, dim_y, coordinates, covariance_matrix, distance_square


def apply_gates_in_place(data_for_cytometry_plots, gates_to_calculate=None):
    # calculate only gates in gates_to_calculate
    # gates_to_calculate should be in order of ancestry (parent to child)

    events = data_for_cytometry_plots['event_data']
    pnn = data_for_cytometry_plots['pnn']
    transforms = data_for_cytometry_plots['transformations']
    lookup_tables = data_for_cytometry_plots['lookup_tables']
    gating = data_for_cytometry_plots['gating']
    gate_membership = data_for_cytometry_plots['gate_membership']

    # loop through gates, produce gate_membership mask for each
    # ignore quadrants but loop through them in parent quadrantgate
    gate_ids = gating.get_gate_ids()
    for gate_id in gate_ids:
        if gate_id[0] in gates_to_calculate:
            if gating._get_gate_node(gate_id[0], gate_id[1]).gate_type != 'Quadrant': # bit of a hack. Can't find a better way of excluding Quadrant
                gate = gating.get_gate(gate_id[0])
                parent_id = gating.get_parent_gate_id(gate_id[0])
                if parent_id is None:
                    parent_id = ('root',)

                channels = gate.get_dimension_ids()
                if len(channels) == 1:
                    xchan = channels[0]
                    ix = pnn.index(xchan)
                    x = events[:, ix]
                    transform_x = transforms[xchan]
                    scale_x = transform_x.scale
                    indices_x_data_searchsorted = np.searchsorted(scale_x, x) - 2
                    if len(scale_x) > len(lookup_tables[gate_id[0]]):
                        # Clamp the top edge into the last table bin. `scale` carries
                        # two ±inf sentinels so it's normally exactly one longer than
                        # the table (harmless). For a *dynamic* (per-sample) Time gate
                        # the table is rebuilt on load at this sample's own scale, so
                        # only this one-bin edge is clamped; a *non-dynamic* Time gate
                        # keeps a shared table built at a different scale, so this also
                        # guards the larger mismatch (see apply_dynamic_gate_dimensions).
                        indices_x_data_searchsorted[indices_x_data_searchsorted >= len(lookup_tables[gate_id[0]])] = len(lookup_tables[gate_id[0]]) - 1
                    indices_data_digitized_flattened = indices_x_data_searchsorted

                else:#len(channels) == 2:
                    if gate.gate_type != 'QuadrantGate':  # 2 channels
                        xchan = channels[0]
                        ychan = channels[1]
                    else:  # quad gate
                        xchan = gate.dimensions[0].dimension_ref
                        ychan = gate.dimensions[1].dimension_ref
                    ix = pnn.index(xchan)
                    iy = pnn.index(ychan)
                    x = events[:, ix]
                    y = events[:, iy]
                    transform_x = transforms[xchan]
                    transform_y = transforms[ychan]
                    scale_x = transform_x.scale
                    scale_y = transform_y.scale
                    indices_x_data_searchsorted = np.searchsorted(scale_x, x) - 2
                    indices_y_data_searchsorted = np.searchsorted(scale_y, y) - 2
                    # The lookup table is a row-major flatten of shape
                    # (x_bins+1, y_bins+1): element (i, j) lives at i*(y_bins+1)+j.
                    # So the row stride is the NUMBER OF Y BINS, not x. This only
                    # matters when x and y have different bin counts (e.g. a
                    # Time x fluorescence gate — Time has far more bins); for equal
                    # bin counts (the usual case) it's the same either way.
                    row_stride = transform_y.scale_bins + 1
                    indices_data_digitized_flattened = indices_x_data_searchsorted * row_stride + indices_y_data_searchsorted

                if gate.gate_type == 'QuadrantGate':
                    quadrant_names = gate.quadrants.keys()
                    for name in quadrant_names:
                        if name not in lookup_tables: # guard: lookup table may not exist yet if called before calculate_lookup_tables
                            continue
                        table = lookup_tables[name]
                        idx = indices_data_digitized_flattened
                        # Safety clamp: a transient template/transform scale
                        # mismatch can produce out-of-range indices; clamp so we
                        # never IndexError (correct values follow on next recalc).
                        if idx.size and (idx.max() >= len(table) or idx.min() < 0):
                            idx = np.clip(idx, 0, len(table) - 1)
                        mask = table[idx]
                        gate_membership[name] = mask * gate_membership[parent_id[0]]
                else:
                    if gate_id[0] not in lookup_tables:
                        continue
                    table = lookup_tables[gate_id[0]]
                    idx = indices_data_digitized_flattened
                    if idx.size and (idx.max() >= len(table) or idx.min() < 0):
                        idx = np.clip(idx, 0, len(table) - 1)
                    mask = table[idx]
                    gate_membership[gate_id[0]] = mask * gate_membership[parent_id[0]]

    # return gate_membership

def initialise_hists(plots, data_for_cytometry_plots):
    # loop over plots, produce set of histograms, default stats
    fluoro_indices = data_for_cytometry_plots['fluoro_indices']
    transformations = data_for_cytometry_plots['transformations']
    hists = []
    if plots:
        for n, plot in enumerate(plots):
            if plot['type'] == 'hist1d':
                bins = transformations[plot['channel_x']].scale_bins
                histogram = np.zeros(bins+1)
            elif plot['type'] == 'hist2d':
                bins_x = transformations[plot['channel_x']].scale_bins
                bins_y = transformations[plot['channel_y']].scale_bins
                histogram = np.zeros((bins_x+1, bins_y+1))
            else:  # 'ribbon'
                bins = transformations['ribbon'].scale_bins
                histogram = np.zeros((bins+1, len(fluoro_indices)))
            hists.append(histogram)
    return hists

def robust_cv(data):
    if len(data)>0:
        return (np.quantile(data, 0.75) - np.quantile(data, 0.25)) / np.quantile(data, 0.5) / 2 * 100
    else:
        return np.nan

def initialise_stats(gating):
    statistics = {'root': {'n_events_gate': 0, 'p_gate_total': 1., 'p_gate_parent': 1., 'event_conc': np.nan}}
    if gating:
        for gate_id in gating.get_gate_ids():
            if gating._get_gate_node(gate_id[0], gate_id[1]).gate_type != 'QuadrantGate': # bit of a hack. Can't find a better way of excluding Quadrants
                statistics[gate_id[0]] = {'n_events_gate': 0, 'p_gate_total': 0, 'p_gate_parent': 0, 'event_conc': np.nan}
    return statistics


def calc_hists(data_for_cytometry_plots, indices_plots_to_calculate=None, status_message_signal=None, density_cutoff=None, dot_plot_by_gate=False):
    plots = data_for_cytometry_plots['plots']
    gate_membership = data_for_cytometry_plots['gate_membership']

    if indices_plots_to_calculate is not None:
        plots = [plots[n] for n in indices_plots_to_calculate]
    pnn = data_for_cytometry_plots['pnn']
    transformations = data_for_cytometry_plots['transformations']
    event_data = data_for_cytometry_plots['event_data']
    fluoro_indices = data_for_cytometry_plots['fluoro_indices']

    hists = []
    for n, plot in enumerate(plots):
        if status_message_signal:
            status_message_signal.emit(f'Calculating {n}/{len(plots)} histograms...')

        source_gate = plot['source_gate']
        mask = gate_membership.get(source_gate)
        if mask is None:
            logger.warning(f"calc_hists: gate '{source_gate}' not in gate_membership — skipping plot {n}] - is this due to _reinitialise_process_plots_worker from a background thread and gate_membership is only partially built when calc_hists is called concurrently?")
            continue

        if plot['type'] == 'hist1d':
            id_channel = pnn.index(plot['channel_x'])
            transform = transformations[plot['channel_x']]
            histogram = calc_hist1d(event_data, mask, id_channel, transform)
        elif plot['type'] == 'hist2d':
            id_channel_x = pnn.index(plot['channel_x'])
            id_channel_y = pnn.index(plot['channel_y'])
            transform_x = transformations[plot['channel_x']]
            transform_y = transformations[plot['channel_y']]
            if dot_plot_by_gate:
                gating = data_for_cytometry_plots['gating']
                gate_ids = [g for g in gating.get_gate_ids() if gating._get_gate_node(g[0], g[1]).gate_type != 'QuadrantGate']
                source_and_child_gates = [source_gate] + [g[0] for g in gate_ids if source_gate in g[1]]
                histogram = calc_dotplot2d(event_data, source_and_child_gates, gate_membership, id_channel_x, id_channel_y, transform_x, transform_y, density_cutoff)
            else:
                histogram = calc_hist2d(event_data, mask, id_channel_x, id_channel_y, transform_x, transform_y, density_cutoff)
        else: # 'ribbon'
            histogram = calc_ribbon_plot(event_data, mask, fluoro_indices, transformations['ribbon'], density_cutoff)

        # add to existing array
        hists.append(histogram)
    return hists

def calc_stats(data_for_cytometry_plots, initialise=True):
    statistics = {}
    # gate_ids = data_for_cytometry_plots['lookup_tables'].keys()
    pnn = data_for_cytometry_plots['pnn']
    gating = data_for_cytometry_plots['gating']
    gate_membership = data_for_cytometry_plots['gate_membership']
    event_data = data_for_cytometry_plots['event_data']

    if event_data is not None:
        if initialise:
            statistics_old = initialise_stats(gating)
        else:
            statistics_old = data_for_cytometry_plots['statistics']

        ###### first version does total stats
        # n_events_total = len(data_for_cytometry_plots['event_data'])
        # statistics['root'] = {'n_events_gate': n_events_total, 'p_gate_total': 1., 'p_gate_parent': 1.}
        # for gate_id in gate_ids:
        #     n_events_gate = gate_membership[gate_id[0]].sum()
        #     p_gate_total = n_events_gate / n_events_total
        #     if gate_ids[0][1]==('root',):
        #         p_gate_parent = p_gate_total
        #     else:
        #         n_events_parent = gate_membership[gate_id[0][1][-1]].sum()
        #         p_gate_parent = n_events_gate / n_events_parent
        #     statistics[gate_id[0]] = {'n_events_gate':n_events_gate, 'p_gate_total':p_gate_total, 'p_gate_parent':p_gate_parent}

        ###### second version adds to previous stats
        n_events_total_old = statistics_old['root']['n_events_gate']
        n_events_total_new = len(event_data)
        n_events_total = n_events_total_old + n_events_total_new
        statistics['root'] = {'n_events_gate': n_events_total, 'p_gate_total': 1., 'p_gate_parent': 1., 'event_conc': np.nan}
        for gate_id_full in gating.get_gate_ids():
            gate_node = gating._get_gate_node(gate_id_full[0], gate_id_full[1])
            if gate_node.gate_type != 'QuadrantGate':  # bit of a hack. Can't find a better way of excluding Quadrants
                gate_id = gate_id_full[0]
                gate_path = gate_id_full[1]
                # parent_id = gate_id[0][1][-1]
                parent_id = data_for_cytometry_plots['gating'].get_parent_gate_id(gate_id, gate_path=gate_path)
                if parent_id is None:
                    parent_id = 'root'
                else:
                    parent_id = gating.get_parent_gate_id(gate_id, gate_path=gate_path)
                    if gating._get_gate_node(parent_id[0], parent_id[1]).gate_type != 'QuadrantGate':
                        parent_id = parent_id[0]
                    else:
                        parent_id = parent_id[1][-1]

                n_events_gate_old = statistics_old[gate_id]['n_events_gate']
                if gate_id not in gate_membership:
                    logger.warning(f'calc_stats: gate "{gate_id}" not in gate_membership — skipping.')
                    statistics[gate_id] = {'n_events_gate': 0, 'p_gate_total': 0, 'p_gate_parent': 0, 'event_conc': np.nan}
                    continue
                n_events_gate_new = gate_membership[gate_id].sum()
                n_events_gate = int(n_events_gate_old + n_events_gate_new)
                p_gate_total = n_events_gate / n_events_total if n_events_total != 0 else 0

                if parent_id == 'root':
                    p_gate_parent = p_gate_total
                else:
                    n_events_parent_old = statistics_old[parent_id]['n_events_gate']
                    n_events_parent_new = gate_membership[parent_id].sum()
                    n_events_parent = n_events_parent_old + n_events_parent_new
                    p_gate_parent = n_events_gate / n_events_parent if n_events_parent != 0 else 0

                # print(gate_id, parent_id)

                # if gate has dimensions, calculate MFI and rCV for each channel
                if hasattr(gate_node.gate, 'dimensions'):
                    channels = [dim.id for dim in gate_node.gate.dimensions]
                    intensity = {}
                    rCV = {}
                    if gate_membership[gate_id].sum() > 0:
                        intensity = {channel: event_data[gate_membership[gate_id], pnn.index(channel)].mean() for channel in channels}
                        rCV = {channel: robust_cv(event_data[gate_membership[gate_id], pnn.index(channel)]) for channel in channels}

                    statistics[gate_id] = {'n_events_gate': n_events_gate, 'p_gate_total': p_gate_total, 'p_gate_parent': p_gate_parent, 'event_conc': np.nan, 'intensity': intensity, 'rCV': rCV}
                else:
                    statistics[gate_id] = {'n_events_gate': n_events_gate, 'p_gate_total': p_gate_total, 'p_gate_parent': p_gate_parent, 'event_conc': np.nan}


    return statistics

def calc_ribbon_plot(event_data, mask, fluoro_indices, transform, density_cutoff):
    heatmap = np.apply_along_axis(lambda x: np.histogram(x, bins=transform.scale)[0], axis=0, arr=event_data[mask][:, fluoro_indices])

    # make sure all unit bins get lowest LUT
    if density_cutoff > 0:
        max_value = heatmap.max()
        mask_1 = (heatmap > 1)
        heatmap[mask_1] += max_value//255+1  # Maps to LUT[1]

    return heatmap


def calc_dotplot2d(event_data, source_and_child_gates, gate_membership, id_channel_x, id_channel_y, transform_x, transform_y, density_cutoff):
    dotmap = np.zeros([len(transform_x.scale)-1, len(transform_y.scale)-1])
    gate_list_ordered = list(gate_membership.keys())
    for gate in source_and_child_gates:
        mask = gate_membership[gate]
        gate_key = gate_list_ordered.index(gate)

        x = event_data[mask, id_channel_x]
        y = event_data[mask, id_channel_y]

        # Calculate 2D histogram (density)
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[transform_x.scale, transform_y.scale])

        mask_1 = (heatmap > density_cutoff)
        dotmap[mask_1] = gate_key

    return dotmap


def calc_hist2d(event_data, mask, id_channel_x, id_channel_y, transform_x, transform_y, density_cutoff):
    x = event_data[mask, id_channel_x]
    y = event_data[mask, id_channel_y]

    # Calculate 2D histogram (density)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[transform_x.scale, transform_y.scale])

    # make sure all unit bins get lowest LUT
    global_max_value = heatmap.max()
    inside_max_value = heatmap[1:-1,1:-1].max()

    if inside_max_value < global_max_value:
        # heatmap[0,:] *= inside_max_value
        # heatmap[-1,:] *= inside_max_value
        # heatmap[1:-1,0] *= inside_max_value
        # heatmap[1:-1,-1] *= inside_max_value
        # heatmap[0,:] /= global_max_value
        # heatmap[-1,:] /= global_max_value
        # heatmap[1:-1,0] /= global_max_value
        # heatmap[1:-1,-1] /= global_max_value
        heatmap[0,:] *= inside_max_value/global_max_value
        heatmap[-1,:] *= inside_max_value/global_max_value
        heatmap[1:-1,0] *= inside_max_value/global_max_value
        heatmap[1:-1,-1] *= inside_max_value/global_max_value

    if density_cutoff > 0:
        if inside_max_value > 0:
            mask_1 = (heatmap >= density_cutoff)
            heatmap[mask_1] += inside_max_value//255+1  # Maps to LUT[1]

            global_max_value = np.percentile(heatmap[mask_1], 99.9)
            np.clip(heatmap, 0, global_max_value, out=heatmap)

    return heatmap


def calc_hist1d(event_data, mask, id_channel, transform):
    x = event_data[mask, id_channel]

    # Calculate 1D histogram
    count, xedges = np.histogram(x, bins=transform.scale)
    return count  # note need to pad length + 1 at end

def raw_gates_list(gating):
    gate_ids = gating.get_gate_ids()
    gate_list = [g[0] for g in gate_ids]
    return gate_list

def get_set_or_initialise_label_offset(plot, gate_name, label_offset=None):
    if 'label_offsets' not in plot:
        plot['label_offsets'] = {gate_name: label_offset} #initialise
    else:
        if label_offset is None:
            if gate_name in plot['label_offsets']:
                label_offset = plot['label_offsets'][gate_name] #get
        else:
            plot['label_offsets'][gate_name] = label_offset #set

    return label_offset

def rename_label_offset(plot, old_gate_name, gate_name):
    if 'label_offsets' in plot:
        if old_gate_name in plot['label_offsets']:
            plot['label_offsets'][gate_name] = plot['label_offsets'][old_gate_name]
            plot['label_offsets'].pop(old_gate_name)

def build_display_label_map(pnn, spectral_model):
    """
    Returns a dict {pnn_name: display_label} for all channels.
    Fluorophore channels with a non-empty antigen get "Antigen Label";
    all other channels map to themselves.
    """
    label_to_antigen = {
        control['label']: control.get('antigen', '')
        for control in (spectral_model or [])
    }
    result = {}
    for name in (pnn or []): # guard against None
        antigen = label_to_antigen.get(name, '')
        result[name] = f'{antigen} {name}'.strip() if antigen else name
    return result