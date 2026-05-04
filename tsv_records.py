"""tab-separated serialisation for analysis outputs.

Two layouts are used.

* ``write_table`` / ``read_table`` store a list of records as a columnar
  TSV with a header row.  Scalars are written in plain text (``true`` and
  ``false`` for booleans, ``null`` for a null value, an empty cell for a
  field the record does not have, ``repr`` for floats).  A nested list or mapping inside a cell is written compactly in
  bracket notation (the same convention already used for list-valued cells
  elsewhere in this repository).  A text cell that would otherwise be
  misread as a number, boolean, empty value or nested value is prefixed with
  a backslash, which ``read_table`` strips.

* ``write_record`` / ``read_record`` store any nested mapping as a
  three-column TSV with the header ``key``, ``type`` and ``value``.  Keys
  are dotted paths; list positions appear as integers.  Every mapping and
  list contributes its own row (type ``dict`` or ``list``, value = number of
  children) before its children, so empty containers and element order
  round-trip exactly.  Dots, tabs, newlines and backslashes inside key
  segments are backslash-escaped.
"""
import csv
import gzip
import json
import math
from pathlib import Path
import sys

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

RECORD_HEADER = ('key', 'type', 'value')
# list-valued cells can be very long (thousands of identifiers); lift the csv module's default field cap
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
_INT_CHARS = set('0123456789+-')


def _native(item):
    if _np is not None:
        if isinstance(item, _np.ndarray):
            return [_native(x) for x in item.tolist()]
        if isinstance(item, _np.generic):
            return item.item()
    if isinstance(item, dict):
        return {str(k): _native(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return [_native(v) for v in item]
    if isinstance(item, Path):
        return str(item)
    return item


def _open(path, mode):
    path = Path(path)
    if path.suffix == '.gz':
        return gzip.open(path, mode + 't', encoding='utf-8', newline='')
    return path.open(mode, encoding='utf-8', newline='')


def _escape_text(text):
    return text.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')


def _unescape_text(text):
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text):
            n = text[i + 1]
            out.append({'t': '\t', 'n': '\n', 'r': '\r', '\\': '\\', '.': '.'}.get(n, n))
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def _format_float(value):
    if math.isnan(value) or math.isinf(value):
        raise ValueError('non-finite float cannot be stored')
    return repr(float(value))


# ---------------------------------------------------------------- tables

def _looks_scalar(text):
    """return the scalar a plain text cell would be read back as, or None if it stays text."""
    if text == '' or text in ('true', 'false', 'null'):
        return True
    if text[:1] in '[{\\':
        return True
    try:
        int(text)
        return True
    except ValueError:
        pass
    try:
        float(text)
        return True
    except ValueError:
        return False


ABSENT = object()


def _cell_out(value):
    value = _native(value)
    if value is ABSENT:
        return ''
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(',', ':'), allow_nan=False)
    text = _escape_text(str(value))
    if _looks_scalar(text):
        text = '\\' + text
    return text


def _cell_in(text):
    if text == '':
        return ABSENT
    if text == 'null':
        return None
    if text == 'true':
        return True
    if text == 'false':
        return False
    if text[0] == '\\':
        return _unescape_text(text[1:])
    if text[0] in '[{':
        return json.loads(text)
    try:
        if set(text) <= _INT_CHARS:
            return int(text)
        return float(text)
    except ValueError:
        return _unescape_text(text)


def write_table(path, rows, fields=None):
    rows = [_native(row) for row in rows]
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with _open(path, 'w') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([_cell_out(row[field]) if field in row else '' for field in fields])


def _row(fields, line):
    return {field: value for field, value in zip(fields, (_cell_in(cell) for cell in line)) if value is not ABSENT}


def read_table(path):
    with _open(path, 'r') as handle:
        reader = csv.reader(handle, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        try:
            fields = next(reader)
        except StopIteration:
            return []
        return [_row(fields, line) for line in reader if line]


def iter_table(path):
    with _open(path, 'r') as handle:
        reader = csv.reader(handle, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        fields = next(reader, None)
        if fields is None:
            return
        for line in reader:
            if line:
                yield _row(fields, line)


def iter_table_rows(handle):
    """yield records from an already open text handle positioned at a table header."""
    reader = csv.reader(handle, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
    fields = next(reader, None)
    if fields is None:
        return
    for line in reader:
        if line:
            yield _row(fields, line)


def append_table_rows(handle, rows, fields, header):
    """stream rows into an already open text handle (used for large files)."""
    writer = csv.writer(handle, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
    if header:
        writer.writerow(fields)
    for row in rows:
        row = _native(row)
        writer.writerow([_cell_out(row[field]) if field in row else '' for field in fields])


# --------------------------------------------------------------- records

def _escape_key(segment):
    return _escape_text(segment).replace('.', '\\.')


def _split_key(key):
    parts, cur, i = [], [], 0
    while i < len(key):
        c = key[i]
        if c == '\\' and i + 1 < len(key):
            cur.append(c + key[i + 1])
            i += 2
        elif c == '.':
            parts.append(''.join(cur))
            cur = []
            i += 1
        else:
            cur.append(c)
            i += 1
    parts.append(''.join(cur))
    return [_unescape_text(p) for p in parts]


def _flatten(value, prefix, rows):
    if isinstance(value, dict):
        rows.append((prefix, 'dict', str(len(value))))
        for key, item in value.items():
            _flatten(item, (prefix + '.' if prefix else '') + _escape_key(str(key)), rows)
    elif isinstance(value, list):
        rows.append((prefix, 'list', str(len(value))))
        for index, item in enumerate(value):
            _flatten(item, (prefix + '.' if prefix else '') + str(index), rows)
    elif value is None:
        rows.append((prefix, 'null', ''))
    elif isinstance(value, bool):
        rows.append((prefix, 'bool', 'true' if value else 'false'))
    elif isinstance(value, int):
        rows.append((prefix, 'int', str(value)))
    elif isinstance(value, float):
        rows.append((prefix, 'float', _format_float(value)))
    elif isinstance(value, str):
        rows.append((prefix, 'str', _escape_text(value)))
    else:
        raise TypeError(type(value).__name__)


def record_rows(value):
    rows = []
    _flatten(_native(value), '', rows)
    return rows


def write_record(path, value):
    with _open(path, 'w') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        writer.writerow(RECORD_HEADER)
        writer.writerows(record_rows(value))


def record_from_rows(rows):
    root = None
    containers = {}
    for key, kind, text in rows:
        if kind == 'dict':
            value = {}
        elif kind == 'list':
            value = []
        elif kind == 'null':
            value = None
        elif kind == 'bool':
            value = text == 'true'
        elif kind == 'int':
            value = int(text)
        elif kind == 'float':
            value = float(text)
        elif kind == 'str':
            value = _unescape_text(text)
        else:
            raise ValueError('unknown type ' + kind)
        if key == '':
            root = value
        else:
            parts = _split_key(key)
            parent = containers['.'.join(_escape_key(p) for p in parts[:-1])]
            if isinstance(parent, list):
                parent.append(value)
            else:
                parent[parts[-1]] = value
        if kind in ('dict', 'list'):
            containers[key] = value
    return root


def read_record(path):
    with _open(path, 'r') as handle:
        reader = csv.reader(handle, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
        header = next(reader)
        if tuple(header) != RECORD_HEADER:
            raise ValueError('not a record file: ' + str(path))
        return record_from_rows([tuple(line) for line in reader if line])


def record_text(value):
    """the exact text ``write_record`` would produce (for hashing or printing)."""
    import io
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, quotechar=None, escapechar=None)
    writer.writerow(RECORD_HEADER)
    writer.writerows(record_rows(value))
    return buffer.getvalue()


# ------------------------------------------------------------ dispatch

def is_table(value):
    """true when ``value`` is a non-empty list of mappings, the shape stored as a columnar table."""
    return isinstance(value, list) and len(value) > 0 and all(isinstance(row, dict) for row in value)


def write_output(path, value):
    """store ``value`` as a columnar table when it is a list of records, otherwise as a key/type/value record."""
    value = _native(value)
    if is_table(value):
        fields = list(dict.fromkeys(key for row in value for key in row))
        if tuple(fields) == RECORD_HEADER:
            raise ValueError('a table with exactly the columns key, type, value cannot be told apart from a record')
        write_table(path, value, fields)
    else:
        write_record(path, value)


def read_output(path):
    """read a file written by ``write_output``, detecting the layout from its header."""
    with _open(path, 'r') as handle:
        header = handle.readline().rstrip('\n').split('\t')
    if tuple(header) == RECORD_HEADER:
        return read_record(path)
    return read_table(path)
