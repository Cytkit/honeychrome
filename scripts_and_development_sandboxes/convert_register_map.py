import pandas as pd
import re

def sanitize_name(name):
    """Convert register name to a valid Python identifier."""
    name = str(name).strip()
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    if name and name[0].isdigit():
        name = '_' + name
    return name

def build_comment(row, keep_cols=('Group', 'Reset Value', 'Reset Value (Hex)', 'Details')):
    """
    Build a comment string from selected columns.
    keep_cols: iterable of column names to include (in order).
    """
    parts = []
    for col in keep_cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        s = str(val).strip()
        if not s:
            continue
        parts.append(s)
    comment = ' | '.join(parts)
    comment = comment.replace('\n', ' ').replace('\r', ' ')
    return comment

def import_register_map(csv_path):
    """
    Import the register map from the CSV file.
    Uses the 'Name' column as the key and 'Address (Hex)' as the value.
    """
    # The CSV has a two-row header (the second row contains bit numbers 15..0).
    # header=0 uses the first row as column names; the bit-number row becomes
    # the first data row and will be skipped naturally because it has no Name.
    df = pd.read_csv(csv_path, header=0, dtype=str)

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    registers_map = {}

    for _, row in df.iterrows():
        name = row.get('Name')
        addr_hex = row.get('Address (Hex)')

        # Skip rows with no name or address
        if pd.isna(name) or str(name).strip() == '':
            continue
        if pd.isna(addr_hex) or str(addr_hex).strip() == '':
            continue

        name_str = str(name).strip()
        addr_str = str(addr_hex).strip()

        # Parse "0x0010" style addresses
        m = re.match(r'0x([0-9A-Fa-f]+)', addr_str)
        if not m:
            continue
        addr_val = int(m.group(1), 16)

        comment = build_comment(row)

        key = sanitize_name(name_str)
        if key in registers_map:
            suffix = 2
            while f"{key}_{suffix}" in registers_map:
                suffix += 1
            key = f"{key}_{suffix}"

        registers_map[key] = (addr_val, comment)

    return registers_map


def print_registers_map(registers_map):
    print("registers_map = {")
    for key, (addr, comment) in registers_map.items():
        if comment:
            print(f"    '{key}':\t0x{addr:04X},\t#\t{comment}")
        else:
            print(f"    '{key}':\t0x{addr:04X},")
    print("}")


if __name__ == '__main__':
    csv_path = '/home/ssr/astute_nextcloud/Projects/CytKit/Documents/FPGA/FPGA Register Map - 2026-09-28.csv'

    regs = import_register_map(csv_path)
    print_registers_map(regs)

