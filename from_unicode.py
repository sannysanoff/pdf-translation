#!/usr/bin/env python3
"""
Script to convert Unicode escape sequences in JSON file to proper UTF-8 characters.
Renames original file to .backup if it doesn't exist, otherwise fails.
"""

import json
import os
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: python from_unicode.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)

    backup_path = file_path + '.backup'
    if os.path.exists(backup_path):
        print(f"Error: Backup file {backup_path} already exists")
        sys.exit(1)

    # First, rename to backup
    os.rename(file_path, backup_path)

    # Read from backup
    with open(backup_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Write to original path with proper UTF-8
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted {file_path}, original backed up as {backup_path}")


if __name__ == '__main__':
    main()