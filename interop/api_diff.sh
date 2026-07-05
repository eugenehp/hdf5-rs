#!/bin/bash
# Mechanical public-API diff: reference FFI crate vs this crate (name-level).
REF="${1:-/Users/Shared/hdf5-rust}"
OURS="$(cd "$(dirname "$0")/.." && pwd)"
for crate in hdf5 hdf5-types hdf5-derive; do
  a=$(mktemp); b=$(mktemp)
  grep -rhoE 'pub (fn|struct|enum|trait|const|type) [A-Za-z_0-9]+' "$REF/$crate/src" 2>/dev/null \
    | grep -vE 'pub fn (test_|with_tmp_)' | grep -v 'struct TestObject' | sort -u > "$a"
  grep -rhoE 'pub (fn|struct|enum|trait|const|type) [A-Za-z_0-9]+' "$OURS/$crate/src" 2>/dev/null \
    | grep -vE 'pub fn (test_|with_tmp_)' | sort -u > "$b"
  echo "== $crate: $(wc -l < "$a" | tr -d ' ') ref items, missing $(comm -23 "$a" "$b" | wc -l | tr -d ' '), extra $(comm -13 "$a" "$b" | wc -l | tr -d ' ')"
  comm -23 "$a" "$b" | sed 's/^/  MISSING /'
  rm -f "$a" "$b"
done
