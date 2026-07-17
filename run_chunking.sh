#!/usr/bin/env bash
# Chunk each PDF in its own process, writing one JSONL per article, then
# combine them. Keeping each PDF in a separate process means a crash (or a
# killed session) only ever loses the file in flight, and progress is visible
# per article rather than as one opaque multi-hour run.
set -u

INPUT_DIR="${1:-data}"
PARTS_DIR="${2:-data_ext_vector/per_file}"
COMBINED="${3:-data_ext_vector/chunks.jsonl}"
PYTHON=".venv/Scripts/python.exe"

# Articles left out of the corpus.
#
# python.pdf is excluded on content: it's the programming-language article and
# has nothing to do with horology.
#
# The rest are excluded because marker has an unresolved memory problem on
# them. Balance_wheel alone grows to ~14GB resident on a 16GB machine, which
# swaps hard and turns a ~3-minute file into 35+ minutes. Capping surya's
# batch sizes (RECOGNITION_BATCH_SIZE et al) does NOT help — the caps are
# applied and honoured, and peak memory is unchanged, so the allocation is
# coming from somewhere else that hasn't been tracked down yet. The other
# three are the largest remaining PDFs and are assumed to hit the same wall.
SKIP_NAMES="${SKIP_NAMES:-python Balance_wheel Clock Seiko Watch}"

mkdir -p "$PARTS_DIR"

total=$(ls "$INPUT_DIR"/*.pdf 2>/dev/null | wc -l)
i=0
failed=()

for pdf in "$INPUT_DIR"/*.pdf; do
    i=$((i + 1))
    name=$(basename "$pdf" .pdf)
    part="$PARTS_DIR/$name.jsonl"

    if echo " $SKIP_NAMES " | grep -q " $name "; then
        echo "[$i/$total] SKIP $name (excluded)"
        continue
    fi

    if [ -s "$part" ]; then
        echo "[$i/$total] SKIP $name (already done)"
        continue
    fi

    echo "[$i/$total] START $name ($(date +%H:%M:%S))"
    # Keep marker's own progress output in a per-file log rather than
    # discarding it, so a slow file can be diagnosed while it runs.
    log="$PARTS_DIR/$name.log"
    if "$PYTHON" "extract_and_chunk.py" "$pdf" --output "$part" >"$log" 2>&1; then
        echo "[$i/$total] DONE  $name ($(wc -l < "$part") chunks, $(date +%H:%M:%S))"
        rm -f "$log"
    else
        echo "[$i/$total] FAIL  $name (see $log)"
        rm -f "$part"   # don't leave a partial file that would be skipped next run
        failed+=("$name")
    fi
done

echo "Combining parts into $COMBINED"
cat "$PARTS_DIR"/*.jsonl > "$COMBINED"
echo "Combined $(ls "$PARTS_DIR"/*.jsonl | wc -l) files -> $(wc -l < "$COMBINED") chunks"

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED (${#failed[@]}): ${failed[*]}"
    exit 1
fi
