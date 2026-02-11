#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/USER_GUIDE.md"
DOCS_DIR="$ROOT_DIR/documentation"
APPENDIX_DIR="$DOCS_DIR/appendices"

if [[ ! -f "$SRC" ]]; then
  echo "Source file not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DOCS_DIR" "$APPENDIX_DIR"

extract_section() {
  local heading="$1"
  local output_file="$2"
  local replace_with="$3"

  awk -v start_heading="## ${heading}" -v replace_with="$replace_with" '
    function starts_with_hash2(line) {
      return line ~ /^## /
    }

    {
      if ($0 == start_heading) {
        in_section = 1
        first_line = 1
      } else if (in_section && starts_with_hash2($0)) {
        exit
      }

      if (in_section) {
        if (first_line) {
          sub(/^## /, replace_with " ")
          first_line = 0
        }
        print
      }
    }
  ' "$SRC" > "$output_file"

  if [[ ! -s "$output_file" ]]; then
    echo "Failed to extract section: ${heading}" >&2
    exit 1
  fi
}

# Home page: title + metadata + introduction section
sed -n '1,6p' "$SRC" > "$DOCS_DIR/index.md"
echo >> "$DOCS_DIR/index.md"
extract_section "Introduction" "$DOCS_DIR/.tmp_introduction.md" "##"
cat "$DOCS_DIR/.tmp_introduction.md" >> "$DOCS_DIR/index.md"
rm -f "$DOCS_DIR/.tmp_introduction.md"

extract_section "Quick Start" "$DOCS_DIR/quick-start.md" "#"
extract_section "Installation" "$DOCS_DIR/installation.md" "#"
extract_section "Complete Workflow" "$DOCS_DIR/complete-workflow.md" "#"
extract_section "Detailed Usage" "$DOCS_DIR/detailed-usage.md" "#"
extract_section "Command Reference" "$DOCS_DIR/command-reference.md" "#"
extract_section "Interpreting Results" "$DOCS_DIR/interpreting-results.md" "#"
extract_section "Troubleshooting" "$DOCS_DIR/troubleshooting.md" "#"
extract_section "FAQ" "$DOCS_DIR/faq.md" "#"
extract_section "Citation" "$DOCS_DIR/citation.md" "#"
extract_section "Support" "$DOCS_DIR/support.md" "#"
extract_section "License" "$DOCS_DIR/license.md" "#"
extract_section "Appendix A: Model Architecture Details" "$APPENDIX_DIR/model-architecture.md" "#"
extract_section "Appendix B: Experimental Protocol" "$APPENDIX_DIR/experimental-protocol.md" "#"
extract_section "Appendix C: Method References" "$APPENDIX_DIR/method-references.md" "#"

echo "MkDocs source pages regenerated in: $DOCS_DIR"
