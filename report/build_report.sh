#!/usr/bin/env bash
# Produit report/rapport.pdf (nécessite pandoc + LaTeX) ; sinon exportez le .md vers Word/PDF.
set -euo pipefail
cd "$(dirname "$0")"
pandoc rapport.md -o rapport.pdf --pdf-engine=xelatex -V geometry:margin=2.2cm -V fontsize=11pt
