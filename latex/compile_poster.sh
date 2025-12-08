#!/bin/bash
# Poster Compilation Script

# Add MacTeX binaries to PATH
export PATH="/Library/TeX/texbin:$PATH"

cd "$(dirname "$0")"

echo "Compiling poster..."

# Run pdflatex first pass
pdflatex -interaction=nonstopmode poster.tex

# Run pdflatex second pass (resolve cross-references)
pdflatex -interaction=nonstopmode poster.tex

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Compilation successful! Output: poster.pdf"
    echo ""
    echo "Cleaning auxiliary files..."
    rm -f poster.aux poster.log poster.out poster.fls poster.fdb_latexmk poster.synctex.gz
    echo "✓ Cleanup complete!"
else
    echo ""
    echo "✗ Compilation failed. Check the errors above."
    exit 1
fi
