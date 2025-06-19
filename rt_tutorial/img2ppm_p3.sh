#!/bin/bash
# Conversão genérica de qualquer imagem para PPM P3 (texto)
# Uso: ./img2ppm_p3.sh input.jpg output.ppm largura altura

if [ "$#" -ne 4 ]; then
    echo "Uso: $0 <input_img> <output_ppm> <largura> <altura>"
    exit 1
fi

INPUT_IMG="$1"
OUTPUT_PPM="$2"
WIDTH="$3"
HEIGHT="$4"

# Arquivo temporário para PPM binário (P6)
TMP_PPM="$(mktemp --suffix=.ppm)"

# 1. Converter para PPM P6 (binário)
ffmpeg -y -i "$INPUT_IMG" -vf scale=${WIDTH}:${HEIGHT} "$TMP_PPM"

# 2. Converter PPM P6 para PPM P3 (texto)
pnmtoplainpnm "$TMP_PPM" > "$OUTPUT_PPM"

# 3. Remover temporário
test -f "$TMP_PPM" && rm "$TMP_PPM"

echo "Arquivo $OUTPUT_PPM gerado com sucesso!"
