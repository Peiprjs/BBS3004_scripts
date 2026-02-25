#!/bin/bash

orig_dir="Original MP4"
conv_dir="Converted AVI"

mkdir -p "$orig_dir"
mkdir -p "$conv_dir"

for input_file in *.mp4; do
    
    [ -e "$input_file" ] || continue
    filename="${input_file%.*}"
    output_file="${filename}.avi"
    
    echo "Processing: $input_file -> $output_file"
    
    if ffmpeg -i "$input_file" -c:v mpeg4 -qscale:v 3 -c:a libmp3lame -qscale:a 4 "$conv_dir/$output_file"; then
        
        echo "Success! Moving '$input_file' to '$orig_dir/'"
        mv "$input_file" "$orig_dir/"
        
    else
        echo "Error: Failed to convert '$input_file'."
    fi

done
