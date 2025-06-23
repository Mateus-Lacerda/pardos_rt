#!/bin/bash

# Compiler definitions
NVCC="nvcc"
GPP="g++"

# Get the directory where the script is located. This is crucial for robust pathing.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Source files (use full paths relative to the script's directory)
# CUDA source files: main.cu (new!) and the kernel file
CUDA_SOURCES="${SCRIPT_DIR}/main.cu ${SCRIPT_DIR}/cuda_raytracer_kernel.cu ${SCRIPT_DIR}/sdl_renderer.cu"
# C++ source files: only sdl_renderer.cc now (and any other pure .cc files)
CPP_SOURCES=""

# Intermediate object files
# Object files for CUDA sources (each .cu compiles to its own .o)
CUDA_OBJS_LIST="" # This will store the list of compiled CUDA object files
# Object files for C++ sources
CPP_OBJS=""
if [ -n "$CPP_SOURCES" ]; then
    for src_path in $CPP_SOURCES; do
        filename=$(basename -- "$src_path")
        base="${filename%.cc}"
        CPP_OBJS+="${SCRIPT_DIR}/${base}.o "
    done
fi

# Output executable name
EXECUTABLE="${SCRIPT_DIR}/raytracer_cuda"

# Include directories
# -I${SCRIPT_DIR}: Include the script's directory for all your project headers
# $(sdl2-config --cflags): Get SDL2 include flags (requires pkg-config)
# -I/usr/local/cuda/include: Standard CUDA include path
INCLUDE_DIRS="-I${SCRIPT_DIR} $(sdl2-config --cflags) -I/usr/local/cuda/include"

# Library directories and libraries to link
# $(sdl2-config --libs): Get SDL2 linker flags (requires pkg-config)
# -L/usr/local/cuda/lib64: Standard CUDA library path
# -lcudart: Link CUDA Runtime library (essential for CUDA applications)
# -lm: Link math library
LINK_LIBS="$(sdl2-config --libs) -L/usr/local/cuda/lib64 -lcudart -lm"

echo "Compiling CUDA sources..."
# Compile each CUDA source file individually into its own .o file
for src_cu in $CUDA_SOURCES; do
    filename=$(basename -- "$src_cu")
    base="${filename%.cu}"
    obj_cu="${SCRIPT_DIR}/${base}.o" # Define the output object file for this .cu source
    CUDA_OBJS_LIST+="$obj_cu "       # Add the compiled object file to the list for linking

    echo "  Compiling $src_cu to $obj_cu"
    # -dc: Device-only compilation
    $NVCC --extended-lambda -std=c++17 -dc "$src_cu" $INCLUDE_DIRS -o "$obj_cu" -arch=sm_86
    if [ $? -ne 0 ]; then
        echo "CUDA compilation of $src_cu failed!"
        exit 1
    fi
done
echo "All CUDA sources compiled successfully."

if [ -n "$CPP_SOURCES" ]; then
    echo "Compiling C++ sources..."
    # Compile C++ source files
    for src_cpp in $CPP_SOURCES; do
        filename=$(basename -- "$src_cpp")
        base="${filename%.cc}"
        obj_cpp="${SCRIPT_DIR}/${base}.o"
        
        echo "  Compiling $src_cpp to $obj_cpp"
        # -c: Compile only (no linking)
        $GPP -std=c++17 -O2 -Wall -Wextra -fopenmp -c "$src_cpp" $INCLUDE_DIRS -o "$obj_cpp"
        if [ $? -ne 0 ]; then
            echo "C++ compilation of $src_cpp failed!"
            exit 1
        fi
    done
    echo "All C++ sources compiled successfully."
fi


echo "Linking..."
# Link all object files (C++ and CUDA) to create the final executable.
# NVCC is used for the final linking step in mixed projects as it handles CUDA libraries automatically.
$NVCC $CPP_OBJS $CUDA_OBJS_LIST -o "$EXECUTABLE" $LINK_LIBS -arch=sm_86
if [ $? -ne 0 ]; then
    echo "Linking failed!"
    exit 1
fi
echo "Linking successful. Executable: $EXECUTABLE"

# Clean up object files (optional)
# Uncomment the line below (`rm -f ...`) to remove intermediate object files after a successful build
# rm -f $CUDA_OBJS_LIST $CPP_OBJS

echo "Build complete."