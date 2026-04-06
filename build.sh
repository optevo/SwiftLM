#!/bin/bash
set -e

echo "=> Initializing submodules..."
git submodule update --init --recursive

echo "=> Building SwiftLM (release)..."
swift build -c release

echo "=> Copying default.metallib..."
METALLIB_SRC="LocalPackages/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/default.metallib"
METALLIB_DEST=".build/arm64-apple-macosx/release/"

# Also resolving the generic release symlink folder just in case
METALLIB_DEST_SYMLINK=".build/release/"

if [ -f "$METALLIB_SRC" ]; then
    mkdir -p "$METALLIB_DEST"
    cp "$METALLIB_SRC" "$METALLIB_DEST"
    
    if [ -d "$METALLIB_DEST_SYMLINK" ]; then
        cp "$METALLIB_SRC" "$METALLIB_DEST_SYMLINK"
    fi
    
    # Also copying to root to be safe
    cp "$METALLIB_SRC" ./
    
    echo "✅ Successfully copied default.metallib."
else
    echo "⚠️  Warning: $METALLIB_SRC not found. MLX GPU operations may fail."
fi

echo "=> Build complete!"
