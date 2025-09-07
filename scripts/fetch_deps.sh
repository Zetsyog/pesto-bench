#!/usr/bin/env bash

# Initialize and start the application

############################
# Configuration
############################
POLYBENCH_URL="https://sourceforge.net/projects/polybench/files/polybench-c-${POLYBENCH_VERSION}.tar.gz/download"

############################
# Functions
############################
function download_and_extract_polybench() {
    if [ -d "polybench" ]; then
        echo "PolyBench directory already exists. Skipping download."
        return
    fi
    echo "Downloading and extracting PolyBench..."
    wget -q "$POLYBENCH_URL" -O polybench.tar.gz
    tar -xzf polybench.tar.gz
    rm polybench.tar.gz
    mv polybench-c-* polybench
    echo "PolyBench downloaded and extracted."
}

function fetch_pesto() {
    if [ -d "pesto" ]; then
        if [ -f "pesto/build/pesto" ]; then
            echo "Pesto already built. Skipping fetch."
            return
        else
            echo "Pesto directory exists but not built. Removing and re-fetching."
            rm -rf pesto
        fi
    fi
    git clone "$PESTO_GIT" pesto
    cd pesto || exit
    git checkout "$PESTO_COMMIT"
    cd ..
    (
        cd pesto || exit
        mkdir build
        cd build || exit
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . -j
    )
}

function fetch_pluto() {
    if [ -d "pluto" ]; then
        if [ -f "pluto/polycc" ]; then
            echo "Pluto already built. Skipping fetch."
            return
        else
            echo "Pluto directory exists but not built. Removing and re-fetching."
            rm -rf pluto
        fi
        return
    fi
    wget -q "$PLUTO_URL" -O pluto.tar.gz
    tar -xzf pluto.tar.gz
    rm pluto.tar.gz
    mv pluto-* pluto
    (
        cd pluto || exit
        ./configure
        make -j
    )
}
echo "Fetching dependencies..."
echo "Fetching polybench..."
download_and_extract_polybench
echo "Polybench fetched."
echo "Fetching pesto..."
fetch_pesto
echo "Pesto fetched."
echo "Fetching pluto..."
fetch_pluto
echo "Pluto fetched."
echo "Submodules fetched."
