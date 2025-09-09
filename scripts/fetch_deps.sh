#!/usr/bin/env bash

# Initialize and start the application

############################
# Configuration
############################
POLYBENCH_VERSION=4.2
POLYBENCH_DIR="$(pwd)/polybench"
POLYBENCH_URL="https://sourceforge.net/projects/polybench/files/polybench-c-${POLYBENCH_VERSION}.tar.gz/download"

PESTO_DIR="$(pwd)/pesto"
PESTO_BUILD_DIR="${PESTO_DIR}/build"
PESTO_GIT="https://gitlab.inria.fr/crossett/pesto.git"
PESTO_TAG="dev"

PLUTO_DIR="$(pwd)/pluto"
PLUTO_VERSION=0.13.0
PLUTO_URL="https://github.com/bondhugula/pluto/releases/download/${PLUTO_VERSION}/pluto-${PLUTO_VERSION}.tgz"

ROOT_DIR="$(pwd)"

############################
# Functions
############################
function download_and_extract_polybench() {
    if [ -d "${POLYBENCH_DIR}" ]; then
        echo "PolyBench directory already exists. Skipping download."
        return
    fi
    echo "Downloading and extracting PolyBench..."
    wget -q "$POLYBENCH_URL" -O polybench.tar.gz
    tar -xzf polybench.tar.gz
    rm polybench.tar.gz
    mv polybench-c-* "${POLYBENCH_DIR}"
    echo "Applying patches to PolyBench..."
    # Apply any necessary patches here
    (
        cd "${POLYBENCH_DIR}" || exit
        patch -p1 <"${ROOT_DIR}/patches/polybench_init_from_data_file.patch"
    )
    echo "PolyBench downloaded and extracted."
}

function fetch_pesto() {
    if [ -d "${PESTO_DIR}" ]; then
        if [ -f "${PESTO_BUILD_DIR}/pesto" ]; then
            echo "Pesto already built. Skipping fetch."
            return
        else
            echo "Pesto directory exists but not built. Removing and re-fetching."
            rm -rf pesto
        fi
    fi
    git clone "$PESTO_GIT" pesto
    cd "${PESTO_DIR}" || exit
    git checkout "$PESTO_TAG"
    cd ..
    (
        cd "${PESTO_DIR}" || exit
        mkdir build
        cd build || exit
        cmake .. -DCMAKE_BUILD_TYPE=Release
        cmake --build . -j
    )
}

function fetch_pluto() {
    if [ -d "${PLUTO_DIR}" ]; then
        if [ -f "${PLUTO_DIR}/polycc" ]; then
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
    mv pluto-* "${PLUTO_DIR}"
    (
        cd "${PLUTO_DIR}" || exit
        patch -p1 <"${ROOT_DIR}/patches/pluto-shebang-fix.patch"
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
