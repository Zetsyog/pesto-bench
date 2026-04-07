#!/usr/bin/env bash

# Initialize and start the application

############################
# Configuration
############################
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
		patch -p1 <"${ROOT_DIR}/patches/pluto-inscop-cc-fix.patch"
		./configure
		make -j
	)
}
echo "Fetching dependencies..."
echo "Fetching pesto..."
fetch_pesto
echo "Pesto fetched."
echo "Fetching pluto..."
fetch_pluto
echo "Pluto fetched."
echo "Submodules fetched."
