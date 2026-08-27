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

NPDP_DIR="$(pwd)/NPDP_bench"
NPDP_GIT="https://github.com/Zetsyog/NPDP_Bench.git"
NPDP_TAG="f75ad9df988443c462a576f68ae887f42913c936"

ROOT_DIR="$(pwd)"

############################
# Functions
############################

function fetch_pesto() {
	if [ -d "${PESTO_DIR}" ]; then
		if [ -f "${PESTO_BUILD_DIR}/cli/pesto" ]; then
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

function fetch_npdp_bench() {
	if [ -d "${NPDP_DIR}" ]; then
		echo "NPDP_Bench already fetched. Skipping."
		return
	fi
	git clone "$NPDP_GIT" NPDP_bench
	cd "${NPDP_DIR}" || exit
	git checkout "$NPDP_TAG"
}

echo "Fetching dependencies..."
echo "Fetching pesto..."
fetch_pesto
echo "done."
echo "Fetching pluto..."
fetch_pluto
echo "done."
echo "Fetching NPDP_Bench..."
fetch_npdp_bench
echo "done."
echo "Submodules fetched."
