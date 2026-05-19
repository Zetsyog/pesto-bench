CC?=gcc
CFLAGS?=-march=native -O3 -fopenmp
LDFLAGS=-I${ROOT_DIR}/polybench/utilities  -lm
POLYBENCH_SRC=${ROOT_DIR}/polybench/utilities/polybench.c
EXTRA_FLAGS?=-DPOLYBENCH_TIME -DLARGE_DATASET

POLYCC?=polycc
PLUTO_FLAGS?=--tile --parallel --nounroll --prevector

PESTO?=pesto
PESTO_TPZ_CONFIG?=$(ROOT_DIR)/pesto_config/tpz.json
PESTO_ATILING_CONFIG?=$(ROOT_DIR)/pesto_config/atiling.json
PESTO_HYBRID_CONFIG?=$(ROOT_DIR)/pesto_config/hybrid.json
PESTO_FLAGS?=--indent

all: baseline pluto

${SRC}.pluto.c: ${SRC}.c
	${POLYCC} ${PLUTO_FLAGS} $^ -o $@

${SRC}.atiled.c: ${SRC}.c
	${PESTO} ${PESTO_FLAGS} --config ${PESTO_ATILING_CONFIG} $^ -o $@

${SRC}.hybrid.c: ${SRC}.c
	${PESTO} ${PESTO_FLAGS} --config ${PESTO_HYBRID_CONFIG} $^ -o $@	

${SRC}.tpz.c: ${SRC}.c
	${PESTO} ${PESTO_FLAGS} --config ${PESTO_TPZ_CONFIG} $^ -o $@

baseline: ${SRC}.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

atiled: ${SRC}.atiled.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

hybrid: ${SRC}.hybrid.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

pluto: ${SRC}.pluto.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

tpz: ${SRC}.tpz.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

check-pluto: baseline pluto
	./baseline 2>baseline.log
	./pluto 2>pluto.log
	sha256sum baseline.log pluto.log

check-atiled: baseline atiled
	./baseline 2>baseline.log
	./atiled 2>atiled.log
	sha256sum baseline.log atiled.log

clean:
	rm -f baseline
	rm -f pluto ${SRC}.pluto.c
	rm -f atiled ${SRC}.atiled.c
	rm -f hybrid ${SRC}.hybrid.c
	rm -f tpz ${SRC}.tpz.c
	rm -f *.trahrhe.*
	rm -f *.o *.cloog *.log