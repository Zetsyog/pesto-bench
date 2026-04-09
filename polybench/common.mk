CC?=gcc
CFLAGS?=-march=native -O3 -fopenmp
LDFLAGS=-I${ROOT_DIR}/polybench/utilities  -lm
POLYBENCH_SRC=${ROOT_DIR}/polybench/utilities/polybench.c
EXTRA_FLAGS?=-DPOLYBENCH_TIME -DLARGE_DATASET

POLYCC?=polycc
PLUTO_FLAGS?=--tile --parallel --nounroll --prevector

PESTO?=pesto
PESTO_ATILING_CONFIG?=
PESTO_FLAGS?=--indent

all: baseline pluto

${SRC}.pluto.c: ${SRC}.c
	${POLYCC} ${PLUTO_FLAGS} $^ -o $@

${SRC}.pesto.c: ${SRC}.c
	${PESTO} ${PESTO_FLAGS} --config ${PESTO_CONFIG} $^ -o $@

baseline: ${SRC}.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

pesto: ${SRC}.pesto.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

pluto: ${SRC}.pluto.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

check-pluto: original pluto
	./original 2>original.log
	./pluto 2>pluto.log
	sha256sum original.log pluto.log

clean:
	rm -f ${SRC}.pluto.c ${SRC}.pesto.c baseline pesto pluto
	rm -f *.o *.cloog *.log