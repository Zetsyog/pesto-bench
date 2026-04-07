CC?=gcc
CFLAGS?=-march=native -O3 -fopenmp
LDFLAGS=-I${ROOT_DIR}/include -L${ROOT_DIR}/lib -lm -lbenchmark

POLYCC?=polycc
PLUTO_FLAGS?=--parallel --nounroll --prevector

PESTO?=pesto
PESTO_ATILING_CONFIG?=
PESTO_FLAGS?=--indent

${SRC}.pluto.c: ${SRC}.c
	${POLYCC} ${PLUTO_FLAGS} $^ -o $@

${SRC}.pesto.c: ${SRC}.c
	${PESTO} ${PESTO_FLAGS} --config ${PESTO_CONFIG} $^ -o $@

baseline: ${SRC}.c
	${CC} ${CFLAGS}  $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

pesto: ${SRC}.pesto.c
	${CC} ${CFLAGS} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

pluto: ${SRC}.pluto.c
	${CC} ${CFLAGS} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

check-pluto: 
	make -s baseline pluto -B EXTRA_FLAGS="-DBENCHMARK_DUMP" >/dev/null 2>&1
	./baseline 2>baseline.log
	./pluto 2>pluto.log
	sha256sum baseline.log pluto.log

clean:
	rm -f ${SRC}.pluto.c ${SRC}.pesto.c baseline pesto pluto
	rm -f *.o *.cloog *.log