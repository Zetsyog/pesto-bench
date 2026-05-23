CC?=gcc
CFLAGS?=-march=native -O3 -fopenmp
LDFLAGS=-I${ROOT_DIR}/include -lm

POLYCC?=polycc
PLUTO_FLAGS?=--parallel --nounroll --prevector

PESTO?=pesto
PESTO_TPZ_CONFIG?=${ROOT_DIR}/pesto_config/tpz.json
PESTO_FLAGS?=--indent

${SRC}.pluto.c: ${SRC}.c
	${POLYCC} ${PLUTO_FLAGS} $^ -o $@

${SRC}.tpz.c: ${SRC}.c
	${PESTO} ${PESTO_FLAGS} --config ${PESTO_TPZ_CONFIG} $^ -o $@

baseline: ${SRC}.c
	${CC} ${CFLAGS} ${ROOT_DIR}/lib/benchmark.c $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

trapez: ${SRC}.tpz.c
	${CC} ${CFLAGS} ${ROOT_DIR}/lib/benchmark.c $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

pluto: ${SRC}.pluto.c
	${CC} ${CFLAGS} ${ROOT_DIR}/lib/benchmark.c $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

check-pluto: 
	make -s baseline pluto -B EXTRA_FLAGS="-DBENCHMARK_DUMP_ARRAYS" >/dev/null 2>&1
	./baseline 2>baseline.log
	./pluto 2>pluto.log
	sha256sum baseline.log pluto.log

check-tpz: 
	make -s baseline trapez -B EXTRA_FLAGS="-DBENCHMARK_DUMP_ARRAYS" >/dev/null 2>&1
	./baseline 2>baseline.log
	./trapez 2>trapez.log
	sha256sum baseline.log trapez.log

clean:
	rm -f baseline
	rm -f pluto ${SRC}.pluto.c
	rm -f trapez ${SRC}.tpz.c
	rm -f *.o *.cloog *.log