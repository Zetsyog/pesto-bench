SRC=symm

include ../../../common.mk

original: $(SRC).original.c
	${CC} ${CFLAGS} ${POLYBENCH_SRC} $^ -o $@ ${LDFLAGS} ${EXTRA_FLAGS}

check-baseline: original baseline
	./original 2>original.log
	./baseline 2>baseline.log
	sha256sum original.log baseline.log