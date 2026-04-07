/*
 * Discretized 2D heat equation stencil with non periodic boundary conditions
 * Adapted from Pochoir test bench
 *
 * Irshad Pananilath: irshad@csa.iisc.ernet.in
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * N is the number of points
 * T is the number of timesteps
 */
#ifndef N
#define N 1600000L
#endif
#ifndef T
#define T 1000L
#endif

#define NUM_FP_OPS 4
#define BENCHMARK_NUM_FP_OPS (NUM_FP_OPS) * (N - 2) * T

#include <benchmark.h>

void init_arrays(param_t tsteps, param_t n,
				 data_t BENCHMARK_2D(A, 2, N, 2, n)) {
	iter_t i;

	srand(BENCHMARK_RSEED);

	for (i = 0; i < N; i++) {
		A[0][i] = 1.0 * (rand() % BENCHMARK_INIT_BASE);
	}
}

void dump_arrays(param_t tsteps, param_t n,
				 data_t BENCHMARK_2D(A, 2, N, 2, n)) {
	iter_t i;

	BENCHMARK_DUMP_START();
#ifdef BENCHMARK_DUMP_CHKSUM
	BENCHMARK_DUMP_BEGIN("parameters");
	fprintf(BENCHMARK_DUMP_FILE, "T=" PARAM_PRINTF_MODIFIER "\n", tsteps);
	fprintf(BENCHMARK_DUMP_FILE, "N=" PARAM_PRINTF_MODIFIER "\n", n);
	BENCHMARK_DUMP_END("parameters");
	BENCHMARK_DUMP_BEGIN("checksum");
	// compute checksum
	data_t sum = 0.0;
	data_t sum_err_sqr = 0.0;
	int chtotal = 0;

	for (i = 0; i < N; i++) {
		sum += A[T % 2][i];
	}
	fprintf(BENCHMARK_DUMP_FILE, "sum: %e\t", sum);

	for (i = 0; i < N; i++) {
		sum_err_sqr += (A[T % 2][i] - (sum / N)) * (A[T % 2][i] - (sum / N));
	}
	fprintf(BENCHMARK_DUMP_FILE, "rms(A) = " DATA_PRINTF_MODIFIER "\n",
			SQRT_FUN(sum_err_sqr));
	for (i = 0; i < N; i++) {
		chtotal += ((char *)A[T % 2])[i];
	}
	fprintf(BENCHMARK_DUMP_FILE, "sum(rep(A)) = %d\n", chtotal);
	BENCHMARK_DUMP_END("checksum");
#endif
#ifdef BENCHMARK_DUMP_ARRAYS
	BENCHMARK_DUMP_BEGIN("A");
	for (iter_t i = 0; i < N; ++i) {
		fprintf(BENCHMARK_DUMP_FILE, " " DATA_PRINTF_MODIFIER, A[T % 2][i]);
		if (i % 10 == 9) {
			fprintf(BENCHMARK_DUMP_FILE, "\n");
		}
	}
#endif /* BENCHMARK_DUMP_ARRAYS */
	BENCHMARK_DUMP_STOP();
}

void kernel_heat_1d(param_t tsteps, param_t n,
					data_t BENCHMARK_2D(A, 2, N, 2, n)) {
	iter_t t, i;
#pragma scop
	for (t = 0; t < T; t++) {
		for (i = 1; i < N - 1; i++) {
			A[(t + 1) % 2][i] =
				0.250 * (A[t % 2][i + 1] - 2.0 * A[t % 2][i] + A[t % 2][i - 1]);
		}
	}
#pragma endscop
}

int main(int argc, char *argv[]) {
	param_t n = N;

	BENCHMARK_2D_ARRAY_DECL(A, data_t, 2, N, 2, n);

	/* initialization */
	init_arrays(T, N, A);

	/* start timer */
	benchmark_measure_start();

	/* execute the main kernel */
	kernel_heat_1d(T, N, A);

	/* stop timer */
	benchmark_measure_stop();

	/* compute and print statistics */
	benchmark_measure_print();

#ifdef BENCHMARK_DUMP
	dump_arrays(T, N, A);
#endif /* BENCHMARK_DUMP */

	BENCHMARK_FREE_ARRAY(A);

	return 0;
}