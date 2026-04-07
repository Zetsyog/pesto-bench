/*
 * Discretized 2D heat equation stencil with non periodic boundary conditions
 * Adapted from Pochoir test bench
 *
 * Irshad Pananilath: irshad@csa.iisc.ernet.in
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/*
 * N is the number of points
 * T is the number of timesteps
 */
#ifndef N
#define N 4000L
#endif /* N */
#ifndef T
#define T 1000L
#endif /* T */

#define NUM_FP_OPS 10
#define BENCHMARK_NUM_FP_OPS (NUM_FP_OPS) * (N - 2) * (N - 2) * (T)

#include <benchmark.h>

void init_array(param_t tsteps, param_t n,
				data_t BENCHMARK_3D(A, 2, N, N, 2, n, n)) {
	iter_t i, j;

	srand(BENCHMARK_RSEED);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			A[0][i][j] = 1.0 * (rand() % BENCHMARK_INIT_BASE);
		}
	}
}

void dump_array(param_t tsteps, param_t n,
				data_t BENCHMARK_3D(A, 2, N, N, 2, n, n)) {
	iter_t i, j;

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
		for (j = 0; j < N; j++) {
			sum += A[T % 2][i][j];
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			sum_err_sqr +=
				(A[T % 2][i][j] - (sum / N)) * (A[T % 2][i][j] - (sum / N));
		}
	}
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			chtotal += ((char *)A[T % 2][i])[j];
		}
	}
	fprintf(BENCHMARK_DUMP_FILE, "sum: %e\n", sum);
	fprintf(BENCHMARK_DUMP_FILE, "rms(A) = " DATA_PRINTF_MODIFIER "\n",
			SQRT_FUN(sum_err_sqr));
	fprintf(BENCHMARK_DUMP_FILE, "sum(rep(A)) = %d\n", chtotal);
#endif
#ifdef BENCHMARK_DUMP_ARRAYS
	BENCHMARK_DUMP_BEGIN("A");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			fprintf(BENCHMARK_DUMP_FILE, " " DATA_PRINTF_MODIFIER,
					A[T % 2][i][j]);
		}
		fprintf(BENCHMARK_DUMP_FILE, "\n");
	}
#endif /* BENCHMARK_DUMP_ARRAYS */
	BENCHMARK_DUMP_END("A");
}

void kernel_heat_2d(param_t tsteps, param_t n,
					data_t BENCHMARK_3D(A, 2, N, N, 2, n, n)) {
	iter_t t, i, j;

#pragma scop
	for (int t = 0; t < T; t++) {
		for (int i = 1; i < N - 1; i++) {
			for (int j = 1; j < N - 1; j++) {
				A[(t + 1) % 2][i][j] =
					0.125 * (A[t % 2][i + 1][j] - 2.0 * A[t % 2][i][j] +
							 A[t % 2][i - 1][j]) +
					0.125 * (A[t % 2][i][j + 1] - 2.0 * A[t % 2][i][j] +
							 A[t % 2][i][j - 1]) +
					A[t % 2][i][j];
			}
		}
	}
#pragma endscop
}

int main(int argc, char *argv[]) {
	param_t t = T;
	param_t n = N;

	BENCHMARK_3D_ARRAY_DECL(A, data_t, 2, N, N, 2, n, n);

	/* initialization */
	init_array(t, n, A);

	/* start timer */
	benchmark_measure_start();

	/* kernel execution */
	kernel_heat_2d(t, n, A);

	/* stop timer */
	benchmark_measure_stop();

	/* compute and print statistics */
	benchmark_measure_print();

#ifdef BENCHMARK_DUMP
	dump_array(t, n, A);
#endif

	BENCHMARK_FREE_ARRAY(A);

	return 0;
}
