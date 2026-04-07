/*
 * Discretized 3D heat equation stencil with non periodic boundary conditions
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
#define N 250L
#endif /* N */
#ifndef T
#define T 100L
#endif /* T */

#define NUM_FP_OPS 15
#define BENCHMARK_NUM_FP_OPS (NUM_FP_OPS) * (N - 2) * (N - 2) * (N - 2) * (T)

#include <benchmark.h>

void init_array(param_t tsteps, param_t n,
				data_t BENCHMARK_4D(A, 2, N, N, N, 2, n, n, n)) {
	iter_t i, j, k;

	srand(BENCHMARK_RSEED);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				A[0][i][j][k] = 1.0 * (rand() % BENCHMARK_INIT_BASE);
			}
		}
	}
}

void dump_array(param_t tsteps, param_t n,
				data_t BENCHMARK_4D(A, 2, N, N, N, 2, n, n, n)) {
	iter_t i, j, k;

	BENCHMARK_DUMP_START();
#ifdef BENCHMARK_DUMP_CHKSUM
	BENCHMARK_DUMP_BEGIN("parameters");
	fprintf(BENCHMARK_DUMP_FILE, "T: %ld\n", T);
	fprintf(BENCHMARK_DUMP_FILE, "N: %ld\n", N);
	BENCHMARK_DUMP_END("parameters");
	BENCHMARK_DUMP_BEGIN("checksum");
	// compute checksum
	data_t sum = 0.0;
	data_t sum_err_sqr = 0.0;
	int chtotal = 0;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				sum += A[T % 2][i][j][k];
			}
		}
	}
	fprintf(BENCHMARK_DUMP_FILE, "sum: %e\t", sum);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				sum_err_sqr += (A[T % 2][i][j][k] - (sum / N)) *
							   (A[T % 2][i][j][k] - (sum / N));
			}
		}
	}
	fprintf(BENCHMARK_DUMP_FILE, "rms(A) = %7.2f\t", sqrt(sum_err_sqr));
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				chtotal += ((char *)A[T % 2][i][j])[k];
			}
		}
	}
	fprintf(BENCHMARK_DUMP_FILE, "sum(rep(A)) = %d\n", chtotal);
#endif
#ifdef BENCHMARK_DUMP_ARRAYS
	BENCHMARK_DUMP_BEGIN("A");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				fprintf(BENCHMARK_DUMP_FILE, "%e ", A[T % 2][i][j][k]);
			}
			fprintf(BENCHMARK_DUMP_FILE, "\n");
		}
		fprintf(BENCHMARK_DUMP_FILE, "\n");
	}
	BENCHMARK_DUMP_END("A");
#endif /* BENCHMARK_DUMP_ARRAYS */
	BENCHMARK_DUMP_STOP();
}

void kernel_heat_3d(param_t tsteps, param_t n,
					data_t BENCHMARK_4D(A, 2, N, N, N, 2, n, n, n)) {
	iter_t t, i, j, k;
#pragma scop
	for (t = 0; t < T; t++) {
		for (i = 1; i < N - 1; i++) {
			for (j = 1; j < N - 1; j++) {
				for (k = 1; k < N - 1; k++) {
					A[(t + 1) % 2][i][j][k] = 0.125 * (A[t % 2][i + 1][j][k] -
													   2.0 * A[t % 2][i][j][k] +
													   A[t % 2][i - 1][j][k]) +
											  0.125 * (A[t % 2][i][j + 1][k] -
													   2.0 * A[t % 2][i][j][k] +
													   A[t % 2][i][j - 1][k]) +
											  0.125 * (A[t % 2][i][j][k - 1] -
													   2.0 * A[t % 2][i][j][k] +
													   A[t % 2][i][j][k + 1]) +
											  A[t % 2][i][j][k];
				}
			}
		}
	}
#pragma endscop
}

int main(int argc, char *argv[]) {
	param_t t = T;
	param_t n = N;

	BENCHMARK_4D_ARRAY_DECL(A, data_t, 2, N, N, N, 2, n, n, n);

	/* initialization */
	init_array(t, n, A);

	/* start timer */
	benchmark_measure_start();

	/* kernel execution */
	kernel_heat_3d(t, n, A);

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