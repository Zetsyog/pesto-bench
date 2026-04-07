/*
 * Order-1, 3D 7 point stencil
 * Adapted from Pochoir test bench
 *
 * Irshad Pananilath: irshad@csa.iisc.ernet.in
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef N
#define N 256L
#endif /* N */

#ifndef T
#define T 200L
#endif /* T */

#define FP_OPS_PER_ITERATION 8
#define BENCHMARK_NUM_FP_OPS FP_OPS_PER_ITERATION *N *N *N *(T - 1)

#include <benchmark.h>

void init_array(param_t n, data_t *alpha, data_t *beta,
				data_t BENCHMARK_4D(A, 2, N, N, N, 2, n, n, n)) {
	int i, j, k;

	*alpha = 0.0876;
	*beta = 0.0765;

	srand(BENCHMARK_RSEED);
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				A[0][i][j][k] = 1.0 * (rand() % (BENCHMARK_INIT_BASE));
			}
		}
	}
}

void dump_array(param_t tsteps, param_t n, data_t alpha, data_t beta,
				data_t BENCHMARK_4D(A, 2, N, N, N, 2, n, n, n)) {
	BENCHMARK_DUMP_START();
#ifdef BENCHMARK_DUMP_CHKSUM
	BENCHMARK_DUMP_BEGIN("parameters");
	fprintf(BENCHMARK_DUMP_FILE, "T=" PARAM_PRINTF_MODIFIER "\n", tsteps);
	fprintf(BENCHMARK_DUMP_FILE, "N=" PARAM_PRINTF_MODIFIER "\n", n);
	fprintf(BENCHMARK_DUMP_FILE, "alpha: %e\n", alpha);
	fprintf(BENCHMARK_DUMP_FILE, "beta: %e\n", beta);
	BENCHMARK_DUMP_END("parameters");

	BENCHMARK_DUMP_BEGIN("checksum");
	// compute checksum
	data_t sum = 0.0;
	for (iter_t i = 0; i < N; ++i) {
		for (iter_t j = 0; j < N; ++j) {
			for (iter_t k = 0; k < N; ++k) {
				sum += A[T % 2][i][j][k];
			}
		}
	}
	fprintf(BENCHMARK_DUMP_FILE, "checksum: %e\n", sum);
	BENCHMARK_DUMP_END("checksum");
#endif
#ifdef BENCHMARK_DUMP_ARRAYS
	BENCHMARK_DUMP_BEGIN("A");
	for (iter_t i = 0; i < N; i++) {
		for (iter_t j = 0; j < N; j++) {
			for (iter_t k = 0; k < N; k++) {
				fprintf(BENCHMARK_DUMP_FILE, " " DATA_PRINTF_MODIFIER,
						A[T % 2][i][j][k]);
			}
			fprintf(BENCHMARK_DUMP_FILE, "\n");
		}
		fprintf(BENCHMARK_DUMP_FILE, "\n");
	}
	BENCHMARK_DUMP_END("A");
#endif /* BENCHMARK_DUMP_ARRAYS */
	BENCHMARK_DUMP_STOP();
}

void kernel_3d7pt(param_t tstep, param_t n, data_t alpha, data_t beta,
				  data_t BENCHMARK_4D(A, 2, N, N, N, 2, n, n, n)) {
	iter_t t, i, j, k;
#pragma scop
	for (t = 0; t < T - 1; t++) {
		for (i = 1; i < N - 1; i++) {
			for (j = 1; j < N - 1; j++) {
				for (k = 1; k < N - 1; k++) {
					A[(t + 1) % 2][i][j][k] =
						alpha * (A[t % 2][i][j][k]) +
						beta * (A[t % 2][i - 1][j][k] + A[t % 2][i][j - 1][k] +
								A[t % 2][i][j][k - 1] + A[t % 2][i + 1][j][k] +
								A[t % 2][i][j + 1][k] + A[t % 2][i][j][k + 1]);
				}
			}
		}
	}
#pragma endscop
}

int main(int argc, char *argv[]) {
	// for timekeeping
	param_t tsteps = T;
	param_t n = N;

	data_t alpha;
	data_t beta;

	BENCHMARK_4D_ARRAY_DECL(A, data_t, 2, N, N, N, 2, n, n, n);

	/* initialization */
	init_array(n, &alpha, &beta, A);

	/* start timer */
	benchmark_measure_start();

	/* serial execution - Addition: 6 && Multiplication: 2 */
	kernel_3d7pt(T, n, alpha, beta, A);

	/* stop timer */
	benchmark_measure_stop();

	/* print time */
	benchmark_measure_print();

#ifdef BENCHMARK_DUMP
	dump_array(T, N, alpha, beta, A);
#endif /* BENCHMARK_DUMP */

	BENCHMARK_FREE_ARRAY(A);

	return 0;
}
