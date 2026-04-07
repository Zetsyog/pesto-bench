/*
 * John Conway's Game of Life - 2D, Periodic - B3/S23
 * Adapted from Pochoir test bench
 *
 * Irshad Pananilath: irshad@csa.iisc.ernet.in
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/*
 * N is the length of one side of the GoL world
 * T is the number of generations to evolve
 */
#ifndef N
#define N 2000L
#endif /* N */
#ifndef T
#define T 500L
#endif /* T */

#define DATA_TYPE_IS_INT

#define NUM_FP_OPS 8
#define BENCHMARK_NUM_FP_OPS (NUM_FP_OPS) * (N - 2) * (N - 2) * (T)

#include <benchmark.h>

#if !defined(PRESET_RANDOM) && !defined(PRESET_TOAD) && !defined(PRESET_PULSAR)
#define PRESET_RANDOM
#endif /* PRESET_RANDOM */

#if defined(PRESET_RANDOM)
#define PRESET_RANDOM_DEFINED 1
#else
#define PRESET_RANDOM_DEFINED 0
#endif

#if defined(PRESET_TOAD)
#define PRESET_TOAD_DEFINED 1
#else
#define PRESET_TOAD_DEFINED 0
#endif

#if defined(PRESET_PULSAR)
#define PRESET_PULSAR_DEFINED 1
#else
#define PRESET_PULSAR_DEFINED 0
#endif

#if (PRESET_RANDOM_DEFINED + PRESET_TOAD_DEFINED + PRESET_PULSAR_DEFINED) != 1
#error "Define exactly one of PRESET_RANDOM, PRESET_TOAD, or PRESET_PULSAR."
#endif

/*
 * Calculate the status of a cell in the next evolution given
 * the neighbors it has
 */
int b2s23(int cell, int neighbors) {
	if ((cell == 1 && ((neighbors < 2) || (neighbors > 3)))) {
		return 0;
	}

	if ((cell == 1 && (neighbors == 2 || neighbors == 3))) {
		return 1;
	}

	if ((cell == 0 && neighbors == 3)) {
		return 1;
	}

	return cell;
}

/*
 * Print the final array
 */
void print_points(param_t tsteps, param_t n,
				  data_t BENCHMARK_3D(life, 2, N, N, 2, n, n)) {
	int a, b;

	for (a = 0; a < N; a++) {
		for (b = 0; b < N; b++) {
			fprintf(BENCHMARK_DUMP_FILE, "%c ",
					(life[tsteps % 2][a][b] ? '1' : '.'));
		}
		fprintf(BENCHMARK_DUMP_FILE, "\n");
	}
	fprintf(BENCHMARK_DUMP_FILE, "\n");
}

void init_array(param_t tsteps, param_t n,
				data_t BENCHMARK_3D(life, 2, N, N, 2, n, n)) {
	iter_t t, i, j;
	srand(BENCHMARK_RSEED); // seed with a constant value to verify results

#if defined(PRESET_RANDOM)
	/* Random initialization */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			life[0][i][j] = (rand() & 0x1) ? 1 : 0;
			life[1][i][j] = 0;
		}
	}
#else
	/* Preset patterns */

	/* Initialize all cells to 0 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			life[0][i][j] = 0;
			life[1][i][j] = 0;
		}
	}

#if defined(PRESET_TOAD)
	/* Toad - Oscillator - Period 2 */
	life[0][2][2] = 1;
	life[0][2][3] = 1;
	life[0][2][4] = 1;
	life[0][3][1] = 1;
	life[0][3][2] = 1;
	life[0][3][3] = 1;
#elif defined(PRESET_PULSAR)
	/* pulsar - oscillator - period 3 */
	life[0][2][4] = 1;
	life[0][2][5] = 1;
	life[0][2][6] = 1;
	life[0][2][10] = 1;
	life[0][2][11] = 1;
	life[0][2][12] = 1;
	life[0][4][2] = 1;
	life[0][4][7] = 1;
	life[0][4][9] = 1;
	life[0][4][14] = 1;
	life[0][5][2] = 1;
	life[0][5][7] = 1;
	life[0][5][9] = 1;
	life[0][5][14] = 1;
	life[0][6][2] = 1;
	life[0][6][7] = 1;
	life[0][6][9] = 1;
	life[0][6][14] = 1;
	life[0][7][4] = 1;
	life[0][7][5] = 1;
	life[0][7][6] = 1;
	life[0][7][10] = 1;
	life[0][7][11] = 1;
	life[0][7][12] = 1;
	life[0][9][4] = 1;
	life[0][9][5] = 1;
	life[0][9][6] = 1;
	life[0][9][10] = 1;
	life[0][9][11] = 1;
	life[0][9][12] = 1;
	life[0][10][2] = 1;
	life[0][10][7] = 1;
	life[0][10][9] = 1;
	life[0][10][14] = 1;
	life[0][11][2] = 1;
	life[0][11][7] = 1;
	life[0][11][9] = 1;
	life[0][11][14] = 1;
	life[0][12][2] = 1;
	life[0][12][7] = 1;
	life[0][12][9] = 1;
	life[0][12][14] = 1;
	life[0][14][4] = 1;
	life[0][14][5] = 1;
	life[0][14][6] = 1;
	life[0][14][10] = 1;
	life[0][14][11] = 1;
	life[0][14][12] = 1;
#endif /* PRESET_PULSAR */
#endif /* PRESET_RANDOM */
}

void dump_array(param_t tsteps, param_t n,
				data_t BENCHMARK_3D(life, 2, N, N, 2, n, n)) {
	iter_t t, i, j;
	BENCHMARK_DUMP_START();
#ifdef BENCHMARK_DUMP_CHKSUM
	BENCHMARK_DUMP_BEGIN("chksum");
	fprintf(BENCHMARK_DUMP_FILE, "T=" PARAM_PRINTF_MODIFIER "\n", tsteps);
	fprintf(BENCHMARK_DUMP_FILE, "N=" PARAM_PRINTF_MODIFIER "\n", n);
	BENCHMARK_DUMP_END("chksum");
#endif /* BENCHMARK_DUMP_CHKSUM */
#ifdef BENCHMARK_DUMP_ARRAYS
	BENCHMARK_DUMP_BEGIN("life");
	print_points(tsteps, n, life);
	BENCHMARK_DUMP_END("life");
#endif /* BENCHMARK_DUMP_ARRAYS */

	BENCHMARK_DUMP_STOP();
}

void kernel_life(param_t tsteps, param_t n,
				 data_t BENCHMARK_3D(life, 2, N, N, 2, n, n)) {
	iter_t t, i, j;
#pragma scop
	for (t = 0; t < T; t++) {
		for (i = 1; i < N - 1; i++) {
			for (j = 1; j < N - 1; j++) {
				life[(t + 1) % 2][i][j] = b2s23(
					life[t % 2][i][j],
					life[t % 2][i - 1][j + 1] + life[t % 2][i - 1][j] +
						life[t % 2][i - 1][j - 1] + life[t % 2][i][j + 1] +
						life[t % 2][i][j - 1] + life[t % 2][i + 1][j + 1] +
						life[t % 2][i + 1][j] + life[t % 2][i + 1][j - 1]);
			}
		}
	}
#pragma endscop
}
int main(int argc, char *argv[]) {
	param_t tsteps = T;
	param_t n = N;

	BENCHMARK_3D_ARRAY_DECL(life, data_t, 2, N, N, 2, n, n);

	/* initialization */
	init_array(T, N, life);

#ifdef BENCHMARK_DUMP_ARRAYS
	/* dump initial array */
	print_points(0, N, life);
#endif /* BENCHMARK_DUMP_ARRAYS */

	/* start timer */
	benchmark_measure_start();

	/* execute the main kernel */
	kernel_life(T, N, life);

	/* stop timer */
	benchmark_measure_stop();

	/* compute and print statistics */
	benchmark_measure_print();

#ifdef BENCHMARK_DUMP
	dump_array(T, N, life);
#endif

	BENCHMARK_FREE_ARRAY(life);

	return 0;
}
