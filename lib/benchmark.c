#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <benchmark.h>
#include <benchmark/time.h>

/* Cache size. By default 32+MB.. */
#ifndef BENCHMARK_CACHE_SIZE_KB
#define BENCHMARK_CACHE_SIZE_KB 128770
#endif

/**
 * Allocations utilities
 */
void *benchmark_alloc_data(size_t n, size_t size) {
	void *ptr = nullptr;
	size_t alloc_sz = n * size;
	int err = posix_memalign(&ptr, 4096, alloc_sz);
	if (err != 0) {
		fprintf(stderr, "Error allocating memory: %d\n", err);
	}
	return ptr;
}

/**
 * Timer utilities
 */
void benchmark_measure_start(bmeasure_t *data) {
	memset(data, 0, sizeof(bmeasure_t));
#ifndef BENCHMARK_CACHE_NOFLUSH
	_benchmark_cache_flush();
#endif
	gettimeofday(&data->start, NULL);
}

void benchmark_measure_stop(bmeasure_t *data) { gettimeofday(&data->end, NULL); }

static void _benchmark_timedif(struct timeval *result, struct timeval *x, struct timeval *y) {
	/* Perform the carry for the later subtraction by updating y. */
	if (x->tv_usec < y->tv_usec) {
		int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;

		y->tv_usec -= 1000000 * nsec;
		y->tv_sec += nsec;
	}

	if (x->tv_usec - y->tv_usec > 1000000) {
		int nsec = (x->tv_usec - y->tv_usec) / 1000000;

		y->tv_usec += 1000000 * nsec;
		y->tv_sec -= nsec;
	}

	/* Compute the time remaining to wait.
	 * tv_usec is certainly positive.
	 */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;
}

void _benchmark_timer_print(bmeasure_t *data) {
	_benchmark_timedif(&data->result, &data->end, &data->start);
	double time = data->result.tv_sec + data->result.tv_usec / 1e6;

	fprintf(BENCHMARK_TIME_TARGET_FILE, "%f\n", time);
}
void _benchmark_timer_print_mflops(bmeasure_t *data, double num_ops) {
	_benchmark_timedif(&data->result, &data->end, &data->start);
	double time = data->result.tv_sec + data->result.tv_usec / 1e6;
	double mflops = num_ops / (time) / 1e6;
	fprintf(BENCHMARK_TIME_TARGET_FILE, "%f\n", mflops);
}

void benchmark_measure_print(bmeasure_t *data) {
#ifdef BENCHMARK_TIME_MFLOPS
	_benchmark_timer_print_mflops(data, BENCHMARK_NUM_FP_OPS);
#elif defined(BENCHMARK_TIME)
	_benchmark_timer_print(data);
#endif
}

/*
	Purge the cache by writing to a large array.
*/
void _benchmark_cache_flush() {
	size_t cs = BENCHMARK_CACHE_SIZE_KB * 1024 / sizeof(double);
	double *flush = calloc(cs, sizeof(double));
	if (flush == NULL) {
		fprintf(stderr, "Error allocating memory for cache flush\n");
		return;
	}
	srand(time(NULL));

	// 3. Purge the cache using OpenMP
	// We parallelize the writing process so that all cores fill their own caches
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < BENCHMARK_CACHE_SIZE_KB; i += 1) {
		// Write a random integer to each cache line
		// This ensures that the CPU actually performs a store operation
		// and populates the cache with "noise" data
		float random_value = (float)rand() / RAND_MAX; // Generate a random float between 0 and 1
		flush[i] = random_value;
	}

	__sync_synchronize();

	free(flush);
}