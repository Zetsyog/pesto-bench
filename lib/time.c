#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <benchmark/time.h>

static struct timeval start, end, result;

void _benchmark_timer_start() { gettimeofday(&start, NULL); }

void _benchmark_timer_stop() { gettimeofday(&end, NULL); }

static void _benchmark_timedif(struct timeval *result, struct timeval *x,
							   struct timeval *y) {
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

void _benchmark_timer_print() {
	_benchmark_timedif(&result, &end, &start);
	double time = result.tv_sec + result.tv_usec / 1e6;

	fprintf(BENCHMARK_TIME_TARGET_FILE, "%f\n", time);
}
void _benchmark_timer_print_mflops(double num_ops) {
	_benchmark_timedif(&result, &end, &start);
	double time = result.tv_sec + result.tv_usec / 1e6;
	double mflops = num_ops / (time) / 1e6;
	fprintf(BENCHMARK_TIME_TARGET_FILE, "%f\n", mflops);
}