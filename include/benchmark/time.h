#ifndef BENCHMARK_TIME_H
#define BENCHMARK_TIME_H

#include <sys/time.h>

#ifndef BENCHMARK_TIME_TARGET_FILE
#define BENCHMARK_TIME_TARGET_FILE stdout
#endif

#if defined(BENCHMARK_TIME_MFLOPS)
#if !defined(BENCHMARK_NUM_FP_OPS)
#error "BENCHMARK_NUM_FP_OPS must be defined when BENCHMARK_TIME_MFLOPS is defined"
#endif /* BENCHMARK_NUM_FP_OPS */

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME
#endif /* BENCHMARK_TIME */
#endif /* BENCHMARK_TIME_MFLOPS */

struct bmeasure_s {
	struct timeval start;
	struct timeval end;
	struct timeval result;
};
typedef struct bmeasure_s bmeasure_t;

void benchmark_measure_start(bmeasure_t *data);
void benchmark_measure_stop(bmeasure_t *data);
void benchmark_measure_print(bmeasure_t *data);

#endif /* BENCHMARK_TIME_H */