#ifndef BENCHMARK_TIME_H
#define BENCHMARK_TIME_H

#ifndef BENCHMARK_TIME_TARGET_FILE
#define BENCHMARK_TIME_TARGET_FILE stdout
#endif

#if defined(BENCHMARK_TIME_MFLOPS)
#if !defined(BENCHMARK_NUM_FP_OPS)
#error                                                                         \
	"BENCHMARK_NUM_FP_OPS must be defined when BENCHMARK_TIME_MFLOPS is defined"
#endif /* BENCHMARK_NUM_FP_OPS */

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME
#endif /* BENCHMARK_TIME */
#endif /* BENCHMARK_TIME_MFLOPS */

#ifdef BENCHMARK_TIME

#define benchmark_measure_start()                                              \
	do {                                                                       \
		_benchmark_timer_start();                                              \
	} while (0)

#define benchmark_measure_stop()                                               \
	do {                                                                       \
		_benchmark_timer_stop();                                               \
	} while (0)

#ifdef BENCHMARK_TIME_MFLOPS
#define benchmark_measure_print()                                              \
	do {                                                                       \
		_benchmark_timer_print_mflops(BENCHMARK_NUM_FP_OPS);                   \
		_benchmark_timer_print();                                              \
	} while (0)
#else
#define benchmark_measure_print()                                              \
	do {                                                                       \
		_benchmark_timer_print();                                              \
	} while (0)
#endif /* BENCHMARK_TIME_MFLOPS */

void _benchmark_timer_start();
void _benchmark_timer_stop();
void _benchmark_timer_print();
void _benchmark_timer_print_mflops(double num_ops);
#else

#define benchmark_measure_start()
#define benchmark_measure_stop()
#define benchmark_measure_print()

#endif

#endif /* BENCHMARK_TIME_H */