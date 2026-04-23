#ifndef BENCHMARK_H
#define BENCHMARK_H

/* C2x/C23 or later? */
#if (defined(__STDC__) && defined(__STDC_VERSION__) &&                         \
	 (__STDC_VERSION__ >= 202300L))
#include <stddef.h> /* nullptr_t */
/* pre C23, pre C++11 or non-standard */
#else
#ifndef nullptr
#define nullptr NULL
#endif // nullptr
#endif // __STDC_VERSION__

#include <benchmark/arrays.h>
#include <benchmark/dump.h>
#include <benchmark/time.h>
#include <benchmark/types.h>

#ifndef BENCHMARK_RSEED
#define BENCHMARK_RSEED 42
#endif

#ifndef BENCHMARK_INIT_BASE
#define BENCHMARK_INIT_BASE 1024
#endif

#endif /* BENCHMARK_H */