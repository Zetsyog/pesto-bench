#ifndef BENCHMARK_DUMP_H
#define BENCHMARK_DUMP_H

/*
	If BENCHMARK_DUMP is defined but neither BENCHMARK_DUMP_CHKSUM nor
   BENCHMARK_DUMP_ARRAYS is defined, enable both by default.
*/
#ifdef BENCHMARK_DUMP
#if !defined(BENCHMARK_DUMP_CHKSUM) && !defined(BENCHMARK_DUMP_ARRAYS)
#define BENCHMARK_DUMP_CHKSUM
#define BENCHMARK_DUMP_ARRAYS
#endif /* BENCHMARK_DUMP_CHKSUM || BENCHMARK_DUMP_ARRAYS */
#endif /* BENCHMARK_DUMP */

/*
	If either BENCHMARK_DUMP_CHKSUM or BENCHMARK_DUMP_ARRAYS is defined, enable
   BENCHMARK_DUMP.
*/
#if defined(BENCHMARK_DUMP_CHKSUM) || defined(BENCHMARK_DUMP_ARRAYS)
#define BENCHMARK_DUMP
#endif /* BENCHMARK_DUMP_CHKSUM || BENCHMARK_DUMP_ARRAYS */

#ifndef BENCHMARK_DUMP_FILE
#define BENCHMARK_DUMP_FILE stderr
#endif

#define BENCHMARK_DUMP_START()                                                 \
	do {                                                                       \
		fprintf(BENCHMARK_DUMP_FILE, "==BEGIN DUMP==\n");                      \
	} while (0)

#define BENCHMARK_DUMP_STOP()                                                  \
	do {                                                                       \
		fprintf(BENCHMARK_DUMP_FILE, "== END DUMP ==\n");                      \
	} while (0)

#define BENCHMARK_DUMP_BEGIN(s)                                                \
	do {                                                                       \
		fprintf(BENCHMARK_DUMP_FILE, "begin dump: %s\n", s);                   \
	} while (0)

#define BENCHMARK_DUMP_END(s)                                                  \
	do {                                                                       \
		fprintf(BENCHMARK_DUMP_FILE, "\nend   dump: %s\n", s);                 \
	} while (0)

#endif /* BENCHMARK_DUMP_H */