#ifndef BENCHMARK_TYPES_H

#if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) &&              \
	!defined(DATA_TYPE_IS_DOUBLE)
#define DATA_TYPE_IS_DOUBLE
#endif /* Default data type */

#ifndef param_t
#define param_t unsigned long
#endif /* Default parameter type */

#ifdef DATA_TYPE_IS_DOUBLE
#define data_t double
#define iter_t size_t
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define SCALAR_VAL(x) x
#define SQRT_FUN(x) sqrt(x)
#define EXP_FUN(x) exp(x)
#define POW_FUN(x, y) pow(x, y)
#endif

#endif /* BENCHMARK_TYPES_H */