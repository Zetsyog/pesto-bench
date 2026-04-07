#ifndef BENCHMARK_TYPES_H

#if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) &&              \
	!defined(DATA_TYPE_IS_DOUBLE)
#define DATA_TYPE_IS_DOUBLE
#endif /* Default data type */

#ifndef param_t
#define param_t unsigned long
#define PARAM_PRINTF_MODIFIER "%lu"
#endif /* Default parameter type */

#ifdef DATA_TYPE_IS_DOUBLE
#define data_t double
#define iter_t size_t
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define SCALAR_VAL(x) x
#define SQRT_FUN(x) sqrt(x)
#define EXP_FUN(x) exp(x)
#define POW_FUN(x, y) pow(x, y)
#endif /* DATA_TYPE_IS_DOUBLE */

#ifdef DATA_TYPE_IS_INT
#define data_t int
#define iter_t size_t
#define DATA_PRINTF_MODIFIER "%d "
#define SCALAR_VAL(x) x
#define SQRT_FUN(x) sqrt((double)x)
#define EXP_FUN(x) exp((double)x)
#define POW_FUN(x, y) pow((double)x, (double)y)
#endif /* DATA_TYPE_IS_INT */

#endif /* BENCHMARK_TYPES_H */