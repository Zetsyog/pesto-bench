#ifndef _FDTD_2D_H
#define _FDTD_2D_H

/* Default to LARGE_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) &&        \
	!defined(EXTRALARGE_DATASET)
#define LARGE_DATASET
#endif

#if !defined(TMAX) && !defined(NX) && !defined(NY)
/* Define sample dataset sizes. */
#ifdef MINI_DATASET
#define TMAX 20
#define NX 20
#define NY 30
#endif

#ifdef SMALL_DATASET
#define TMAX 40
#define NX 60
#define NY 80
#endif

#ifdef MEDIUM_DATASET
#define TMAX 100
#define NX 200
#define NY 240
#endif

#ifdef LARGE_DATASET
#define TMAX 500
#define NX 1000
#define NY 1200
#endif

#ifdef EXTRALARGE_DATASET
#define TMAX 1000
#define NX 2000
#define NY 2600
#endif

#ifdef XL2_DATASET
#define TMAX 2000
#define NX 4000
#define NY 5200
#endif

#ifdef XL3_DATASET
#define TMAX 3000
#define NX 6000
#define NY 7800
#endif

#ifdef XL4_DATASET
#define TMAX 4000
#define NX 8000
#define NY 10400
#endif

#ifdef XL5_DATASET
#define TMAX 5000
#define NX 10000
#define NY 13000
#endif

#ifdef XL6_DATASET
#define TMAX 6000
#define NX 12000
#define NY 15600
#endif

#endif /* !(TMAX NX NY) */

#define _PB_TMAX POLYBENCH_LOOP_BOUND(TMAX, tmax)
#define _PB_NX POLYBENCH_LOOP_BOUND(NX, nx)
#define _PB_NY POLYBENCH_LOOP_BOUND(NY, ny)

/* Default data type */
#if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#define DATA_TYPE_IS_DOUBLE
#endif

#ifdef DATA_TYPE_IS_INT
#define DATA_TYPE int
#define DATA_PRINTF_MODIFIER "%d "
#endif

#ifdef DATA_TYPE_IS_FLOAT
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2f "
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x, y) powf(x, y)
#endif

#ifdef DATA_TYPE_IS_DOUBLE
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define SCALAR_VAL(x) x
#define SQRT_FUN(x) sqrt(x)
#define EXP_FUN(x) exp(x)
#define POW_FUN(x, y) pow(x, y)
#endif

#endif /* !_FDTD_2D_H */
