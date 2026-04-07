#ifndef BENCHMARK_ARRAYS_H
#define BENCHMARK_ARRAYS_H

#include <stdlib.h>

#ifdef BENCHMARK_USE_RESTRICT
#define BENCHMARK_RESTRICT restrict
#else
#define BENCHMARK_RESTRICT
#endif

#define BENCHMARK_DECL_VAR(x) (*x)

#define BENCHMARK_ARRAY_SIZE_SELECT(macroDim, varDIM) (macroDim)

#ifdef __STDC_NO_VLA__
#error                                                                         \
	"VLAs are not supported by the compiler. Please compile with a C99-compliant compiler that supports VLAs."

#endif

#ifdef BENCHMARK_USE_SCALAR_LB
#define BENCHMARK_LOOP_BOUND(x, y) x
#else
#define BENCHMARK_LOOP_BOUND(x, y) y
#endif

/*
	Macro for referencing arrays in function prototypes.
	We use pointers to VLAs to allow for dynamic allocation of arrays on the
	heap, while still allowing for the use of C99 array syntax in the function
	body.
 */
#define BENCHMARK_1D(var, macroDim1, varDim1) (*BENCHMARK_RESTRICT var)

#define BENCHMARK_2D(var, macroDim1, macroDim2, varDim1, varDim2)              \
	(*BENCHMARK_RESTRICT var)[BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2)]

#define BENCHMARK_3D(var, macroDim1, macroDim2, macroDim3, varDim1, varDim2,   \
					 varDim3)                                                  \
	(*BENCHMARK_RESTRICT var)[BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2)] \
							 [BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3)]

#define BENCHMARK_4D(var, macroDim1, macroDim2, macroDim3, macroDim4, varDim1, \
					 varDim2, varDim3, varDim4)                                \
	(*BENCHMARK_RESTRICT var)[BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2)] \
							 [BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3)] \
							 [BENCHMARK_ARRAY_SIZE_SELECT(macroDim4, varDim4)]

#define BENCHMARK_5D(var, macroDim1, macroDim2, macroDim3, macroDim4,          \
					 macroDim5, varDim1, varDim2, varDim3, varDim4, varDim5)   \
	(*BENCHMARK_RESTRICT var)[BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2)] \
							 [BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3)] \
							 [BENCHMARK_ARRAY_SIZE_SELECT(macroDim4, varDim4)] \
							 [BENCHMARK_ARRAY_SIZE_SELECT(macroDim5, varDim5)]

/*
	Macros for using arrays within functions
*/
#define BENCHMARK_1D_F(var, macroDim1, varDim1) (*var)

#define BENCHMARK_2D_F(var, macroDim1, macroDim2, varDim1, varDim2)            \
	(*var)[BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2)]

#define BENCHMARK_3D_F(var, macroDim1, macroDim2, macroDim3, varDim1, varDim2, \
					   varDim3)                                                \
	(*var)[BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2)]                    \
		  [BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3)]

#define BENCHMARK_4D_F(var, macroDim1, macroDim2, macroDim3, macroDim4,        \
					   varDim1, varDim2, varDim3, varDim4)                     \
	(*var)[BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2)]                    \
		  [BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3)]                    \
		  [BENCHMARK_ARRAY_SIZE_SELECT(macroDim4, varDim4)]

#define BENCHMARK_5D_F(var, macroDim1, macroDim2, macroDim3, macroDim4,        \
					   macroDim5, varDim1, varDim2, varDim3, varDim4, varDim5) \
	(*var)[BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2)]                    \
		  [BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3)]                    \
		  [BENCHMARK_ARRAY_SIZE_SELECT(macroDim4, varDim4)]                    \
		  [BENCHMARK_ARRAY_SIZE_SELECT(macroDim5, varDim5)]

/*
	Macros to allocate heap arrays
*/
#define BENCHMARK_ALLOC_1D_ARRAY(type, n1)                                     \
	benchmark_alloc_data((n1), sizeof(type))

#define BENCHMARK_ALLOC_2D_ARRAY(type, n1, n2)                                 \
	benchmark_alloc_data((n1) * (n2), sizeof(type))

#define BENCHMARK_ALLOC_3D_ARRAY(type, n1, n2, n3)                             \
	benchmark_alloc_data((n1) * (n2) * (n3), sizeof(type))

#define BENCHMARK_ALLOC_4D_ARRAY(type, n1, n2, n3, n4)                         \
	benchmark_alloc_data((n1) * (n2) * (n3) * (n4), sizeof(type))

#define BENCHMARK_ALLOC_5D_ARRAY(type, n1, n2, n3, n4, n5)                     \
	benchmark_alloc_data((n1) * (n2) * (n3) * (n4) * (n5), sizeof(type))

/*                                                                             \
	Macros for array declaration. If BENCHMARK_ARRAYS_STACK is defined, arrays \
	are declared on the stack. Otherwise, they are declared as pointers and    \
	allocated on the heap.                                                     \
 */

#ifndef BENCHMARK_ARRAYS_STACK // heap allocated arrays
#define BENCHMARK_1D_ARRAY_DECL(var, type, macroDim1, varDim1)                 \
	type BENCHMARK_1D_F(var, macroDim1, varDim1);                              \
	var = BENCHMARK_ALLOC_1D_ARRAY(                                            \
		type, BENCHMARK_ARRAY_SIZE_SELECT(macroDim1, varDim1))

#define BENCHMARK_2D_ARRAY_DECL(var, type, macroDim1, macroDim2, varDim1,      \
								varDim2)                                       \
	type BENCHMARK_2D_F(var, macroDim1, macroDim2, varDim1, varDim2);          \
	var = BENCHMARK_ALLOC_2D_ARRAY(                                            \
		type, BENCHMARK_ARRAY_SIZE_SELECT(macroDim1, varDim1),                 \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2))

#define BENCHMARK_3D_ARRAY_DECL(var, type, macroDim1, macroDim2, macroDim3,    \
								varDim1, varDim2, varDim3)                     \
	type BENCHMARK_3D_F(var, macroDim1, macroDim2, macroDim3, varDim1,         \
						varDim2, varDim3);                                     \
	var = BENCHMARK_ALLOC_3D_ARRAY(                                            \
		type, BENCHMARK_ARRAY_SIZE_SELECT(macroDim1, varDim1),                 \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2),                       \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3))

#define BENCHMARK_4D_ARRAY_DECL(var, type, macroDim1, macroDim2, macroDim3,    \
								macroDim4, varDim1, varDim2, varDim3, varDim4) \
	type BENCHMARK_4D_F(var, macroDim1, macroDim2, macroDim3, macroDim4,       \
						varDim1, varDim2, varDim3, varDim4);                   \
	var = BENCHMARK_ALLOC_4D_ARRAY(                                            \
		type, BENCHMARK_ARRAY_SIZE_SELECT(macroDim1, varDim1),                 \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2),                       \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3),                       \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim4, varDim4))

#define BENCHMARK_5D_ARRAY_DECL(var, type, macroDim1, macroDim2, macroDim3,    \
								macroDim4, macroDim5, varDim1, varDim2,        \
								varDim3, varDim4, varDim5)                     \
	type BENCHMARK_5D_F(var, macroDim1, macroDim2, macroDim3, macroDim4,       \
						macroDim5, varDim1, varDim2, varDim3, varDim4,         \
						varDim5);                                              \
	var = BENCHMARK_ALLOC_5D_ARRAY(                                            \
		type, BENCHMARK_ARRAY_SIZE_SELECT(macroDim1, varDim1),                 \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim2, varDim2),                       \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim3, varDim3),                       \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim4, varDim4),                       \
		BENCHMARK_ARRAY_SIZE_SELECT(macroDim5, varDim5))
#else
#error                                                                         \
	"Stack arrays are not currently supported. Please undefine BENCHMARK_ARRAYS_STACK to use heap arrays."
#endif /* BENCHMARK_STACK_ARRAYS */

/*
	Macro to free arrays
*/
#ifndef BENCHMARK_ARRAYS_STACK
#define BENCHMARK_FREE_ARRAY(x) free(x);
#else
#define BENCHMARK_ARRAY_FREE(x)
#endif /* BENCHMARK_ARRAYS_STACK */

void *benchmark_alloc_data(size_t n, size_t size);

#endif /* BENCHMARK_ARRAYS_H */