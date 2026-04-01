
#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    (((x) > (y)) ? (x) : (y))
#define min(x,y)    (((x) < (y)) ? (x) : (y))
#ifndef DIV0
#define DIV0 8
#endif
#ifndef DIV1
#define DIV1 8
#endif
#ifndef DIV2
#define DIV2 8
#endif
#ifndef DIV3
#define DIV3 8
#endif
#include "jacobi-1d.trahrhe.n0.g0.h"
#include "jacobi-1d.trahrhe.schedule.n0.h"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-1d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
      {
	A[i] = ((DATA_TYPE) i+ 2) / n;
	B[i] = ((DATA_TYPE) i+ 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_1d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;



long int lbp;
long int ubp;
long int lbv;
long int ubv;
long int t1;
long int t2;
long int t3;
long int t4;
long int t5;
long int g0_t3_pcmax;
long int g0_TARGET_VOL_L0;
long int g0_ubt1;
long int g0_lbt3;
long int g0_ubt3;
long int g0_t4_pcmax;
long int g0_TARGET_VOL_L1;
long int g0_ubt2;
long int g0_lbt4;
long int g0_ubt4;
if (_PB_N >= 3) {
  SCHEDULE_ALLOC(n0_g0_tiles, n0_g0_schedule);SCHEDULE_COMPUTE(n0_g0, n0_g0_tiles, n0_g0_schedule, _PB_TSTEPS, _PB_N);g0_ubt1 = n0_g0_schedule.len - 1;;
  for (t1=0;t1<=g0_ubt1;t1++) {
        ;
    g0_ubt2 = n0_g0_schedule.fronts[t1].len - 1;;
    lbp=0;
    ubp=g0_ubt2;
#pragma omp parallel for firstprivate(g0_ubt1, g0_t3_pcmax, g0_TARGET_VOL_L0) private(t3,t4,t5, g0_lbt4, g0_ubt4, g0_lbt3, g0_ubt3)
    for (t2=lbp;t2<=ubp;t2++) {
      g0_lbt3 = n0_g0_schedule.fronts[t1].tiles[t2]->lb0;g0_ubt3 = n0_g0_schedule.fronts[t1].tiles[t2]->ub0;g0_lbt4 = n0_g0_schedule.fronts[t1].tiles[t2]->lb1;g0_ubt4 = n0_g0_schedule.fronts[t1].tiles[t2]->ub1;;
      for (t3=g0_lbt3;t3<=min(_PB_TSTEPS-1,g0_ubt3);t3++) {
        for (t4=max(2*t3+1,g0_lbt4);t4<=min(2*t3+_PB_N-1,g0_ubt4);t4++) {
          if (t3 >= ceild(t4-_PB_N+2,2)) {
            B[(-2*t3+t4)] = 0.33333 * (A[(-2*t3+t4)-1] + A[(-2*t3+t4)] + A[(-2*t3+t4) + 1]);;
          }
          if (t3 <= floord(t4-2,2)) {
            A[(-2*t3+t4-1)] = 0.33333 * (B[(-2*t3+t4-1)-1] + B[(-2*t3+t4-1)] + B[(-2*t3+t4-1) + 1]);;
          }
        }
      }
    }
  }
}

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
