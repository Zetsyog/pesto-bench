
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
#include "covariance.trahrhe.n0.g0.h"
#include "covariance.trahrhe.n0.g1.h"
#include "covariance.trahrhe.n0.g2.h"
#include "covariance.trahrhe.n0.g3.h"
#include "covariance.trahrhe.n0.g4.h"
#include "covariance.trahrhe.n0.g5.h"
#include "covariance.trahrhe.n0.g6.h"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* covariance.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "covariance.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)n;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_covariance(int m, int n,
		       DATA_TYPE float_n,
		       DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
		       DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
		       DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;



long int lbp;
long int ubp;
long int lbv;
long int ubv;
long int t2;
long int t3;
long int t4;
long int t5;
long int t6;
long int t7;
long int t1;
long int t8;
long int g0_t5_pcmax;
long int g0_TARGET_VOL_L0;
long int g0_ubt2;
long int g0_lbt5;
long int g0_ubt5;
long int g0_t7_pcmax;
long int g0_TARGET_VOL_L1;
long int g0_ubt3;
long int g0_lbt7;
long int g0_ubt7;
long int g0_t6_pcmax;
long int g0_TARGET_VOL_L2;
long int g0_ubt4;
long int g0_lbt6;
long int g0_ubt6;
long int g1_t4_pcmax;
long int g1_TARGET_VOL_L0;
long int g1_ubt2;
long int g1_lbt4;
long int g1_ubt4;
long int g1_t5_pcmax;
long int g1_TARGET_VOL_L1;
long int g1_ubt3;
long int g1_lbt5;
long int g1_ubt5;
long int g2_t4_pcmax;
long int g2_TARGET_VOL_L0;
long int g2_ubt2;
long int g2_lbt4;
long int g2_ubt4;
long int g2_t5_pcmax;
long int g2_TARGET_VOL_L1;
long int g2_ubt3;
long int g2_lbt5;
long int g2_ubt5;
long int g3_t5_pcmax;
long int g3_TARGET_VOL_L0;
long int g3_ubt2;
long int g3_lbt5;
long int g3_ubt5;
long int g3_t4_pcmax;
long int g3_TARGET_VOL_L1;
long int g3_ubt3;
long int g3_lbt4;
long int g3_ubt4;
long int g4_t4_pcmax;
long int g4_TARGET_VOL_L0;
long int g4_ubt2;
long int g4_lbt4;
long int g4_ubt4;
long int g4_t5_pcmax;
long int g4_TARGET_VOL_L1;
long int g4_ubt3;
long int g4_lbt5;
long int g4_ubt5;
long int g5_t3_pcmax;
long int g5_TARGET_VOL_L0;
long int g5_ubt2;
long int g5_lbt3;
long int g5_ubt3;
long int g6_t3_pcmax;
long int g6_TARGET_VOL_L0;
long int g6_ubt2;
long int g6_lbt3;
long int g6_ubt3;
g4_ubt2 = DIV0 - 1;g4_t4_pcmax = n0_g4_t4_Ehrhart(_PB_M, _PB_N);g4_TARGET_VOL_L0 = g4_t4_pcmax / DIV0;
lbp=0;
ubp=g4_ubt2;
#pragma omp parallel for firstprivate(g4_ubt2, g4_t4_pcmax, g4_TARGET_VOL_L0) private(lbv, ubv,t3,t4,t5,t6,t7,t8, g4_lbt4, g4_ubt4, g4_ubt3, g4_t5_pcmax, g4_TARGET_VOL_L1, g4_lbt5, g4_ubt5)
for (t2=lbp;t2<=ubp;t2++) {
  g4_lbt4 = n0_g4_t4_trahrhe_n0_g4_t4(max(1, (t2) * g4_TARGET_VOL_L0), _PB_M, _PB_N);g4_ubt4 = n0_g4_t4_trahrhe_n0_g4_t4(min(g4_t4_pcmax, (t2 + 1) * g4_TARGET_VOL_L0), _PB_M, _PB_N) - 1;if(t2 == g4_ubt2) { g4_ubt4 = _PB_M-1;};
  g4_ubt3 = DIV1 - 1;g4_t5_pcmax = n0_g4_t5_Ehrhart(_PB_M, _PB_N, g4_lbt4, g4_ubt4);g4_TARGET_VOL_L1 = g4_t5_pcmax / DIV1;
  for (t3=0;t3<=g4_ubt3;t3++) {
    g4_lbt5 = n0_g4_t5_trahrhe_n0_g4_t5(max(1, (t3) * g4_TARGET_VOL_L1), _PB_M, _PB_N, g4_lbt4, g4_ubt4);g4_ubt5 = n0_g4_t5_trahrhe_n0_g4_t5(min(g4_t5_pcmax, (t3 + 1) * g4_TARGET_VOL_L1), _PB_M, _PB_N, g4_lbt4, g4_ubt4) - 1;if(t3 == g4_ubt3) { g4_ubt5 = _PB_M-1;};
    for (t4=g4_lbt4;t4<=min(_PB_M-1,g4_ubt4);t4++) {
      lbv=max(t4,g4_lbt5);
      ubv=min(_PB_M-1,g4_ubt5);
#pragma GCC ivdep
      for (t5=lbv;t5<=ubv;t5++) {
        cov[t4][t5] = SCALAR_VAL(0.0);;
      }
    }
  }
}
g6_ubt2 = DIV0 - 1;g6_t3_pcmax = n0_g6_t3_Ehrhart(_PB_M, _PB_N);g6_TARGET_VOL_L0 = g6_t3_pcmax / DIV0;
lbp=0;
ubp=g6_ubt2;
#pragma omp parallel for firstprivate(g6_ubt2, g6_t3_pcmax, g6_TARGET_VOL_L0) private(lbv, ubv,t3,t4,t5,t6,t7,t8, g6_lbt3, g6_ubt3)
for (t2=lbp;t2<=ubp;t2++) {
  g6_lbt3 = n0_g6_t3_trahrhe_n0_g6_t3(max(1, (t2) * g6_TARGET_VOL_L0), _PB_M, _PB_N);g6_ubt3 = n0_g6_t3_trahrhe_n0_g6_t3(min(g6_t3_pcmax, (t2 + 1) * g6_TARGET_VOL_L0), _PB_M, _PB_N) - 1;if(t2 == g6_ubt2) { g6_ubt3 = _PB_M-1;};
  lbv=g6_lbt3;
  ubv=min(_PB_M-1,g6_ubt3);
#pragma GCC ivdep
  for (t3=lbv;t3<=ubv;t3++) {
    mean[t3] = SCALAR_VAL(0.0);;
  }
}
g3_ubt2 = DIV0 - 1;g3_t5_pcmax = n0_g3_t5_Ehrhart(_PB_M, _PB_N);g3_TARGET_VOL_L0 = g3_t5_pcmax / DIV0;
lbp=0;
ubp=g3_ubt2;
#pragma omp parallel for firstprivate(g3_ubt2, g3_t5_pcmax, g3_TARGET_VOL_L0) private(lbv, ubv,t3,t4,t5,t6,t7,t8, g3_lbt5, g3_ubt5, g3_ubt3, g3_t4_pcmax, g3_TARGET_VOL_L1, g3_lbt4, g3_ubt4)
for (t2=lbp;t2<=ubp;t2++) {
  g3_lbt5 = n0_g3_t5_trahrhe_n0_g3_t5(max(1, (t2) * g3_TARGET_VOL_L0), _PB_M, _PB_N);g3_ubt5 = n0_g3_t5_trahrhe_n0_g3_t5(min(g3_t5_pcmax, (t2 + 1) * g3_TARGET_VOL_L0), _PB_M, _PB_N) - 1;if(t2 == g3_ubt2) { g3_ubt5 = _PB_M-1;};
  g3_ubt3 = DIV1 - 1;g3_t4_pcmax = n0_g3_t4_Ehrhart(_PB_M, _PB_N, g3_lbt5, g3_ubt5);g3_TARGET_VOL_L1 = g3_t4_pcmax / DIV1;
  for (t3=0;t3<=g3_ubt3;t3++) {
    g3_lbt4 = n0_g3_t4_trahrhe_n0_g3_t4(max(1, (t3) * g3_TARGET_VOL_L1), _PB_M, _PB_N, g3_lbt5, g3_ubt5);g3_ubt4 = n0_g3_t4_trahrhe_n0_g3_t4(min(g3_t4_pcmax, (t3 + 1) * g3_TARGET_VOL_L1), _PB_M, _PB_N, g3_lbt5, g3_ubt5) - 1;if(t3 == g3_ubt3) { g3_ubt4 = _PB_N-1;};
    for (t4=g3_lbt4;t4<=min(_PB_N-1,g3_ubt4);t4++) {
      lbv=g3_lbt5;
      ubv=min(_PB_M-1,g3_ubt5);
#pragma GCC ivdep
      for (t5=lbv;t5<=ubv;t5++) {
        mean[t5] += data[t4][t5];;
      }
    }
  }
}
g5_ubt2 = DIV0 - 1;g5_t3_pcmax = n0_g5_t3_Ehrhart(_PB_M, _PB_N);g5_TARGET_VOL_L0 = g5_t3_pcmax / DIV0;
lbp=0;
ubp=g5_ubt2;
#pragma omp parallel for firstprivate(g5_ubt2, g5_t3_pcmax, g5_TARGET_VOL_L0) private(lbv, ubv,t3,t4,t5,t6,t7,t8, g5_lbt3, g5_ubt3)
for (t2=lbp;t2<=ubp;t2++) {
  g5_lbt3 = n0_g5_t3_trahrhe_n0_g5_t3(max(1, (t2) * g5_TARGET_VOL_L0), _PB_M, _PB_N);g5_ubt3 = n0_g5_t3_trahrhe_n0_g5_t3(min(g5_t3_pcmax, (t2 + 1) * g5_TARGET_VOL_L0), _PB_M, _PB_N) - 1;if(t2 == g5_ubt2) { g5_ubt3 = _PB_M-1;};
  lbv=g5_lbt3;
  ubv=min(_PB_M-1,g5_ubt3);
#pragma GCC ivdep
  for (t3=lbv;t3<=ubv;t3++) {
    mean[t3] /= float_n;;
  }
}
g2_ubt2 = DIV0 - 1;g2_t4_pcmax = n0_g2_t4_Ehrhart(_PB_M, _PB_N);g2_TARGET_VOL_L0 = g2_t4_pcmax / DIV0;
lbp=0;
ubp=g2_ubt2;
#pragma omp parallel for firstprivate(g2_ubt2, g2_t4_pcmax, g2_TARGET_VOL_L0) private(lbv, ubv,t3,t4,t5,t6,t7,t8, g2_lbt4, g2_ubt4, g2_ubt3, g2_t5_pcmax, g2_TARGET_VOL_L1, g2_lbt5, g2_ubt5)
for (t2=lbp;t2<=ubp;t2++) {
  g2_lbt4 = n0_g2_t4_trahrhe_n0_g2_t4(max(1, (t2) * g2_TARGET_VOL_L0), _PB_M, _PB_N);g2_ubt4 = n0_g2_t4_trahrhe_n0_g2_t4(min(g2_t4_pcmax, (t2 + 1) * g2_TARGET_VOL_L0), _PB_M, _PB_N) - 1;if(t2 == g2_ubt2) { g2_ubt4 = _PB_N-1;};
  g2_ubt3 = DIV1 - 1;g2_t5_pcmax = n0_g2_t5_Ehrhart(_PB_M, _PB_N, g2_lbt4, g2_ubt4);g2_TARGET_VOL_L1 = g2_t5_pcmax / DIV1;
  for (t3=0;t3<=g2_ubt3;t3++) {
    g2_lbt5 = n0_g2_t5_trahrhe_n0_g2_t5(max(1, (t3) * g2_TARGET_VOL_L1), _PB_M, _PB_N, g2_lbt4, g2_ubt4);g2_ubt5 = n0_g2_t5_trahrhe_n0_g2_t5(min(g2_t5_pcmax, (t3 + 1) * g2_TARGET_VOL_L1), _PB_M, _PB_N, g2_lbt4, g2_ubt4) - 1;if(t3 == g2_ubt3) { g2_ubt5 = _PB_M-1;};
    for (t4=g2_lbt4;t4<=min(_PB_N-1,g2_ubt4);t4++) {
      lbv=g2_lbt5;
      ubv=min(_PB_M-1,g2_ubt5);
#pragma GCC ivdep
      for (t5=lbv;t5<=ubv;t5++) {
        data[t4][t5] -= mean[t5];;
      }
    }
  }
}
g0_ubt2 = DIV0 - 1;g0_t5_pcmax = n0_g0_t5_Ehrhart(_PB_M, _PB_N);g0_TARGET_VOL_L0 = g0_t5_pcmax / DIV0;
lbp=0;
ubp=g0_ubt2;
#pragma omp parallel for firstprivate(g0_ubt2, g0_t5_pcmax, g0_TARGET_VOL_L0) private(lbv, ubv,t3,t4,t5,t6,t7,t8, g0_lbt5, g0_ubt5, g0_ubt3, g0_t7_pcmax, g0_TARGET_VOL_L1, g0_lbt7, g0_ubt7, g0_ubt4, g0_t6_pcmax, g0_TARGET_VOL_L2, g0_lbt6, g0_ubt6)
for (t2=lbp;t2<=ubp;t2++) {
  g0_lbt5 = n0_g0_t5_trahrhe_n0_g0_t5(max(1, (t2) * g0_TARGET_VOL_L0), _PB_M, _PB_N);g0_ubt5 = n0_g0_t5_trahrhe_n0_g0_t5(min(g0_t5_pcmax, (t2 + 1) * g0_TARGET_VOL_L0), _PB_M, _PB_N) - 1;if(t2 == g0_ubt2) { g0_ubt5 = _PB_M-1;};
  g0_ubt3 = DIV1 - 1;g0_t7_pcmax = n0_g0_t7_Ehrhart(_PB_M, _PB_N, g0_lbt5, g0_ubt5);g0_TARGET_VOL_L1 = g0_t7_pcmax / DIV1;
  for (t3=0;t3<=g0_ubt3;t3++) {
    g0_lbt7 = n0_g0_t7_trahrhe_n0_g0_t7(max(1, (t3) * g0_TARGET_VOL_L1), _PB_M, _PB_N, g0_lbt5, g0_ubt5);g0_ubt7 = n0_g0_t7_trahrhe_n0_g0_t7(min(g0_t7_pcmax, (t3 + 1) * g0_TARGET_VOL_L1), _PB_M, _PB_N, g0_lbt5, g0_ubt5) - 1;if(t3 == g0_ubt3) { g0_ubt7 = _PB_M-1;};
    g0_ubt4 = DIV2 - 1;g0_t6_pcmax = n0_g0_t6_Ehrhart(_PB_M, _PB_N, g0_lbt5, g0_ubt5, g0_lbt7, g0_ubt7);g0_TARGET_VOL_L2 = g0_t6_pcmax / DIV2;
    for (t4=0;t4<=g0_ubt4;t4++) {
      g0_lbt6 = n0_g0_t6_trahrhe_n0_g0_t6(max(1, (t4) * g0_TARGET_VOL_L2), _PB_M, _PB_N, g0_lbt5, g0_ubt5, g0_lbt7, g0_ubt7);g0_ubt6 = n0_g0_t6_trahrhe_n0_g0_t6(min(g0_t6_pcmax, (t4 + 1) * g0_TARGET_VOL_L2), _PB_M, _PB_N, g0_lbt5, g0_ubt5, g0_lbt7, g0_ubt7) - 1;if(t4 == g0_ubt4) { g0_ubt6 = _PB_N-1;};
      for (t5=g0_lbt5;t5<=min(_PB_M-1,g0_ubt5);t5++) {
        for (t6=g0_lbt6;t6<=min(_PB_N-1,g0_ubt6);t6++) {
          lbv=max(t5,g0_lbt7);
          ubv=min(_PB_M-1,g0_ubt7);
#pragma GCC ivdep
          for (t7=lbv;t7<=ubv;t7++) {
            cov[t5][t7] += data[t6][t5] * data[t6][t7];;
          }
        }
      }
    }
  }
}
g1_ubt2 = DIV0 - 1;g1_t4_pcmax = n0_g1_t4_Ehrhart(_PB_M, _PB_N);g1_TARGET_VOL_L0 = g1_t4_pcmax / DIV0;
lbp=0;
ubp=g1_ubt2;
#pragma omp parallel for firstprivate(g1_ubt2, g1_t4_pcmax, g1_TARGET_VOL_L0) private(lbv, ubv,t3,t4,t5,t6,t7,t8, g1_lbt4, g1_ubt4, g1_ubt3, g1_t5_pcmax, g1_TARGET_VOL_L1, g1_lbt5, g1_ubt5)
for (t2=lbp;t2<=ubp;t2++) {
  g1_lbt4 = n0_g1_t4_trahrhe_n0_g1_t4(max(1, (t2) * g1_TARGET_VOL_L0), _PB_M, _PB_N);g1_ubt4 = n0_g1_t4_trahrhe_n0_g1_t4(min(g1_t4_pcmax, (t2 + 1) * g1_TARGET_VOL_L0), _PB_M, _PB_N) - 1;if(t2 == g1_ubt2) { g1_ubt4 = _PB_M-1;};
  g1_ubt3 = DIV1 - 1;g1_t5_pcmax = n0_g1_t5_Ehrhart(_PB_M, _PB_N, g1_lbt4, g1_ubt4);g1_TARGET_VOL_L1 = g1_t5_pcmax / DIV1;
  for (t3=0;t3<=g1_ubt3;t3++) {
    g1_lbt5 = n0_g1_t5_trahrhe_n0_g1_t5(max(1, (t3) * g1_TARGET_VOL_L1), _PB_M, _PB_N, g1_lbt4, g1_ubt4);g1_ubt5 = n0_g1_t5_trahrhe_n0_g1_t5(min(g1_t5_pcmax, (t3 + 1) * g1_TARGET_VOL_L1), _PB_M, _PB_N, g1_lbt4, g1_ubt4) - 1;if(t3 == g1_ubt3) { g1_ubt5 = _PB_M-1;};
    for (t4=g1_lbt4;t4<=min(_PB_M-1,g1_ubt4);t4++) {
      lbv=max(t4,g1_lbt5);
      ubv=min(_PB_M-1,g1_ubt5);
#pragma GCC ivdep
      for (t5=lbv;t5<=ubv;t5++) {
        cov[t4][t5] /= (float_n - SCALAR_VAL(1.0));;
        cov[t5][t4] = cov[t4][t5];;
      }
    }
  }
}

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(cov,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);


  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_covariance (m, n, float_n,
		     POLYBENCH_ARRAY(data),
		     POLYBENCH_ARRAY(cov),
		     POLYBENCH_ARRAY(mean));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(cov)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}
