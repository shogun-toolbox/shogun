/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 *  Implementation of SVM-Ocas solver.
 *
 *  Linear binary SVM solver without bias term.
 *
 * Modifications:
 * 10-oct-2007, VF, created.
 * 14-nov-2007, VF, timing statistics added
 * ----------------------------------------------------------------------*/

#include "lib/common.h"

/** ocas return value */
typedef struct {
  /** number of iterations */
  uint32_t nIter;
  /** number of cutitng buffered planes */
  uint32_t nCutPlanes;
  /** number of non-zero Lagrangeans (effective number of CPs) */
  uint32_t nNZAlpha;
  /** number of training errors */
  float64_t trn_err;
  /** primal objective value */
  float64_t Q_P;
  /** dual objective value */
  float64_t Q_D;
  /** time spent in computing outputs */
  float64_t output_time;
  /** time spent in sorting */
  float64_t sort_time;
  /** time spent in adding examples to compute cutting planes */
  float64_t add_time;
  /** time spent in computing parameter vector  */
  float64_t w_time;
  /** time spend in inner QP solver  */
  float64_t solver_time;
  /** total time spent in svm_ocas_solver */
  float64_t ocas_time;

  /** 1 .. ocas.Q_P - ocas.Q_D <= TolRel*ABS(ocas.Q_P)
   *  2 .. ocas.Q_P - ocas.Q_D <= TolAbs
   *  3 .. ocas.Q_P <= QPBound -1 .. ocas.nCutPlanes >= BufSize
   */
  int8_t exitflag;
} ocas_return_value_T;

ocas_return_value_T svm_ocas_solver(
		float64_t C,            /* regularizarion constant */
		uint32_t nData,      /* number of exmaples */
		float64_t TolRel,       /* halts if 1-Q_P/Q_D <= TolRel */
		float64_t TolAbs,       /* halts if Q_P-Q_D <= TolRel */
		float64_t QPBound,      /* halts if QP <= QPBound */
		uint32_t BufSize,    /* maximal number of buffered cutting planes  */
		uint8_t Method,      /* 0..standard CP (SVM-Perf,BMRM), 1..OCAS */
		void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
		float64_t (*update_W)(float64_t, void*),
		void (*add_new_cut)(float64_t*, uint32_t*, uint32_t, uint32_t, void*),
		void (*compute_output)( float64_t*, void* ),
		void (*sort)(float64_t*, uint32_t*, uint32_t),
		void* user_data);

void qsort_index(float64_t* value, uint32_t* index, uint32_t size);
