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

#include <stdint.h>

/** ocas return value */
typedef struct {
  /** number of iterations */
  uint32_t nIter;
  /** number of cutitng buffered planes */
  uint32_t nCutPlanes;
  /** number of non-zero Lagrangeans (effective number of CPs) */
  uint32_t nNZAlpha;
  /** number of training errors */
  double trn_err;
  /** primal objective value */
  double Q_P;
  /** dual objective value */
  double Q_D;
  /** time spent in computing outputs */
  double output_time;
  /** time spent in sorting */
  double sort_time;
  /** time spent in adding examples to compute cutting planes */
  double add_time;
  /** time spent in computing parameter vector  */
  double w_time;
  /** time spend in inner QP solver  */
  double solver_time;
  /** total time spent in svm_ocas_solver */
  double ocas_time;

  /** 1 .. ocas.Q_P - ocas.Q_D <= TolRel*ABS(ocas.Q_P)
   *  2 .. ocas.Q_P - ocas.Q_D <= TolAbs
   *  3 .. ocas.Q_P <= QPBound -1 .. ocas.nCutPlanes >= BufSize
   */
  int8_t exitflag;
} ocas_return_value_T;

ocas_return_value_T svm_ocas_solver(
		double C,            /* regularizarion constant */
		uint32_t nData,      /* number of exmaples */
		double TolRel,       /* halts if 1-Q_P/Q_D <= TolRel */
		double TolAbs,       /* halts if Q_P-Q_D <= TolRel */
		double QPBound,      /* halts if QP <= QPBound */
		uint32_t BufSize,    /* maximal number of buffered cutting planes  */
		uint8_t Method,      /* 0..standard CP (SVM-Perf,BMRM), 1..OCAS */
		void (*compute_W)(double*, double*, double*, uint32_t, void*),
		double (*update_W)(double, void*),
		void (*add_new_cut)(double*, uint32_t*, uint32_t, uint32_t, void*),
		void (*compute_output)( double*, void* ),
		void (*sort)(double*, uint32_t*, uint32_t),
		void* user_data);

void qsort_index(double* value, uint32_t* index, uint32_t size);
