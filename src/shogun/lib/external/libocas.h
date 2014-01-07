/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * libocas.h: Implementation of the OCAS solver for training
 *            linear SVM classifiers.
 *
 * Copyright (C) 2008, 2009 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 *                          Soeren Sonnenburg, soeren.sonnenburg@first.fraunhofer.de
 *  Implementation of SVM-Ocas solver.
 *-------------------------------------------------------------------- */

#include <lib/common.h>

#ifndef libocas_h
#define libocas_h
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace shogun
{
#define LIBOCAS_PLUS_INF (-log(0.0))
#define LIBOCAS_CALLOC(x,y) SG_CALLOC(y,x)
#define LIBOCAS_FREE(x) SG_FREE(x)
#define LIBOCAS_INDEX(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define LIBOCAS_MIN(A,B) ((A) > (B) ? (B) : (A))
#define LIBOCAS_MAX(A,B) ((A) < (B) ? (B) : (A))
#define LIBOCAS_ABS(A) ((A) < 0 ? -(A) : (A))

typedef struct {
  uint32_t nIter;        /* number of iterations */
  uint32_t nCutPlanes;   /* number of cutitng buffered planes */
  uint32_t nNZAlpha;     /* number of non-zero Lagrangeans (effective number of CPs) */
  uint32_t trn_err;      /* number of training errors */
  float64_t Q_P;            /* primal objective value */
  float64_t Q_D;            /* dual objective value */
  float64_t output_time;    /* time spent in computing outputs */
  float64_t sort_time;      /* time spent in sorting */
  float64_t add_time;       /* time spent in adding examples to compute cutting planes */
  float64_t w_time;         /* time spent in computing parameter vector  */
  float64_t qp_solver_time; /* time spent in inner QP solver  */
  float64_t ocas_time;      /* total time spent in svm_ocas_solver */
  float64_t print_time;     /* time spent in ocas_print function */
  int8_t qp_exitflag;    /* exitflag from the last call of the inner QP solver */
  int8_t exitflag;       /*  1 .. ocas.Q_P - ocas.Q_D <= TolRel*ABS(ocas.Q_P)
                             2 .. ocas.Q_P - ocas.Q_D <= TolAbs
                             3 .. ocas.Q_P <= QPBound
                             4 .. optimization time >= MaxTime
                            -1 .. ocas.nCutPlanes >= BufSize
                            -2 .. not enough memory for the solver */
} ocas_return_value_T;

/* binary linear SVM solver */
ocas_return_value_T svm_ocas_solver(
         float64_t C,            /* regularizarion constant */
         uint32_t nData,      /* number of exmaples */
         float64_t TolRel,       /* halts if 1-Q_P/Q_D <= TolRel */
         float64_t TolAbs,       /* halts if Q_P-Q_D <= TolRel */
         float64_t QPBound,      /* halts if QP <= QPBound */
         float64_t MaxTime,      /* maximal time in seconds spent in optmization */
         uint32_t BufSize,    /* maximal number of buffered cutting planes  */
         uint8_t Method,      /* 0..standard CP (SVM-Perf,BMRM), 1..OCAS */
         void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
         float64_t (*update_W)(float64_t, void*),
         int (*add_new_cut)(float64_t*, uint32_t*, uint32_t, uint32_t, void*),
         int (*compute_output)( float64_t*, void* ),
         int (*sort)(float64_t*, float64_t*, uint32_t),
         void (*ocas_print)(ocas_return_value_T),
         void* user_data);

/* binary linear SVM solver which allows using different C for each example*/
ocas_return_value_T svm_ocas_solver_difC(
         float64_t *C,           /* regularizarion constants for each example */
         uint32_t nData,      /* number of exmaples */
         float64_t TolRel,       /* halts if 1-Q_P/Q_D <= TolRel */
         float64_t TolAbs,       /* halts if Q_P-Q_D <= TolRel */
         float64_t QPBound,      /* halts if QP <= QPBound */
         float64_t MaxTime,      /* maximal time in seconds spent in optmization */
         uint32_t BufSize,    /* maximal number of buffered cutting planes  */
         uint8_t Method,      /* 0..standard CP (SVM-Perf,BMRM), 1..OCAS */
         void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
         float64_t (*update_W)(float64_t, void*),
         int (*add_new_cut)(float64_t*, uint32_t*, uint32_t, uint32_t, void*),
         int (*compute_output)( float64_t*, void* ),
         int (*sort)(float64_t*, float64_t*, uint32_t),
         void (*ocas_print)(ocas_return_value_T),
         void* user_data);

/* multi-class (Singer-Crammer formulation) linear SVM solver */
ocas_return_value_T msvm_ocas_solver(
            float64_t C,
            float64_t *data_y,
            uint32_t nY,
            uint32_t nData,
            float64_t TolRel,
            float64_t TolAbs,
            float64_t QPBound,
            float64_t MaxTime,
            uint32_t _BufSize,
            uint8_t Method,
            void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
            float64_t (*update_W)(float64_t, void*),
            int (*add_new_cut)(float64_t*, uint32_t*, uint32_t, void*),
            int (*compute_output)(float64_t*, void* ),
            int (*sort)(float64_t*, float64_t*, uint32_t),
			void (*ocas_print)(ocas_return_value_T),
			void* user_data);


/* binary linear SVM solver */
ocas_return_value_T svm_ocas_solver_nnw(
         float64_t C,            /* regularizarion constant */
         uint32_t nData,      /* number of exmaples */
         uint32_t num_nnw,    /* number of components of W which must non-negative*/
         uint32_t* nnw_idx,   /* indices of W which must be non-negative */
         float64_t TolRel,       /* halts if 1-Q_P/Q_D <= TolRel */
         float64_t TolAbs,       /* halts if Q_P-Q_D <= TolRel */
         float64_t QPBound,      /* halts if QP <= QPBound */
         float64_t MaxTime,      /* maximal time in seconds spent in optmization */
         uint32_t BufSize,    /* maximal number of buffered cutting planes  */
         uint8_t Method,      /* 0..standard CP (SVM-Perf,BMRM), 1..OCAS */
         int (*add_pw_constr)(uint32_t, uint32_t, void*),
         void (*clip_neg_w)(uint32_t, uint32_t*, void*),
         void (*compute_W)(float64_t*, float64_t*, float64_t*, uint32_t, void*),
         float64_t (*update_W)(float64_t, void*),
         int (*add_new_cut)(float64_t*, uint32_t*, uint32_t, uint32_t, void*),
         int (*compute_output)( float64_t*, void* ),
         int (*sort)(float64_t*, float64_t*, uint32_t),
         void (*ocas_print)(ocas_return_value_T),
         void* user_data);

}
#endif // DOXYGEN_SHOULD_SKIP_THIS
#endif /* libocas_h */

