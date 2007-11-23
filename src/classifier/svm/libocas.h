/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 *  Implementation of SVM-Ocas solver. 
 *
 *  Linear binary SVM solver without bias term.
 *
 * Modifications:
 * 10-oct-2007, VF, created.
 * ----------------------------------------------------------------------*/

#include <stdint.h>

typedef struct {
  uint32_t nIter;      
  uint32_t nCutPlanes;
  double trn_err;      
  double Q_P;          
  double Q_D;
  double output_time;
  double sort_time;
  double solver_time;
  int8_t exitflag;       
} ocas_return_value_T;


ocas_return_value_T svm_ocas_solver(
            double C,
            uint32_t nData, 
            double TolRel,
            double TolAbs,
            double QPBound,
            uint32_t BufSize,
            uint8_t Method,
            void (*compute_W)(double*, double*, double*, uint32_t, void*),
            double (*update_W)(double, void*),
            void (*add_new_cut)(double*, uint32_t*, uint32_t, uint32_t, void*),
            void (*compute_output)( double*, void* ),
            void (*sort)(double*, uint32_t*, uint32_t),
			int (*ocas_print)(const char *format, ...),
			void* user_data);


void qsort_index(double* value, uint32_t* index, uint32_t size);
