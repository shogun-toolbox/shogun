/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * Written (W) 2007 Vojtech Franc 
 */

int qpssvm_solver(const void* (*get_col)(uint32_t),
                  double *diag_H,
                  double *f,
                  double b,
                  uint16_t *I,
                  double *x,
                  uint32_t n,
                  uint32_t tmax,
                  double tolabs,
                  double tolrel,
                  double *QP,
                  double *QD,
				  int (*qpssvmlib_print)(const char *format, ...),
                  uint32_t verb);
