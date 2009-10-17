/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Vojtech Franc
 */
namespace shogun
{
int8_t qpssvm_solver(const void* (*get_col)(uint32_t),
                  float64_t *diag_H,
                  float64_t *f,
                  float64_t b,
                  uint16_t *I,
                  float64_t *x,
                  uint32_t n,
                  uint32_t tmax,
                  float64_t tolabs,
                  float64_t tolrel,
                  float64_t *QP,
                  float64_t *QD,
                  uint32_t verb);
}
