/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef EIGEN3_H_
#define EIGEN3_H_

#ifdef HAVE_EIGEN3
#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#define EIGEN_MATRIXBASE_PLUGIN <shogun/lib/tapkee/utils/matrix.hpp>
#endif

#endif
