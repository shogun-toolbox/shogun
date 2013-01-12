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
	//#define EIGEN_RUNTIME_NO_MALLOC
	#include <Eigen/Eigen>
	#include <Eigen/Dense>
	#if EIGEN_VERSION_AT_LEAST(3,1,0)
		#include <Eigen/Sparse>
	#else
		#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
		#include <unsupported/Eigen/SparseExtra>
	#endif
#endif

#endif
