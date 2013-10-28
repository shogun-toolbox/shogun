#ifndef TAPKEE_DEFINES_EIGEN3_H_
#define TAPKEE_DEFINES_EIGEN3_H_

//// Eigen 3 library includes
#ifdef TAPKEE_EIGEN_INCLUDE_FILE
	#include TAPKEE_EIGEN_INCLUDE_FILE
#else
	#ifndef TAPKEE_DEBUG
		#define EIGEN_NO_DEBUG
	#endif
	#define EIGEN_RUNTIME_NO_MALLOC
	#include <Eigen/Eigen>
	#include <Eigen/Dense>
	#if EIGEN_VERSION_AT_LEAST(3,0,93)
		#include <Eigen/Sparse>
		#if defined(TAPKEE_SUPERLU_AVAILABLE) && defined(TAPKEE_USE_SUPERLU)
			#include <Eigen/SuperLUSupport>
		#endif
	#else
		#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
		#include <unsupported/Eigen/SparseExtra>
	#endif
#endif

#ifdef EIGEN_RUNTIME_NO_MALLOC
	#define RESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(false)
	#define UNRESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(true)
#else
	#define RESTRICT_ALLOC
	#define UNRESTRICT_ALLOC
#endif
//// end of Eigen 3 library includes

#endif
