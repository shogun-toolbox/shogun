/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef LINALG_H_
#define LINALG_H_

#include <shogun/lib/config.h>

/**
 * Just include this file to use in your applications.
 */

/**
 * This namespace contains all linear algebra specific modules and operations.
 * The rest of the library is available only when the one of the supported
 * backend exists in the local system
 */

#ifdef HAVE_LINALG_LIB
namespace linalg
{

/**
 * Developer's Note :
 * - Changing the default backend would just require to change it in the following
 *   enum
 * - Please use the same names as HAVE_<BACKEND> macros (e.g. HAVE_EIGEN3)
 */

/**
 * @brief
 * All currently supported linear algebra backend libraries, with a default
 * backend, which will be used for all the tasks if any particular backend is
 * not set explicitly via cmake options.
 *
 * The enum defines these backends in order of priority as default backend, as
 * in, first defined one will be used as default
 *
 * Note - Currently EIGEN3 is the default (if its available)
 *
 */
enum class Backend
{
#ifdef HAVE_EIGEN3
	EIGEN3,
#endif
#ifdef HAVE_VIENNACL
	VIENNACL,
#endif
	DEFAULT = 0
};

/**
 * @brief
 * General purpose linalg_traits for compile time information about backends
 * set per module (see below). This uses the backend from the modules.
 * To get the backend set globally, use linalg_traits<ModuleName>::backend
 */
template <class Module>
struct linalg_traits : Module
{
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/**
 * Define the modules as type with information about backend
 */
#ifndef SET_MODULE_BACKEND
#define SET_MODULE_BACKEND(MODULE, BACKEND) \
struct MODULE \
{ \
	const static Backend backend = Backend::BACKEND; \
};
#endif // SET_MODULE_BACKEND

/**
 * Set global backend should define all the module types with same backend.
 * Currently supported modules are
 * Redux        - For reduction to a scalar from vector or matrix (e.g. norm, sum, dot)
 * Linsolver    - Solvers for linear systems (SVD, Cholesky, QR etc)
 * Eigsolver    - Different eigensolvers
 */
#ifndef SET_GLOBAL_BACKEND
#define SET_GLOBAL_BACKEND(BACKEND) \
	SET_MODULE_BACKEND(Redux, BACKEND) \
	SET_MODULE_BACKEND(Linsolver, BACKEND) \
	SET_MODULE_BACKEND(Eigsolver, BACKEND)
#endif // SET_GLOBAL_BACKEND

/** set global backend for all modules if a particular backend is specified */
#ifdef USE_EIGEN3
	SET_GLOBAL_BACKEND(EIGEN3)
#elif USE_VIENNACL
	SET_GLOBAL_BACKEND(VIENNACL)
#else

/** set module specific backends */

/** Reduction module */
#ifdef USE_EIGEN3_REDUX
	SET_MODULE_BACKEND(Redux, EIGEN3)
#elif USE_VIENNACL_REDUX
	SET_MODULE_BACKEND(Redux, VIENNACL)
#else // the default case
	SET_MODULE_BACKEND(Redux, DEFAULT)
#endif

/** Linear solver module */
#ifdef USE_EIGEN3_LINSLV
	SET_MODULE_BACKEND(Linsolver, EIGEN3)
#elif USE_VIENNACL_LINSLV
	SET_MODULE_BACKEND(Linsolver, VIENNACL)
#else // the default case
	SET_MODULE_BACKEND(Linsolver, DEFAULT)
#endif

/** Eigen solver module */
#ifdef USE_EIGEN3_EIGSLV
	SET_MODULE_BACKEND(Eigsolver, EIGEN3)
#elif USE_VIENNACL_EIGSLV
	SET_MODULE_BACKEND(Eigsolver, VIENNACL)
#else // the default case
	SET_MODULE_BACKEND(Eigsolver, DEFAULT)
#endif

#endif // end of global settings

#undef SET_GLOBAL_BACKEND
#undef SET_MODULE_BACKEND

#endif // DOXYGEN_SHOULD_SKIP_THIS
}

/** include all the modules here */

#include <shogun/mathematics/linalg/internal/modules/redux.h>

#endif // HAVE_LINALG_LIB

#endif // LINALG_H_
