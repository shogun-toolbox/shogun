/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_H_
#define TAPKEE_DEFINES_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/exceptions.hpp>
#include <shogun/lib/tapkee/parameters/parameters.hpp>
#include <shogun/lib/tapkee/traits/callbacks_traits.hpp>
#include <shogun/lib/tapkee/traits/methods_traits.hpp>
/* End of Tapkee includes */

#include <string>

#define TAPKEE_WORLD_VERSION 1
#define TAPKEE_MAJOR_VERSION 0
#define TAPKEE_MINOR_VERSION 0

/* Tapkee includes */
#include <shogun/lib/tapkee/defines/eigen3.hpp>
#include <shogun/lib/tapkee/defines/types.hpp>
#include <shogun/lib/tapkee/defines/methods.hpp>
#include <shogun/lib/tapkee/defines/keywords.hpp>
#include <shogun/lib/tapkee/defines/stdtypes.hpp>
#include <shogun/lib/tapkee/defines/synonyms.hpp>
#include <shogun/lib/tapkee/defines/random.hpp>
#include <shogun/lib/tapkee/projection.hpp>
/* End of Tapkee includes */

#ifdef TAPKEE_CUSTOM_PROPERTIES
	#include TAPKEE_CUSTOM_PROPERTIES
#else
	//! Base of covertree. Could be overrided if TAPKEE_CUSTOM_PROPERTIES file is defined.
	#define COVERTREE_BASE 1.3
#endif

namespace tapkee
{
	//! Return result of the library - a pair of @ref DenseMatrix (embedding) and @ref ProjectingFunction
	struct TapkeeOutput
	{
		TapkeeOutput() :
			embedding(), projection()
		{
		}
		TapkeeOutput(const tapkee::DenseMatrix& e, const tapkee::ProjectingFunction& p) :
			embedding(), projection(p)
		{
			embedding.swap(e);
		}
		TapkeeOutput(const TapkeeOutput& that) :
			embedding(), projection(that.projection)
		{
			this->embedding.swap(that.embedding);
		}
		tapkee::DenseMatrix embedding;
		tapkee::ProjectingFunction projection;
	};
}

#endif // TAPKEE_DEFINES_H_
