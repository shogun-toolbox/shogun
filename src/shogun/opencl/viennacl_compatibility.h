#ifndef VIENNACL_COMPATIBILITY_HPP
#define VIENNACL_COMPATIBILITY_HPP

#include <shogun/features/FeatureTypes.h>

namespace shogun{

		template<class ST>
		struct make_vcl_compatible{
			typedef float Result;
		};

		template<>
		struct make_vcl_compatible<double>{
			typedef double Result;
		};
}

#endif