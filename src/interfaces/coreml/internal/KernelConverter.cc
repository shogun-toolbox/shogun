#include "internal/KernelConverter.h"

#include "format/SVM.pb.h"

#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/SigmoidKernel.h>

using namespace CoreML;
using namespace std;

namespace shogun
{
	namespace coreml
	{
		namespace detail
		{
			template<typename I>
			struct converter
			{
				static auto convert(const I* k);
				static auto convert(const CKernel* k)
				{
					return convert(k->as<const I>());
				}
			};

			template<>
			auto converter<CGaussianKernel>::convert(const CGaussianKernel *k)
			{
				auto rbf = new Specification::RBFKernel();
				rbf->set_gamma(k->get_width());
				return rbf;
			}

			template<>
			auto converter<CSigmoidKernel>::convert(const CSigmoidKernel *k)
			{
				auto sigmoid_kernel = new Specification::SigmoidKernel();
				sigmoid_kernel->set_gamma(k->get<float64_t>("gamma"));
				sigmoid_kernel->set_c(k->get<float64_t>("coef0"));
				return sigmoid_kernel;
			}

			template<>
			auto converter<CPolyKernel>::convert(const CPolyKernel *k)
			{
				auto poly_kernel = new Specification::PolyKernel();
				poly_kernel->set_degree(k->get<int32_t>("degree"));
				poly_kernel->set_c(k->get<float64_t>("c"));
				poly_kernel->set_gamma(k->get<float64_t>("gamma"));
				return poly_kernel;
			}
		} // namespace detail

		Specification::Kernel* KernelConverter::convert(const CKernel *k)
		{
			auto kernel_normalizer = k->get_normalizer();
			REQUIRE(kernel_normalizer == NULL || string(kernel_normalizer->get_name()) == "IdentityKernelNormalizer",
				"The kernel has a normaliser set that cannot be converted to CoreML format!")

			auto kernel_spec = new Specification::Kernel();
			auto kernel_type = string(k->get_name());

			if (kernel_type == "LinearKernel")
			{
				kernel_spec->set_allocated_linearkernel(new Specification::LinearKernel());
			}
			else if (kernel_type == "GaussianKernel")
			{
				kernel_spec->set_allocated_rbfkernel(detail::converter<CGaussianKernel>::convert(k));
			}
			else if (kernel_type == "SigmoidKernel")
			{
				kernel_spec->set_allocated_sigmoidkernel(detail::converter<CSigmoidKernel>::convert(k));
			}
			else if (kernel_type == "PolyKernel")
			{
				kernel_spec->set_allocated_polykernel(detail::converter<CPolyKernel>::convert(k));
			}
			else
			{
				delete kernel_spec;
				throw runtime_error("Kernel type is not supported in CoreML");
			}

			return kernel_spec;
		}
	}
}
