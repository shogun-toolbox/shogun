/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef __AUTO_INIT_FACTORY_H__
#define __AUTO_INIT_FACTORY_H__

#include <iostream>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/abstract_auto_init.h>
#include <shogun/lib/any.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

namespace shogun
{
	namespace params
	{
		class GammaFeatureNumberInit : public AutoInit
		{
			static constexpr std::string_view kName = "GammaFeatureNumberInit";
			static constexpr std::string_view kDescription =
			    "Automatic initialisation of the gamma dot product scaling "
			    "parameter. If the standard deviation of the features can be "
			    "calculated then gamma = 1 / (n_features * std(features)), "
			    "else gamma = 1 / n_features.";

		public:
			explicit GammaFeatureNumberInit(Kernel& kernel)
			    : AutoInit(kName, kDescription), m_kernel(kernel)
			{
			}

			~GammaFeatureNumberInit() override = default;

			Any operator()() const final
			{
				auto lhs = m_kernel.get_lhs();
				switch (lhs->get_feature_class())
				{
				case EFeatureClass::C_DENSE:
				case EFeatureClass::C_SPARSE:
				{
					auto dot_features = lhs->as<DotFeatures>();
					return make_any(
					    1.0 / (static_cast<float64_t>(
					               dot_features->get_dim_feature_space()) *
					           dot_features->get_std(false)[0]));
				}
				default:
				{
					auto dot_features = lhs->as<DotFeatures>();
					return make_any(
					    1.0 / static_cast<float64_t>(
					              dot_features->get_dim_feature_space()));
				}
				}
			}

		private:
			Kernel& m_kernel;
			SG_DELETE_COPY_AND_ASSIGN(GammaFeatureNumberInit);
		};

		class GaussianWidthAutoInit : public AutoInit
		{
			static constexpr std::string_view kName = "GaussianWidthInit";
			static constexpr std::string_view kDescription =
			    "Automatic initialisation of the kernel log width "
			    "using the median of the pairwise euclidean distance "
			    "of all the features.";

		public:
			explicit GaussianWidthAutoInit(GaussianKernel& kernel)
			    : AutoInit(kName, kDescription), m_kernel(kernel)
			{
			}

			~GaussianWidthAutoInit() override = default;

			Any operator()() const final
			{
				auto lhs = m_kernel.get_lhs();
				auto rhs = m_kernel.get_rhs();

				switch (lhs->get_feature_class())
				{
				case EFeatureClass::C_DENSE:
				{
					auto result = SGVector<float64_t>(
					    (lhs->get_num_vectors() * lhs->get_num_vectors() -
					     lhs->get_num_vectors()) /
					    2);

					// copy upper triangular wihout a particular order
					index_t idx = 0;
					for (int i = 0; i < rhs->get_num_vectors(); ++i)
					{
						for (int j = i + 1; j < lhs->get_num_vectors(); ++j)
						{
							result[idx] = m_kernel.distance(i, j);
							++idx;
						}
					}
					return make_any(
					    GaussianKernel::to_log_width(linalg::median(result)));
				}
				default:
				{
					float64_t default_value =
					    GaussianKernel::to_log_width(m_kernel.get_width());
					return make_any(default_value);
				}
				}
			}

		private:
			GaussianKernel& m_kernel;
			SG_DELETE_COPY_AND_ASSIGN(GaussianWidthAutoInit);
		};
	}; // namespace params
} // namespace shogun

#endif // __AUTO_INIT_FACTORY_H__
