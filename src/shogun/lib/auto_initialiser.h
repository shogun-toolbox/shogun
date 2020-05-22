/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef __AUTO_INIT_FACTORY_H__
#define __AUTO_INIT_FACTORY_H__

#include <shogun/distance/EuclideanDistance.h>
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
		template <typename KernelType>
		class GammaFeatureNumberInit : public AutoInit
		{
			static constexpr std::string_view kName = "GammaFeatureNumberInit";
			static constexpr std::string_view kDescription =
			    "Automatic initialisation of the gamma dot product scaling "
			    "parameter. If the standard deviation of the features can be "
			    "calculated then gamma = 1 / (n_features * std(features)), "
			    "else gamma = 1 / n_features.";

		public:
			explicit GammaFeatureNumberInit(KernelType& kernel, float64_t alternative_value)
			    : AutoInit(kName, kDescription), m_kernel(kernel), m_alternative_value(alternative_value)
			{
			}

			~GammaFeatureNumberInit() override = default;

			Any operator()() const final
			{
				AutoValue<float64_t> result;
				const auto& gamma_param = m_kernel.m_gamma;
				if (std::holds_alternative<AutoValueEmpty>(gamma_param)) {
					const auto& lhs = m_kernel.get_lhs();
					switch (lhs->get_feature_class())
						{
						case EFeatureClass::C_DENSE:
						case EFeatureClass::C_SPARSE:
						{
							auto dot_features = lhs->template as<DotFeatures>();
							result = 1.0 / (static_cast<float64_t>(
							               dot_features->get_dim_feature_space()) *
							           dot_features->get_std(false)[0]);
						}
						default:
						{
							auto dot_features = lhs->template as<DotFeatures>();
							result = 1.0 / static_cast<float64_t>(
							              dot_features->get_dim_feature_space());
						}
					}
				}
				else
				{
					result = m_alternative_value;
				}
				return make_any(result);
			}

		private:
			KernelType& m_kernel;
			float64_t m_alternative_value;
			SG_DELETE_COPY_AND_ASSIGN(GammaFeatureNumberInit);
		};

		class GaussianWidthAutoInit : public AutoInit
		{
			static constexpr std::string_view kName = "GaussianWidthInit";
			static constexpr std::string_view kDescription =
			    "Automatic initialisation of the kernel log width "
			    "using the median of the pairwise euclidean distance "
			    "of all the data points, and then applying the square root "
			    "to the median devided by two.";

		public:
			explicit GaussianWidthAutoInit(GaussianKernel& kernel, float64_t alternative_value)
			    : AutoInit(kName, kDescription), m_kernel(kernel), m_alternative_value(alternative_value)
			{
			}

			~GaussianWidthAutoInit() override = default;

			Any operator()() const final
			{
				AutoValue<float64_t> width;
				if (std::holds_alternative<shogun::AutoValueEmpty>(m_kernel.m_log_width)) {
					
					const auto& lhs = m_kernel.get_lhs();
					const auto& rhs = m_kernel.get_rhs();

					switch (lhs->get_feature_class())
					{
					case EFeatureClass::C_DENSE:
					{
						auto dist = EuclideanDistance(std::static_pointer_cast<DotFeatures>(lhs), 
													  std::static_pointer_cast<DotFeatures>(rhs));
						const auto& distance_matrix = dist.get_distance_matrix<float64_t>();

						auto result = SGVector<float64_t>(
						    (lhs->get_num_vectors() * lhs->get_num_vectors() -
						     lhs->get_num_vectors()) /
						    2);

						// copy upper triangular wihout a particular order
						index_t idx = 0;
						for (auto j: range(lhs->get_num_vectors()))
						{
							for (auto i: range(j + 1, rhs->get_num_vectors()))
							{
								result[idx] = distance_matrix(i, j);
								++idx;
							}
						}
						width = GaussianKernel::to_log_width(linalg::median(result));
					} break;
					default:
						width = GaussianKernel::to_log_width(m_alternative_value);
					}
				}
				else
					width = m_kernel.m_log_width;
				return make_any(width);
			}

		private:
			GaussianKernel& m_kernel;
			float64_t m_alternative_value;
			SG_DELETE_COPY_AND_ASSIGN(GaussianWidthAutoInit);
		};
	} // namespace params
} // namespace shogun

#endif // __AUTO_INIT_FACTORY_H__
