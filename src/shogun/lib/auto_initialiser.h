/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef __AUTO_INIT_FACTORY_H__
#define __AUTO_INIT_FACTORY_H__

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/abstract_auto_init.h>
#include <shogun/lib/any.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/kernel/DistanceKernel.h>

#include <iostream>

namespace shogun
{
	namespace params
	{
		class GammaFeatureNumberInit : public AutoInit
		{
			static const char* const kName;
			static const char* const kDescription;

		public:
			explicit GammaFeatureNumberInit(Kernel* kernel)
			    : AutoInit(kName, kDescription), m_kernel(kernel)
			{
			}

			~GammaFeatureNumberInit() override = default;

			Any operator()() final
			{
				require(m_kernel != nullptr, "m_kernel is not pointing to a Kernel object");

				auto lhs = m_kernel->get_lhs();
				switch (lhs->get_feature_class())
				{
					case EFeatureClass::C_DENSE:
					case EFeatureClass::C_SPARSE:
					{
						auto dot_features = lhs->as<DotFeatures>();
						return make_any(
						    1.0 /
						    (static_cast<float64_t>(
								dot_features->get_dim_feature_space()) *
								dot_features->get_std(false)[0]));
					}
					default:
					{
						auto dot_features = lhs->as<DotFeatures>();
						return make_any(
							1.0 /
							static_cast<float64_t>(
								dot_features->get_dim_feature_space()));
					}
				}
				case EFeatureClass::C_SPARSE:
				{
					auto dot_features = (CDotFeatures*)lhs;
					return make_any(
					    1.0 / (static_cast<float64_t>(
					               dot_features->get_dim_feature_space()) *
					           dot_features->get_std(false)[0]));
				}
				default:
					auto dot_features = (CDotFeatures*)lhs;
					return make_any(
					    1.0 / static_cast<float64_t>(
					              dot_features->get_dim_feature_space()));
				}
			}

		private:
			Kernel* m_kernel;
			SG_DELETE_COPY_AND_ASSIGN(GammaFeatureNumberInit);
		};

		class GaussianWidthAutoInit : public AutoInit
		{
			static const char* const kName;
			static const char* const kDescription;

		public:
			explicit GaussianWidthAutoInit(CKernel* kernel)
			    : AutoInit(kName, kDescription), m_kernel(kernel)
			{
			}

			~GaussianWidthAutoInit() override = default;

			Any operator()() final
			{
				REQUIRE(
				    m_kernel != nullptr,
				    "m_kernel is not pointing to a CKernel object");
				auto lhs = m_kernel->get_lhs();
				auto rhs = m_kernel->get_rhs();
				switch (lhs->get_feature_class())
				{
				case EFeatureClass::C_DENSE:
				{
					auto distance_kernel = CEuclideanDistance();
					distance_kernel.init(lhs, rhs);
					auto pdist = distance_kernel.get_distance_matrix();
					auto result = SGVector<float64_t>((lhs->get_num_vectors()*lhs->get_num_vectors()-lhs->get_num_vectors())/2);
					// copy upper triangular wihout a particular order
					index_t idx = 0;
					for (int i = 0; i < lhs->get_num_vectors(); ++i) {
						for (int j = i + 1; j < lhs->get_num_vectors(); ++j) {
							result.set_element(pdist.get_element(i, j), idx);
							++idx;
						}
					}
					return make_any(std::log(linalg::median(result) / 2.0) / 2.0);
				}
				default:
					return make_any(1.0);
				}
			}

		private:
			CKernel* m_kernel;
			SG_DELETE_COPY_AND_ASSIGN(GaussianWidthAutoInit);
		};
	}; // namespace params
} // namespace shogun

#endif // __AUTO_INIT_FACTORY_H__
