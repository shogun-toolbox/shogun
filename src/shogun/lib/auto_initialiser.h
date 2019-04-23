/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef __AUTO_INIT_FACTORY_H__
#define __AUTO_INIT_FACTORY_H__

#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/abstract_auto_init.h>
#include <shogun/lib/any.h>

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

			Any operator()() override
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
			}

		private:
			Kernel* m_kernel;
			SG_DELETE_COPY_AND_ASSIGN(GammaFeatureNumberInit);
		};
	} // namespace factory
} // namespace shogun

#endif // __AUTO_INIT_FACTORY_H__
