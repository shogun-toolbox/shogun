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
	namespace factory
	{
		class GammaFeatureNumberInit : public AutoInit
		{
			static const char* const kName;
			static const char* const kDescription;

		public:
			GammaFeatureNumberInit(CKernel* kernel)
			    : AutoInit(kName, kDescription)
			{
				SG_REF(kernel);
				m_kernel = kernel;
			}

			~GammaFeatureNumberInit() override
			{
				SG_UNREF(m_kernel);
			}

			Any operator()() override
			{
				Any result;
				auto m_lhs = m_kernel->get<CFeatures*>("lhs");
				if (m_lhs->get_feature_class() >=
				        EFeatureClass::C_STREAMING_DENSE &&
				    m_lhs->get_feature_class() <= EFeatureClass::C_STREAMING_VW)
					result = make_any(
					    1.0 /
					    static_cast<double>(
					        ((CDotFeatures*)m_lhs)->get_dim_feature_space()));
				else
					result = make_any(
					    1.0 /
					    (static_cast<double>(
					         ((CDotFeatures*)m_lhs)->get_dim_feature_space()) *
					     ((CDenseFeatures<float64_t>*)(m_lhs))
					         ->std(false)
					         .get_element(0)));
				return result;
			}

		private:
			CKernel* m_kernel;
			SG_DELETE_COPY_AND_ASSIGN(GammaFeatureNumberInit);
		};
	} // namespace factory
} // namespace shogun

#endif // __AUTO_INIT_FACTORY_H__