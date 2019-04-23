/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef _SCATTERKERNELNORMALIZER_H___
#define _SCATTERKERNELNORMALIZER_H___

#include <shogun/lib/config.h>

#include <shogun/kernel/normalizer/KernelNormalizer.h>
#include <shogun/kernel/normalizer/IdentityKernelNormalizer.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
/** @brief the scatter kernel normalizer */
class ScatterKernelNormalizer: public KernelNormalizer
{

public:
	/** default constructor  */
	ScatterKernelNormalizer() : KernelNormalizer()
	{
		init();
	}

	/** default constructor
	 */
	ScatterKernelNormalizer(float64_t const_diag, float64_t const_offdiag,
			std::shared_ptr<Labels> labels,std::shared_ptr<KernelNormalizer> normalizer=NULL)
		: KernelNormalizer()
	{
		init();

		m_testing_class=-1;
		m_const_diag=const_diag;
		m_const_offdiag=const_offdiag;

		ASSERT(labels)

		ASSERT(labels->get_label_type()==LT_MULTICLASS)
		m_labels = labels->as<MulticlassLabels>();
		labels->ensure_valid();

		if (!normalizer)
			normalizer=std::make_shared<IdentityKernelNormalizer>();

		m_normalizer=normalizer;

		SG_DEBUG("Constructing ScatterKernelNormalizer with const_diag={:g}"
				" const_offdiag={:g} num_labels={} and normalizer='{}'",
				const_diag, const_offdiag, labels->get_num_labels(),
				normalizer->get_name());
	}

	/** default destructor */
	virtual ~ScatterKernelNormalizer()
	{


	}

	/** initialization of the normalizer
	 * @param k kernel */
	virtual bool init(Kernel* k)
	{
		m_normalizer->init(k);
		return true;
	}

	/** get testing class
	 *
	 * @return testing class (-1 disabled, 0...class otherwise)
	 */
	int32_t get_testing_class() const noexcept
	{
		return m_testing_class;
	}

	/** set testing status
	 *
	 * @param c set class to test for
	 */
	void set_testing_class(int32_t c)
	{
		m_testing_class=c;
	}

	/** normalize the kernel value
	 * @param value kernel value
	 * @param idx_lhs index of left hand side vector
	 * @param idx_rhs index of right hand side vector
	 */
	virtual float64_t normalize(float64_t value, int32_t idx_lhs,
			int32_t idx_rhs) const
	{
		value=m_normalizer->normalize(value, idx_lhs, idx_rhs);
		float64_t c=m_const_offdiag;

		if (m_testing_class>=0)
		{
			if (m_labels->get_label(idx_lhs) == m_testing_class)
				c=m_const_diag;
		}
		else
		{
			if (m_labels->get_label(idx_lhs) == m_labels->get_label(idx_rhs))
				c=m_const_diag;

		}
		return value*c;
	}

	/** normalize only the left hand side vector
	 * @param value value of a component of the left hand side feature vector
	 * @param idx_lhs index of left hand side vector
	 */
	virtual float64_t normalize_lhs(float64_t value, int32_t idx_lhs) const
	{
		error("normalize_lhs not implemented");
		return 0;
	}

	/** normalize only the right hand side vector
	 * @param value value of a component of the right hand side feature vector
	 * @param idx_rhs index of right hand side vector
	 */
	virtual float64_t normalize_rhs(float64_t value, int32_t idx_rhs) const
	{
		error("normalize_rhs not implemented");
		return 0;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "ScatterKernelNormalizer";
	}

private:
	void init()
	{
		m_const_diag = 1.0;
		m_const_offdiag = 1.0;

		m_labels = NULL;
		m_normalizer = NULL;

		m_testing_class = -1;

		SG_ADD(&m_testing_class, "m_testing_class",
				"Testing Class.");
		SG_ADD(&m_const_diag, "m_const_diag",
				"Factor to multiply to diagonal elements.", ParameterProperties::HYPER);
		SG_ADD(&m_const_offdiag, "m_const_offdiag",
				"Factor to multiply to off-diagonal elements.", ParameterProperties::HYPER);

		SG_ADD((std::shared_ptr<SGObject>*) &m_labels, "m_labels", "Labels");
		SG_ADD((std::shared_ptr<SGObject>*) &m_normalizer, "m_normalizer", "Kernel normalizer.",
		    ParameterProperties::HYPER);
	}

protected:

	/** factor to multiply to diagonal elements */
	float64_t m_const_diag;
	/** factor to multiply to off-diagonal elements */
	float64_t m_const_offdiag;

	/** labels **/
	std::shared_ptr<MulticlassLabels> m_labels;

	/** labels **/
	std::shared_ptr<KernelNormalizer> m_normalizer;

	/** upon testing which class to test for */
	int32_t m_testing_class;
};
}
#endif

