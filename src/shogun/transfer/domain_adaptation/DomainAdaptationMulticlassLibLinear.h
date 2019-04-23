/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Bjoern Esser
 */

#ifndef _DOMAINADAPTATIONMULTICLASSLIBLINEAR_H___
#define _DOMAINADAPTATIONMULTICLASSLIBLINEAR_H___
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/multiclass/MulticlassLibLinear.h>

namespace shogun
{

/** @brief domain adaptation multiclass LibLinear wrapper
 * Source domain is assumed to b
 */
class DomainAdaptationMulticlassLibLinear : public MulticlassLibLinear
{
	public:
		/** default constructor  */
		DomainAdaptationMulticlassLibLinear();

		/** standard constructor
		 * @param target_C C regularization constant value for target domain
		 * @param target_features target domain features
		 * @param target_labels target domain labels
		 * @param source_machine source domain machine to regularize against
		 */
		DomainAdaptationMulticlassLibLinear(float64_t target_C,
				std::shared_ptr<DotFeatures> target_features, std::shared_ptr<Labels> target_labels,
				std::shared_ptr<LinearMulticlassMachine> source_machine);

		/** destructor */
		virtual ~DomainAdaptationMulticlassLibLinear();

		/** get submachine outputs */
		virtual std::shared_ptr<BinaryLabels> get_submachine_outputs(int32_t);

		/** get name */
		virtual const char* get_name() const
		{
			return "DomainAdaptationMulticlassLibLinear";
		}

		/** getter for source bias
		 * @return source bias
		 */
		float64_t get_source_bias() const;
		/** setter for source bias
		 * @param source_bias source bias
		 */
		void set_source_bias(float64_t source_bias);

		/** getter for train factor
		 * @return train factor
		 */
		float64_t get_train_factor() const;
		/** setter for train factor
		 * @param train_factor train factor
		 */
		void set_train_factor(float64_t train_factor);

		/** getter for source machine
		 * @return source machine
		 */
		std::shared_ptr<LinearMulticlassMachine> get_source_machine() const;
		/** setter for source machine
		 * @param source_machine source machine
		 */
		void set_source_machine(std::shared_ptr<LinearMulticlassMachine> source_machine);

protected:

		/** obtain regularizer (w0) matrix */
		virtual SGMatrix<float64_t> obtain_regularizer_matrix() const;

private:

		/** init defaults */
		void init_defaults();

		/** register parameters */
		void register_parameters();

protected:

		/** train factor */
		float64_t m_train_factor;

		/** source bias */
		float64_t m_source_bias;

		/** source domain machine */
		std::shared_ptr<LinearMulticlassMachine> m_source_machine;
};
}
#endif /* HAVE_LAPACK */
#endif
