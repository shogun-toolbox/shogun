/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann,
 *          Evgeniy Andreev, Yuyu Zhang, Chiyuan Zhang, Thoralf Klein,
 *          Evan Shelhamer, Youssef Emad El-Din, Bjoern Esser, Sanuj Sharma,
 *          Saurabh Goyal
 */

#ifndef _LINEARCLASSIFIER_H__
#define _LINEARCLASSIFIER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DotFeatures.h>


namespace shogun
{

class BinaryLabels;
class DotFeatures;
class Features;
class RegressionLabels;

/** @brief Class LinearMachine is a generic interface for all kinds of linear
 * machines like classifiers.
 *
 * A linear classifier computes
 *
 *  \f[
 *		f({\bf x})= {\bf w} \cdot {\bf x} + b
 *	\f]
 *
 * where \f${\bf w}\f$ are the weights assigned to each feature in training
 * and \f$b\f$ the bias.
 *
 * To implement a linear classifier all that is required is to define the
 * train() function that delivers \f${\bf w}\f$ above.
 *
 * Note that this framework works with linear classifiers of arbitraty feature
 * type, e.g. dense and sparse and even string based features. This is
 * implemented by using DotFeatures that may provide a mapping function
 * \f$\Phi({\bf x})\mapsto {\cal R^D}\f$ encapsulating all the required
 * operations (like the dot product). The decision function is thus
 *
 *  \f[
 *		f({\bf x})= {\bf w} \cdot \Phi({\bf x}) + b.
 *	\f]
 *
 *	The following linear classifiers are implemented
 *	\li Linear Descriminant Analysis (CLDA)
 *	\li Linear Programming Machines (CLPM, CLPBoost)
 *	\li Perceptron (Perceptron)
 *	\li Linear SVMs (SVMSGD, LibLinear, SVMOcas, SVMLin, CSubgradientSVM)
 *
 *	\sa DotFeatures
 *
 * */
class LinearMachine : public Machine
{
	public:
		/** default constructor */
		LinearMachine();

		/** destructor */
		virtual ~LinearMachine();

		/** copy constructor */
		LinearMachine(const std::shared_ptr<LinearMachine>& machine);

		/** get w
		 *
		 * @return weight vector
		 */
		virtual SGVector<float64_t> get_w() const;

		/** set w
		 *
		 * @param src_w new w
		 */
		virtual void set_w(const SGVector<float64_t> src_w);

		/** set bias
		 *
		 * @param b new bias
		 */
		virtual void set_bias(float64_t b);

		/** get bias
		 *
		 * @return bias
		 */
		virtual float64_t get_bias() const;

		/** apply linear machine to data
		 * for binary classification problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual std::shared_ptr<BinaryLabels>
		apply_binary(std::shared_ptr<Features> data);

		/** apply linear machine to data
		 * for regression problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual std::shared_ptr<RegressionLabels>
		apply_regression(std::shared_ptr<Features> data);

		/** applies to one vector */
		virtual float64_t apply_one(
		    const std::shared_ptr<DotFeatures>& features, int32_t vec_idx);

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "LinearMachine"; }

	protected:

		/** apply get outputs
		 *
		 * @param data features to compute outputs
		 * @return outputs
		 */
		virtual SGVector<float64_t> apply_get_outputs(std::shared_ptr<Features> data);

		bool train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs) final
		{
			const auto dot_feat = data->as<DotFeatures>();
			return train_machine(dot_feat, labs);
		}

		virtual bool train_machine(const std::shared_ptr<DotFeatures>& data, const std::shared_ptr<Labels>& labs)
		{
			not_implemented(SOURCE_LOCATION);
			return false;
		}
	private:

		void init();

	protected:
		/** w */
		SGVector<float64_t> m_w;

		/** bias */
		float64_t bias = 0.0;
};
}
#endif
