/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Roman Votyakov, Yuyu Zhang, Abinash Panda
 */

#ifndef __STRUCTURED_ACCURACY_H__
#define __STRUCTURED_ACCURACY_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/Evaluation.h>
#include <shogun/labels/StructuredLabels.h>

namespace shogun
{

/**
 * @brief class StructuredAccuracy used to compute accuracy of structured classification
 */
class StructuredAccuracy : public Evaluation
{
public:
	/** default constructor */
	StructuredAccuracy();

	/** destructor */
	virtual ~StructuredAccuracy();

	/** evaluate accuracy
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return accuracy
	 */
	virtual float64_t evaluate(std::shared_ptr<Labels > predicted, std::shared_ptr<Labels > ground_truth);

	/** NOT IMPLEMENTED
	 * constructs confusion matrix for multiclass classification
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return confusion matrix
	 */
	static SGMatrix<int32_t> get_confusion_matrix(const std::shared_ptr<Labels >& predicted, const std::shared_ptr<Labels >& ground_truth);

	/** whether the evaluation criterion has to be maximimed or minimized
	*
	* @return maximize evaluation criterion
	*/
	inline EEvaluationDirection get_evaluation_direction() const
	{
		return ED_MAXIMIZE;
	}

	/** @return name of SGSerializable */
	virtual const char * get_name() const
	{
		return "StructuredAccuracy";
	}

private:
	/** evaluate accuracy for structured labels composed of real numbers
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return accuracy
	 */
	float64_t evaluate_real(const std::shared_ptr<StructuredLabels >& predicted, const std::shared_ptr<StructuredLabels >& ground_truth);

	/** evaluate accuracy for structured labels composed of sequences
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return accuracy
	 */
	float64_t evaluate_sequence(const std::shared_ptr<StructuredLabels >& predicted, const std::shared_ptr<StructuredLabels >& ground_truth);

	/** evaluate accuracy for structured labels composed of sparse multi
	 * labels. Formally the accuracy is defined as
	 *
	 * \f[
	 *      $\frac{1}{p}\sum_{i=1}^{p}\frac{ |Y_i \cap h(x_i)|}{|Y_i \cup
	 *      h(x_i)|}$
	 * \f]
	 *
	 * @param predicted labels to be evaluated
	 * @param ground_truth labels assumed to be correct
	 *
	 * @return accuracy
	 */
	float64_t evaluate_sparse_multilabel(const std::shared_ptr<StructuredLabels >& predicted,
	                                     const std::shared_ptr<StructuredLabels >& ground_truth);

}; /* class StructuredAccuracy*/

} /* namespace shogun */

#endif /* __STRUCTURED_ACCURACY_H__ */
