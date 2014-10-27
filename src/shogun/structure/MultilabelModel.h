/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (C) 2014 Abinash Panda
 * Written (W) 2014 Abinash Panda
 */

#ifndef _MULTILABEL_MODEL__H__
#define _MULTILABEL_MODEL__H__

#include <shogun/lib/config.h>
#include <shogun/structure/StructuredModel.h>

namespace shogun
{

/** @brief Class CMultilabelModel represents application specific model and
 * contains application dependent logic for solving multilabel classification
 * within a generic SO framework.
 *
 * [1] C. Lampert. Maximum Margin Multi-Label Structured Prediction, NIPS 2011.
 * http://machinelearning.wustl.edu/mlpapers/paper_files/NIPS2011_0207.pdf
 */
class CMultilabelModel : public CStructuredModel
{
public:
	/** default constructor */
	CMultilabelModel();

	/** constructor
	 *
	 * @param features features
	 * @param labels structured labels
	 */
	CMultilabelModel(CFeatures * features, CStructuredLabels * labels);

	/** destructor */
	virtual ~CMultilabelModel();


	/** create empty StructuredLabels object */
	virtual CStructuredLabels * structured_labels_factory(int32_t num_labels = 0);

	/** return the dimensionality of the joint feature space, i.e., the
	 * dimension of the weight vector \f$w\f$.
	 */
	virtual int32_t get_dim() const;

	/** get joint feature vector
	 *
	 * \f[
	 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y})
	 * \f]
	 *
	 * @param feat_idx index of the feature vector to use
	 * @param y structured label to use
	 */
	virtual SGVector<float64_t> get_joint_feature_vector(int32_t feat_idx,
	                CStructuredData * y);

	/** obtain the argmax of \f$ \Delta(y_{pred}, y_{truth}) + \langle w,
	 * \Psi(x_{truth}, y_{pred}) \rangle \f$
	 *
	 * @param w weight vector
	 * @param feat_idx index of the feature to compute the argmax
	 * @param training true if argmax is called during training
	 * Then, it is assumed that the label indexed by feat_idx in m_labels
	 * corresponds to the true label of the corresponding feature vector
	 *
	 * @return structure with the predicted output
	 */
	virtual CResultSet * argmax(SGVector<float64_t> w, int32_t feat_idx,
	                            bool const training = true);

	/** computes \f$ \Delta(y_{1}, y_{2}) \f$
	 *
	 * @param y1 an instance of structured data
	 * @param y2 another instance of structured data
	 *
	 * @return loss value
	 */
	virtual float64_t delta_loss(CStructuredData * y1, CStructuredData * y2);

	/** set misclassification cost for false positive and
	 * false negative
	 *
	 * @param false_positive cost for false positive
	 * @param false_negative cost for false negative
	 */
	virtual void set_misclass_cost(float64_t false_positive,
	                               float64_t false_negative);

	/** initialize the optimization problem
	 *
	 * @param regularization regularization strength
	 * @param A is [-dPsi(y) | -I_N ] with M+N columns => max, M+1 nnz per row
	 * @param a unused input
	 * @param B unused input
	 * @param b upper bound of the constraints, Ax <= b
	 * @param lb lower bounds for w
	 * @param ub upper bounds for w
	 * @param C regularization matrix, w'Cw
	 */
	virtual void init_primal_opt(
	        float64_t regularization,
	        SGMatrix<float64_t> &A,
	        SGVector<float64_t> a,
	        SGMatrix<float64_t> B,
	        SGVector<float64_t> &b,
	        SGVector<float64_t> &lb,
	        SGVector<float64_t> &ub,
	        SGMatrix<float64_t> &C);

	/** @return name of the SGSerializable */
	virtual const char * get_name() const
	{
		return "MultilabelModel";
	}

private:
	float64_t m_false_positive;
	float64_t m_false_negative;
	int32_t m_num_classes;

private:
	void init();

	/** different versions of delta loss function */
	float64_t delta_loss(SGVector<float64_t> y1, SGVector<float64_t> y2);
	float64_t delta_loss(float64_t y1, float64_t y2);

	/** convert dense vector to sparse
	 * dense vector would be in the form of {d_true, d_false}^dense_dim
	 * sparse vector would contain the indices where the value of
	 * dense_vector is d_true
	 */
	SGVector<int32_t> to_sparse(SGVector<float64_t> dense_vector,
	                            float64_t d_true, float64_t d_false);

}; /* class CMultilabelModel */

} /* namespace shogun */

#endif /* _MULTILABEL_MODEL__H__ */



