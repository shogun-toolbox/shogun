/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Thoralf Klein
 * Written(W) 2014 Thoralf Klein
 * Written(W) 2014 Abinash Panda
 */

#ifndef _MULTILABEL_CLR_MODEL__H__
#define _MULTILABEL_CLR_MODEL__H__

#include <shogun/lib/config.h>
#include <shogun/structure/StructuredModel.h>

namespace shogun
{

/** @brief Class MultilabelCLRModel represents application specific model and
 * contains application dependent logic for solving multi-label classification
 * using Calibrated Label Ranking (CLR) [1] method within a generic SO framework.
 *
 * [1] J. Furnkranz, et.al. Multilabel Classification via Calibrated Label Ranking.
 *     https://www.mathematik.uni-marburg.de/~eyke/publications/ml08.pdf
 */
class MultilabelCLRModel : public StructuredModel
{
public:
	/** default constructor */
	MultilabelCLRModel();

	/** constructor
	 *
	 * @param features features
	 * @param labels structured labels
	 */
	MultilabelCLRModel(std::shared_ptr<Features > features, std::shared_ptr<StructuredLabels > labels);

	/** destructor */
	~MultilabelCLRModel() override;

	/** create empty StructuredLabels object */
	std::shared_ptr<StructuredLabels > structured_labels_factory(int32_t num_labels = 0) override;

	/** return the dimensionality of the joint feature space, i.e., the
	 * dimension of the weight vector \f$w\f$.
	 */
	int32_t get_dim() const override;

	/** get joint feature vector
	 *
	 * \f[
	 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y})
	 * \f]
	 *
	 * @param feat_idx index of the feature vector to use
	 * @param y structured label to use
	 */
	SGVector<float64_t> get_joint_feature_vector(int32_t feat_idx,
	                std::shared_ptr<StructuredData > y) override;

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
	std::shared_ptr<ResultSet > argmax(SGVector<float64_t> w, int32_t feat_idx,
	                            bool const training = true) override;

	/** computes \f$ \Delta(y_{1}, y_{2}) \f$
	 *
	 * @param y1 an instance of structured data
	 * @param y2 another instance of structured data
	 *
	 * @return loss value
	 */
	float64_t delta_loss(std::shared_ptr<StructuredData > y1, std::shared_ptr<StructuredData > y2) override;

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
	void init_primal_opt(
	        float64_t regularization,
	        SGMatrix<float64_t> &A,
	        SGVector<float64_t> a,
	        SGMatrix<float64_t> B,
	        SGVector<float64_t> &b,
	        SGVector<float64_t> &lb,
	        SGVector<float64_t> &ub,
	        SGMatrix<float64_t> &C) override;

	/** @return name of the SGSerializable */
	const char * get_name() const override
	{
		return "MultilabelCLRModel";
	}

private:
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

}; /* class CMultilabelCLRModel */

} /* namespace shogun */

#endif /* _MULTILABEL_CLR_MODEL__H__ */

