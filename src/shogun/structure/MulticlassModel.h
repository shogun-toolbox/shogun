/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shell Hu, Yuyu Zhang, Thoralf Klein, 
 *          Sergey Lisitsyn, Bjoern Esser, Viktor Gal
 */

#ifndef _MULTICLASS_MODEL__H__
#define _MULTICLASS_MODEL__H__

#include <shogun/lib/config.h>

#include <shogun/structure/StructuredModel.h>

namespace shogun
{

/**
 * @brief Class CMulticlassModel that represents the application specific model
 * and contains the application dependent logic to solve multiclass
 * classification within a generic SO framework.
 */
class MulticlassModel : public StructuredModel
{

	public:
		/** default constructor */
		MulticlassModel();

		/** constructor
		 *
		 * @param features
		 * @param labels
		 */
		MulticlassModel(std::shared_ptr<Features> features, std::shared_ptr<StructuredLabels> labels);

		/** destructor */
		virtual ~MulticlassModel();

		/** create empty StructuredLabels object */
		virtual std::shared_ptr<StructuredLabels> structured_labels_factory(int32_t num_labels=0);

		/**
		 * return the dimensionality of the joint feature space, i.e.
		 * the dimension of the weight vector \f$w\f$
		 */
		virtual int32_t get_dim() const;

		/**
		 * get joint feature vector
		 *
		 * \f[
		 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y})
		 * \f]
		 *
		 * @param feat_idx index of the feature vector to use
		 * @param y structured label to use
		 *
		 * @return the joint feature vector
		 */
		virtual SGVector< float64_t > get_joint_feature_vector(int32_t feat_idx, std::shared_ptr<StructuredData> y);

		/**
		 * obtains the argmax of \f$ \Delta(y_{pred}, y_{truth}) +
		 * \langle w, \Psi(x_{truth}, y_{pred}) \rangle \f$
		 *
		 * @param w weight vector
		 * @param feat_idx index of the feature to compute the argmax
		 * @param training true if argmax is called during training.
		 * Then, it is assumed that the label indexed by feat_idx in
		 * m_labels corresponds to the true label of the corresponding
		 * feature vector.
		 *
		 * @return structure with the predicted output
		 */
		virtual std::shared_ptr<ResultSet> argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training = true);

		/** computes \f$ \Delta(y_{1}, y_{2}) \f$
		 *
		 * @param y1 an instance of structured data
		 * @param y2 another instance of structured data
		 *
		 * @return loss value
		 */
		virtual float64_t delta_loss(std::shared_ptr<StructuredData> y1, std::shared_ptr<StructuredData> y2);

		/** initialize the optimization problem
		 *
		 * @param regularization regularization strength
		 * @param A  is [-dPsi(y) | -I_N ] with M+N columns => max. M+1 nnz per row
		 * @param a
		 * @param B
		 * @param b rhs of the equality constraints
		 * @param b  upper bounds of the constraints, Ax <= b
		 * @param lb lower bound for the weight vector
		 * @param ub upper bound for the weight vector
		 * @param C  regularization matrix, w'Cw
		 */
		virtual void init_primal_opt(
				float64_t regularization,
				SGMatrix< float64_t > & A,  SGVector< float64_t > a,
				SGMatrix< float64_t > B,  SGVector< float64_t > & b,
				SGVector< float64_t > & lb, SGVector< float64_t > & ub,
				SGMatrix < float64_t > & C);

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "MulticlassModel"; }

	private:
		void init();

		/** Different flavours of the delta_loss that become handy */
		float64_t delta_loss(float64_t y1, float64_t y2);
		float64_t delta_loss(int32_t y1_idx, float64_t y2);

	private:
		/** number of classes */
		int32_t m_num_classes;

}; /* MulticlassModel */

} /* namespace shogun */

#endif /* _MULTICLASS_MODEL__H__ */
