/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Shell Hu, Fernando Iglesias, Bjoern Esser,
 *          Soeren Sonnenburg, Viktor Gal
 */

#ifndef DIRECTOR_STRUCTURED_MODEL_H_
#define DIRECTOR_STRUCTURED_MODEL_H_

#include <shogun/lib/config.h>

#ifdef USE_SWIG_DIRECTORS
#include <shogun/structure/StructuredModel.h>

namespace shogun
{

class StructuredModel;

#define IGNORE_IN_CLASSLIST
/**
 * @brief Class DirectorStructuredModel that represents the application specific model
 * with structured output implemented in target interface language. It is a base class
 * that needs to be extended with real implementations before using.
 *
 * @see StructuredModel
 */
IGNORE_IN_CLASSLIST class DirectorStructuredModel : public StructuredModel
{
	public:
		/** default constructor */
		DirectorStructuredModel();

		/** destructor */
		virtual ~DirectorStructuredModel();

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

		/**
		 * method to be called from a SO machine before training
		 * to ensure that the training data is valid (e.g. check that
		 * there is at least one example for every class). In this class
		 * the method is empty and it can be re-implemented for any
		 * application (e.g. HM-SVM).
		 */
		virtual bool check_training_setup() const;

		/** initialize the optimization problem
		 *
		 * @param A
		 * @param a
		 * @param B
		 * @param b
		 * @param lb
		 * @param ub
		 * @param C
		 */
		virtual void init_primal_opt(
				float64_t regularization,
				SGMatrix< float64_t > & A,  SGVector< float64_t > a,
				SGMatrix< float64_t > B,  SGVector< float64_t > & b,
				SGVector< float64_t > & lb, SGVector< float64_t > & ub,
				SGMatrix < float64_t > & C);

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "DirectorStructuredModel"; }

		/** initializes the part of the model that needs to be used during training. */
		virtual void init_training();

}; /* class DirectorStructuredModel */
} /* namespace shogun */
#endif /* USE_SWIG_DIRECTORS */
#endif /* DIRECTOR_STRUCTURED_MODEL_H_ */
