/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Abinash Panda, Sergey Lisitsyn, 
 *          Soeren Sonnenburg, Shell Hu, Thoralf Klein, Michal Uricar, 
 *          Yuyu Zhang, Bjoern Esser, Viktor Gal
 */

#ifndef _STRUCTURED_MODEL__H__
#define _STRUCTURED_MODEL__H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>
#include <shogun/labels/StructuredLabels.h>

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/StructuredData.h>

namespace shogun
{

#define IGNORE_IN_CLASSLIST

/**
 * \struct TMultipleCPinfo
 * Multiple cutting plane models helper
 */
IGNORE_IN_CLASSLIST struct TMultipleCPinfo {
	/** standard constructor
	 *
	 * @param from where this portion of data starts
	 * @param N total number of examples in portion
	 */
	TMultipleCPinfo(uint32_t from, uint32_t N) : m_from(from), m_N(N) {  }
	/** where this portion of data starts */
	uint32_t m_from;
	/** how many examples belong to this portion of data */
	uint32_t m_N;
};

class StructuredModel;

/** output of the argmax function */
struct ResultSet : public SGObject
{
	/** constructor */
	ResultSet();

	/** destructor */
	virtual ~ResultSet();

	/** @return name of SGSerializable */
	virtual const char* get_name() const;

	/** argmax */
	std::shared_ptr<StructuredData> argmax;

	/** whether joint feature vector is sparse or not */
	bool psi_computed_sparse;

	/** whether joint feature vector is dense or not */
	bool psi_computed;

	/** joint feature vector for the given truth */
	SGVector< float64_t > psi_truth;

	/** joint feature vector for the prediction */
	SGVector< float64_t > psi_pred;

	/** joint feature vector for the given truth */
	SGSparseVector< float64_t > psi_truth_sparse;

	/** joint feature vector for the prediction */
	SGSparseVector< float64_t > psi_pred_sparse;

	/** \f$ \Delta(y_{pred}, y_{truth}) + \langle w,
	 *  \Psi(x_{truth}, y_{pred}) - \Psi(x_{truth}, y_{truth}) \rangle \f$ */
	float64_t score;

	/** delta loss for the prediction vs. truth */
	float64_t delta;
};

/**
 * @brief Class StructuredModel that represents the application specific model
 * and contains most of the application dependent logic to solve structured
 * output (SO) problems. The idea of this class is to be instantiated giving
 * pointers to the functions that are dependent on the application, i.e. the
 * combined feature representation \f$\Psi(\bold{x},\bold{y})\f$ and the argmax
 * function \f$ {\arg\max} _{\bold{y} \neq \bold{y}_i} \left \langle { \bold{w},
 * \Psi(\bold{x}_i,\bold{y}) }  \right \rangle \f$. See: MulticlassModel.h and
 * .cpp for an example of these functions implemented.
 */
class StructuredModel : public SGObject
{
	public:
		/** default constructor */
		StructuredModel();

		/** constructor
		 *
		 * @param features the feature vectors
		 * @param labels structured labels
		 */
		StructuredModel(std::shared_ptr<Features> features, std::shared_ptr<StructuredLabels> labels);

		/** destructor */
		virtual ~StructuredModel();

		/** initialize the optimization problem for primal solver
		 *
		 * @param regularization regularization strength
		 * @param A  is [-dPsi(y) | -I_N ] with M+N columns => max. M+1 nnz per row
		 * @param a
		 * @param B
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
				SGMatrix < float64_t >  & C);

		/**
		 * return the dimensionality of the joint feature space, i.e.
		 * the dimension of the weight vector \f$w\f$
		 */
		virtual int32_t get_dim() const = 0;

		/** set labels
		 *
		 * @param labs labels
		 */
		void set_labels(std::shared_ptr<StructuredLabels> labs);

		/** get labels
		 *
		 * @return labels
		 */
		std::shared_ptr<StructuredLabels> get_labels();

		/** create empty StructuredLabels object */
		virtual std::shared_ptr<StructuredLabels> structured_labels_factory(int32_t num_labels=0);

		/** set features
		 *
		 * @param feats features
		 */
		void set_features(std::shared_ptr<Features> feats);

		/** get features
		 *
		 * @return features
		 */
		std::shared_ptr<Features> get_features();

		/**
		 * gets joint feature vector
		 *
		 * \f[
		 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y}_\text{lab\_idx})
		 * \f]
		 *
		 * @param feat_idx index of the feature vector to use
		 * @param lab_idx index of the structured label to use
		 *
		 * @return the joint feature vector
		 */
		SGVector< float64_t > get_joint_feature_vector(int32_t feat_idx, int32_t lab_idx);

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
		 * gets joint feature vector
		 *
		 * \f[
		 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y}_\text{lab\_idx})
		 * \f]
		 *
		 * @param feat_idx index of the feature vector to use
		 * @param lab_idx index of the structured label to use
		 *
		 * @return the joint feature vector
		 */
		SGSparseVector< float64_t > get_sparse_joint_feature_vector(int32_t feat_idx,
				int32_t lab_idx);

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
		virtual SGSparseVector< float64_t > get_sparse_joint_feature_vector(int32_t feat_idx,
				std::shared_ptr<StructuredData> y);

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
		virtual std::shared_ptr<ResultSet> argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training = true) = 0;

		/** computes \f$ \Delta(y_{\text{true}}, y_{\text{pred}}) \f$
		 *
		 * @param ytrue_idx index of the true label in labels
		 * @param ypred the predicted label
		 *
		 * @return loss value
		 */
		float64_t delta_loss(int32_t ytrue_idx, std::shared_ptr<StructuredData> ypred);

		/** computes \f$ \Delta(y_{1}, y_{2}) \f$
		 *
		 * @param y1 an instance of structured data
		 * @param y2 another instance of structured data
		 *
		 * @return loss value
		 */
		virtual float64_t delta_loss(std::shared_ptr<StructuredData> y1, std::shared_ptr<StructuredData> y2);

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "StructuredModel"; }

		/** initializes the part of the model that needs to be used during training.
		 * In this class this method is empty and it can be re-implemented for any
		 * particular StructuredModel
		 */
		virtual void init_training();

		/**
		 * method to be called from a SO machine before training
		 * to ensure that the training data is valid (e.g. check that
		 * there is at least one example for every class). In this class
		 * the method is empty and it can be re-implemented for any
		 * application (e.g. HM-SVM).
		 */
		virtual bool check_training_setup() const;

		/**
		 * get the number of auxiliary variables to introduce in the
		 * optimization problem. By default, this class do not impose
		 * the use of auxiliary variables and it will return zero.
		 * Re-implement this method subclasses to use auxiliary
		 * variables.
		 *
		 * return the number of auxiliary variables
		 */
		virtual int32_t get_num_aux() const;

		/**
		 * get the number of auxiliary constraints to introduce in the
		 * optimization problem. By default, this class do not impose
		 * the use of any auxiliary constraints and it will return zero.
		 * Re-implement this method in subclasses to use auxiliary
		 * constraints.
		 *
		 * return the number of auxiliary constraints
		 */
		virtual int32_t get_num_aux_con() const;

	private:
		/** internal initialization */
		void init();

	protected:
		/** structured labels */
		std::shared_ptr<StructuredLabels> m_labels;

		/** feature vectors */
		std::shared_ptr<Features> m_features;

}; /* class StructuredModel */

} /* namespace shogun */

#endif /* _STRUCTURED_MODEL__H__ */
