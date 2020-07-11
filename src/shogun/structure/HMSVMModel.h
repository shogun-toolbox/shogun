/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Shell Hu, Heiko Strathmann, Yuyu Zhang
 */

#ifndef _HMSVM_MODEL__H__
#define _HMSVM_MODEL__H__

#include <shogun/lib/config.h>

#include <shogun/structure/StructuredModel.h>
#include <shogun/structure/SequenceLabels.h>
#include <shogun/structure/StateModelTypes.h>
#include <shogun/structure/StateModel.h>

namespace shogun
{

enum EStateModelType;

/**
 * @brief Class CHMSVMModel that represents the application specific model
 * and contains the application dependent logic to solve Hidden Markov Support
 * Vector Machines (HM-SVM) type of problems within a generic SO framework.
 */
class HMSVMModel : public StructuredModel
{
	public:
		/** default constructor */
		HMSVMModel();

		/** constructor
		 *
		 * @param features the feature vectors, must be of type MatrixFeatures
		 * @param labels sequence labels
		 * @param smt internal state representation
		 * @param num_obs number of observations
		 * @param use_plifs whether to model the observations using PLiFs
		 */
		HMSVMModel(std::shared_ptr<Features> features, std::shared_ptr<StructuredLabels> labels, EStateModelType smt, int32_t num_obs=0, bool use_plifs=false);

		/** destructor */
		~HMSVMModel() override;

		/**
		 * return the dimensionality of the joint feature space, i.e.
		 * the dimension of the weight vector \f$w\f$
		 */
		int32_t get_dim() const override;

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
		SGVector< float64_t > get_joint_feature_vector(int32_t feat_idx, std::shared_ptr<StructuredData> y) override;

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
		std::shared_ptr<ResultSet> argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training = true) override;

		/** computes \f$ \Delta(y_{1}, y_{2}) \f$
		 *
		 * @param y1 an instance of structured data
		 * @param y2 another instance of structured data
		 *
		 * @return loss value
		 */
		float64_t delta_loss(std::shared_ptr<StructuredData> y1, std::shared_ptr<StructuredData> y2) override;

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
		void init_primal_opt(
				float64_t regularization,
				SGMatrix< float64_t > & A,  SGVector< float64_t > a,
				SGMatrix< float64_t > B,  SGVector< float64_t > & b,
				SGVector< float64_t > & lb, SGVector< float64_t > & ub,
				SGMatrix < float64_t > & C) override;

		/**
		 * method to be called from a SO machine before training
		 * to ensure that the training data is valid
		 */
		bool check_training_setup() const override;

		/**
		 * get the number of auxiliary variables to introduce in the
		 * optimization problem. The auxiliary variables are used to
		 * implement smoothness regularization between adjacent emission
		 * scores via constraints.
		 *
		 * @return the number of auxiliary variables
		 */
		int32_t get_num_aux() const override;

		/**
		 * get the number of auxiliary constraints to introduce in the
		 * optimization problem. These constraints are used to implement
		 * smoothness regularization between adjacent emission scores.
		 *
		 * @return the number of auxiliary constraints
		 */
		int32_t get_num_aux_con() const override;

		/** setter for use_plifs
		 *
		 * @param use_plifs whether PLiFs shall be used
		 */
		void set_use_plifs(bool use_plifs);

		/**
		 * initializes the emission and transmission vectors of weights used in Viterbi
		 * decoding. In case PLiFs are used, it also initializes the matrix of PLiFs and
		 * automatically selects the supporting points based on the feature values
		 */
		void init_training() override;

		/** get transmission weights
		 *
		 * @return vector with the transmission weights
		 */
		SGMatrix< float64_t > get_transmission_weights() const;

		/** get emission weights
		 *
		 * @return vector with the emission weights
		 */
		SGVector< float64_t > get_emission_weights() const;

		/** get state model
		 *
		 * @return model with the description of the states
		 */
		std::shared_ptr<StateModel> get_state_model() const;

		/** return the SGSerializable's name
		 *
		 * @return name Gaussian
		 */
		const char* get_name() const override { return "HMSVMModel"; }

	private:
		/* internal initialization */
		void init();

	private:
		/** in case of discrete observations, the cardinality of the space of observations */
		int32_t m_num_obs;

		/** the number of auxiliary variables */
		int32_t m_num_aux;

		/** the state model */
		std::shared_ptr<StateModel> m_state_model;

		/** the state model type */
		EStateModelType m_state_model_type;

		/** transition weights used in Viterbi */
		SGMatrix< float64_t > m_transmission_weights;

		/** emission weights used in Viterbi */
		SGVector< float64_t > m_emission_weights;

		/** number of supporting points for each PLiF */
		int32_t m_num_plif_nodes;

		/** PLiF matrix of dimensions (num_states, num_features) */
		std::vector<std::shared_ptr<Plif>> m_plif_matrix;

		/** whether to use PLiFs. Otherwise, the observations must be discrete and finite */
		bool m_use_plifs;
}; /* class CHMSVMModel */

} /* namespace shogun */

#endif /* _HMSVM_MODEL__H__ */
