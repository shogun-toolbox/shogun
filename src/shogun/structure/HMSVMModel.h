/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _HMSVM_MODEL__H__
#define _HMSVM_MODEL__H__

#include <shogun/structure/StructuredModel.h>
#include <shogun/structure/HMSVMLabels.h>
#include <shogun/structure/StateModelTypes.h>
#include <shogun/structure/StateModel.h>

namespace shogun
{

enum EStateModelType;

/**
 * @brief Class CHMSVMModel TODO DOC
 */
class CHMSVMModel : public CStructuredModel
{
	public:
		/** default constructor */
		CHMSVMModel();

		/** constructor
		 *
		 * @param features the feature vectors, must be of type MatrixFeatures
		 * @param labels HMSVM labels
		 * @param num_obs number of observations
		 */
		CHMSVMModel(CFeatures* features, CStructuredLabels* labels, EStateModelType smt, int32_t num_obs);

		/** destructor */
		virtual ~CHMSVMModel();

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
		virtual SGVector< float64_t > get_joint_feature_vector(int32_t feat_idx, CStructuredData* y);

		/** obtains the argmax
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
		virtual CResultSet* argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training = true);

		/** computes \f$ \Delta(y_{1}, y_{2}) \f$
		 *
		 * @param y1 an instance of structured data
		 * @param y2 another instance of structured data
		 *
		 * @return loss value
		 */
		virtual float64_t delta_loss(CStructuredData* y1, CStructuredData* y2);

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
		virtual void init_opt(
				SGMatrix< float64_t > & A,  SGVector< float64_t > a,
				SGMatrix< float64_t > B,  SGVector< float64_t > & b,
				SGVector< float64_t > lb, SGVector< float64_t > ub,
				SGMatrix < float64_t > & C);

		/**
		 * method to be called from a SO machine before training
		 * to ensure that the training data is valid
		 */
		virtual bool check_training_setup() const;

		/**
		 * get the number of auxiliary variables to introduce in the
		 * optimization problem. The auxiliary variables are used to
		 * implement smoothness regularization between adjacent emission
		 * scores via constraints.
		 *
		 * return the number of auxiliary variables
		 */
		virtual int32_t get_num_aux() const;

		/**
		 * get the number of auxiliary constraints to introduce in the
		 * optimization problem. These constraints are used to implement
		 * smoothness regularization between adjacent emission scores.
		 *
		 * return the number of auxiliary constraints
		 */
		virtual int32_t get_num_aux_con() const;

	private:
		/* internal initialization */
		void init();

	private:
		/** the number of states */
		int32_t m_num_states;

		/** the number of observations */
		int32_t m_num_obs;

		/** the number of auxiliary variables */
		int32_t m_num_aux;

		/** the state model */
		CStateModel* m_state_model;

		/** transition weights used in Viterbi */
		SGMatrix< float64_t > m_transmission_weights;

		/** emission weights used in Viterbi */
		SGVector< float64_t > m_emission_weights;

}; /* class CHMSVMModel */

} /* namespace shogun */

#endif /* _HMSVM_MODEL__H__ */
