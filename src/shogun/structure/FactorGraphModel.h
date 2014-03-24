/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#ifndef __FACTOR_GRAPH_MODEL_H__
#define __FACTOR_GRAPH_MODEL_H__

#include <shogun/lib/config.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/structure/StructuredModel.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/MAPInference.h>

namespace shogun
{

/** @brief CFactorGraphModel defines a model in terms of CFactorGraph
 * and CMAPInference, where parameters are associated with factor types,
 * in the model. There is a mapping vector records the locations of
 * local factor parameters in the global parameter vector.
 *
 * TODO: implement functions for SGD
 */
class CFactorGraphModel : public CStructuredModel
{
public:
	/** constructor */
	CFactorGraphModel();

	/** constructor
	 *
	 * @param features pointer to structured inputs
	 * @param labels pointer to structured outputs
	 * @param inf_type MAP inference type, default is tree max-product inference
	 * @param verbose whether output verbose information, such as energy table, slack variables etc.
	 * NOTE: do NOT set this up when training with large data, massive printing will crash the program
	 */
	CFactorGraphModel(CFeatures* features, CStructuredLabels* labels,
		EMAPInferType inf_type = TREE_MAX_PROD, bool verbose = false);

	/** destructor */
	~CFactorGraphModel();

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "FactorGraphModel"; }

	/** add a new factor type, NOTE: a factor type is not allowed to change
	 * once it has been added to the FactorGraphModel. Secondly, the model itself
	 * should not be modified during training, i.e. no add/delete operations.
	 *
	 * @param ftype pointer to new factor type
	 */
	void add_factor_type(CFactorType* ftype);

	/** delete a factor type
	 *
	 * @param ftype_id factor type id
	 */
	void del_factor_type(const int32_t ftype_id);

	/** @return pointer to the array of factor types */
	CDynamicObjectArray* get_factor_types() const;

	/** get a factor type specified by its id
	 *
	 * @param ftype_id factor type id
	 */
	CFactorType* get_factor_type(const int32_t ftype_id) const;

	/** @return parameter mapping for all factor types */
	SGVector<int32_t> get_global_params_mapping() const;

	/** get parameter mapping for a factor type
	 *
	 * @param ftype_id factor type id
	 */
	SGVector<int32_t> get_params_mapping(const int32_t ftype_id);

	/** @return concatenated parameter vector from local parameters */
	SGVector<float64_t> fparams_to_w();

	/** update local parameters
	 *
	 * @param w new global parameter vector
	 */
	void w_to_fparams(SGVector<float64_t> w);

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
	virtual CResultSet* argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training = true);

	/** computes \f$ \Delta(y_{1}, y_{2}) \f$
	 *
	 * @param y1 an instance of structured data
	 * @param y2 another instance of structured data
	 *
	 * @return loss value
	 */
	virtual float64_t delta_loss(CStructuredData* y1, CStructuredData* y2);

	/** initializes the part of the model that needs to be used during training.
	 * In this class this method is empty and it can be re-implemented for any
	 * particular StructuredModel
	 */
	virtual void init_training();

	/** initialize the optimization problem for primal solver
	 *
	 * @param regularization input for C
	 * @param A  is [-dPsi(y) | -I_N ] with M+N columns => max. M+1 nnz per row
	 * @param a  unused input
	 * @param B  unused input
	 * @param b  upper bounds of the constraints, Ax <= b
	 * @param lb lower bounds for w
	 * @param ub upper bounds for w
	 * @param C  regularization matrix, w'Cw
	 */
	virtual void init_primal_opt(
			float64_t regularization,
			SGMatrix< float64_t > & A,  SGVector< float64_t > a,
			SGMatrix< float64_t > B,  SGVector< float64_t > & b,
			SGVector< float64_t > lb, SGVector< float64_t > ub,
			SGMatrix < float64_t >  & C);

	/**
	 * return the dimensionality of the joint feature space, i.e.
	 * the dimension of the weight vector \f$w\f$
	 */
	virtual int32_t get_dim() const;

private:
	/** register and initialize parameters */
	void init();

protected:
	/** array of factor types */
	CDynamicObjectArray* m_factor_types;

	/** index of factor type */
	SGVector<int32_t> m_w_map;

	/** cache of global parameters */
	SGVector<float64_t> m_w_cache;

	/** MAP inference type */
	EMAPInferType m_inf_type;

	/** whether print verbose information */
	bool m_verbose;
};

}

#endif

