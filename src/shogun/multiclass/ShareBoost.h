/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef SHAREBOOST_H__
#define SHAREBOOST_H__

#include <machine/LinearMulticlassMachine.h>
#include <multiclass/MulticlassOneVsRestStrategy.h>
#include <features/DenseFeatures.h>
#include <labels/MulticlassLabels.h>

namespace shogun
{

/** ShareBoost is a linear multiclass algorithm that efficiently
 * learns a subset of features shared by all classes.
 *
 * See the following paper for details:
 *
 *     Shai Shalev-Shwartz, Yonatan Wexler, Amnon Shashua. ShareBoost: Efficient
 *     Multiclass Learning with Feature Sharing. NIPS 2011.
 */
class CShareBoost: public CLinearMulticlassMachine
{
public:
    /** default constructor */
	CShareBoost();

	/** constructor */
	CShareBoost(CDenseFeatures<float64_t> *features, CMulticlassLabels *labs, int32_t num_nonzero_feas);

    /** destructor */
	virtual ~CShareBoost() {}

    /** get name */
    virtual const char* get_name() const { return "ShareBoost"; }

	/** set number of non-zero features the algorithm should seek */
	void set_num_nonzero_feas(int32_t n) { m_nonzero_feas = n; }

	/** get number of non-zero features the algorithm should seek */
	int32_t get_num_nonzero_feas() const { return m_nonzero_feas; }

	/** assign features */
	void set_features(CFeatures *f);

	/** get active set */
	SGVector<int32_t> get_activeset();

	friend class ShareBoostOptimizer;
protected:

	/** train machine */
	virtual bool train_machine(CFeatures* data = NULL);

private:
	void init_sb_params(); ///< init machine parameters

	void compute_rho(); ///< compute the rho matrix
	int32_t choose_feature(); ///< choose next feature greedily
	void optimize_coefficients(); ///< optimize coefficients with gradient descent
	void compute_pred(); ///< compute predictions on training data, according to W in m_machines
	void compute_pred(const float64_t *W); ///< compute predictions on training data, according to given W

	int32_t m_nonzero_feas; ///< number of non-zero features to seek
	SGVector<int32_t> m_activeset; ///< selected features

	SGMatrix<float64_t> m_fea; ///< feature matrix used during training
	SGMatrix<float64_t> m_rho; ///< cache_matrix for rho
	SGVector<float64_t> m_rho_norm; ///< column sum of m_rho
	SGMatrix<float64_t> m_pred; ///< predictions, used in training
};

} /* shogun */

#endif /* end of include guard: SHAREBOOST_H__ */

