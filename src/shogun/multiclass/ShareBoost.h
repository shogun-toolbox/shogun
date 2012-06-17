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

#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>

namespace shogun
{

class CShareBoost: public CLinearMulticlassMachine
{
public:
    /** default constructor */
	CShareBoost();

	/** constructor */
	CShareBoost(CDenseFeatures *features, CMulticlassLabels *labs);

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
protected:

	/** train machine */
	virtual bool train_machine(CFeatures* data = NULL);

private:
	init_sb_params();

	int32_t m_nonzero_feas;

	SGMatrix<float64_t> m_fea; ///< feature matrix used during training
	SGMatrix<float64_t> m_rho; ///< cache_matrix for rho
};

} /* shogun */ 

#endif /* end of include guard: SHAREBOOST_H__ */

