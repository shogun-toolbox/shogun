/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <lib/config.h>

#ifndef WAVEKERNEL_H_
#define WAVEKERNEL_H_

#include <lib/common.h>
#include <kernel/Kernel.h>
#include <distance/Distance.h>

namespace shogun
{

class CDistance;

/** @brief Wave kernel
 *
 * Formally described as
 *
 * \f[
 *		K(x,x') = \frac{\theta}{\| x-x' \|} \sin \frac{\| x-x' \|}{\theta}
 * \f]
 *
 */

class CWaveKernel: public CKernel
{
public:
	/** default constructor */
	CWaveKernel();

	/** constructor
	 * @param cache size of cache
	 * @param theta kernel parameter theta
	 * @param dist distance to be used
	 */
	CWaveKernel(int32_t cache, float64_t theta, CDistance* dist);

	/** constructor
	 * @param l features left-side
	 * @param r features right-side
	 * @param theta kernel parameter theta
	 * @param dist distance to be used
	 */
	CWaveKernel(CFeatures *l, CFeatures *r, float64_t theta, CDistance* dist);

	/** initialize kernel with features
	 * @param l features left-side
	 * @param r features right-side
	 * @return true if successful
	 */
	virtual bool init(CFeatures* l, CFeatures* r);

	/**
	 * @return kernel type
	 */
	virtual EKernelType get_kernel_type() { return K_WAVE; }

	/**
	 * @return type of features
	 */
	virtual EFeatureType get_feature_type() { return m_distance->get_feature_type(); }

	/**
	 * @return class of features
	 */
	virtual EFeatureClass get_feature_class() { return m_distance->get_feature_class(); }

	/**
	 * @return name of kernel
	 */
	virtual const char* get_name() const { return "WaveKernel"; }

	virtual ~CWaveKernel();

protected:
	/**
	 * compute kernel for specific feature vectors
	 * corresponding to [idx_a] of left-side and [idx_b] of right-side
	 * @param idx_a left-side index
	 * @param idx_b right-side index
	 * @return kernel value
	 */
	virtual float64_t compute(int32_t idx_a, int32_t idx_b);

private:
	void init();

protected:

	/// distance to be used
	CDistance* m_distance;

	/// theta parameter of kernel
	float64_t m_theta;
};
}

#endif /* WAVEKERNEL_H_ */
