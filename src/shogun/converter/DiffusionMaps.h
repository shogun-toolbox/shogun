/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef DIFFUSIONMAPS_H_
#define DIFFUSIONMAPS_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief CDiffusionMaps used to preprocess given data using 
  * diffusion maps dimensionality reduction technique
  */
class CDiffusionMaps: public CEmbeddingConverter
{
public:

	/** constructor */
	CDiffusionMaps();

	/** destructor */
	virtual ~CDiffusionMaps();

	/** apply preprocessor to features
	 * @param features
	 */
	virtual CFeatures* apply(CFeatures* features);

	/** setter for t parameter
	 * @param t t value
	 */
	void set_t(int32_t t);

	/** getter for t parameter
	 * @return t value
	 */
	int32_t get_t() const;

	/** get name */
	virtual const char* get_name() const;

protected:

	/** default init */
	void init();

protected:

	/** number of steps */
	int32_t m_t;

};
}

#endif /* HAVE_LAPACK */
#endif /* DIFFUSIONMAPS_H_ */
