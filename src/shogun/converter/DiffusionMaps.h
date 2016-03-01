/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef DIFFUSIONMAPS_H_
#define DIFFUSIONMAPS_H_
#include <shogun/lib/config.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief class DiffusionMaps used to preprocess given data
 * using Diffusion Maps dimensionality
 * reduction technique as described in
 *
 * Coifman, R., & Lafon, S. (2006).
 * Diffusion maps.
 * Applied and Computational Harmonic Analysis, 21(1), 5-30. Elsevier.
 * Retrieved from http://linkinghub.elsevier.com/retrieve/pii/S1063520306000546
 *
 * Uses implementation from the Tapkee library.
 *
 * To use this converter with static interfaces please refer it by
 * sg('create_converter','diffusion_maps',t,width);
 *
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

	/** embed distance
	 * @param distance to use for embedding
	 * @return embedding simple features
	 */
	virtual CDenseFeatures<float64_t>* embed_distance(CDistance* distance);

	/** setter for t parameter
	 * @param t t value
	 */
	void set_t(int32_t t);

	/** getter for t parameter
	 * @return t value
	 */
	int32_t get_t() const;

	/** setter for width parameter
	 * @param width width value
	 */
	void set_width(float64_t width);

	/** getter for width parameter
	 * @return width value
	 */
	float64_t get_width() const;

	/** get name */
	virtual const char* get_name() const;

protected:

	/** default init */
	void init();

protected:

	/** number of steps */
	int32_t m_t;

	/** gaussian kernel width */
	float64_t m_width;

};
}

#endif /* DIFFUSIONMAPS_H_ */
