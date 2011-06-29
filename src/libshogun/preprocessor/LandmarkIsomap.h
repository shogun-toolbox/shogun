/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LANDMARKISOMAP_H_
#define LANDMARKISOMAP_H_
#ifdef HAVE_LAPACK
#include "preprocessor/Isomap.h"
#include "preprocessor/LandmarkMDS.h"
#include "features/Features.h"
#include "distance/Distance.h"
#include "distance/CustomDistance.h"

namespace shogun
{

class CDistance;

/** @brief the class LandmarkIsomap
 *
 */
class CLandmarkIsomap: public CIsomap
{
public:

	/** constructor */
	CLandmarkIsomap(): CIsomap(), m_landmark_number(3) {};

	/** destructor */
	virtual ~CLandmarkIsomap() {};

	/** get name */
	virtual inline const char* get_name() const { return "LandmarkIsomap"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_LANDMARKISOMAP; };

	/** set number of landmarks */
	void set_landmark_number(int32_t num)
	{
		m_landmark_number = num;
	};

	/** get number of landmarks */
	int32_t get_landmark_number()
	{
		return m_landmark_number;
	};

protected:

	/** number of landmarks */
	int32_t m_landmark_number;

	/** mds embedding */
	virtual SGMatrix<float64_t> mds_embed(CDistance* distance)
	{
		CLandmarkMDS* landmark_mds = new CLandmarkMDS();
		landmark_mds->set_landmark_number(m_landmark_number);
		landmark_mds->set_target_dim(m_target_dim);
		SGMatrix<float64_t> embedding = landmark_mds->embed_by_distance(distance);
		delete landmark_mds;
		return embedding;
	};
};
}
#endif /* HAVE_LAPACK */
#endif /* LANDMARKISOMAP_H_ */
