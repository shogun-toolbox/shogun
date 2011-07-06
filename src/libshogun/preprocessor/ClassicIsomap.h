/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef CLASSICISOMAP_H_
#define CLASSICISOMAP_H_
#ifdef HAVE_LAPACK
#include "preprocessor/Isomap.h"
#include "preprocessor/ClassicMDS.h"
#include "features/Features.h"
#include "distance/Distance.h"
#include "distance/CustomDistance.h"

namespace shogun
{

class CDistance;

/** @brief the class ClassicIsomap used
 * to perform Isomap using ClassicMDS
 */
class CClassicIsomap: public CIsomap
{
public:

	/** constructor */
	CClassicIsomap(): CIsomap() {};

	/** destructor */
	virtual ~CClassicIsomap() {};

	/** get name */
	virtual inline const char* get_name() const { return "ClassicIsomap"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_CLASSICISOMAP; };

protected:

	/** mds embedding */
	virtual SGMatrix<float64_t> mds_embed(CDistance* distance)
	{
		CClassicMDS* classic_mds = new CClassicMDS();
		classic_mds->set_target_dim(m_target_dim);
		SGMatrix<float64_t> embedding = classic_mds->embed_by_distance(distance);
		delete classic_mds;
		return embedding;
	};
};
}
#endif /* HAVE_LAPACK */
#endif /* CLASSICISOMAP_H_ */
