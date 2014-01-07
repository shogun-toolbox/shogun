/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Michal Uricar, uricamic@cmp.felk.cvut.cz
 *
 */

#ifndef BMRM_RETURN_VALUE_H_
#define BMRM_RETURN_VALUE_H_

#include <lib/common.h>
#include <io/SerializableFile.h>
#include <lib/SGVector.h>

namespace shogun
{

/** BMRM statistics like number of iterations etc */
struct BmrmStatistics
{
	/** constructor */
	BmrmStatistics()
	{
		nIter = 0;
		nCP = 0;
		nzA = 0;
		Fp = 0;
		Fd = 0;
		qp_exitflag = 0;
		exitflag = 0;
	};

	/** destructor */
	virtual ~BmrmStatistics() { };

	/** dummy load serializable */
	bool load_serializable(CSerializableFile* file, const char* prefix="") { return false; }

	/** dummy save serializable */
	bool save_serializable(CSerializableFile* file, const char* prefix="") { return false; }

	/** number of iterations  */
	uint32_t nIter;

	/** getter for nIter */
	uint32_t get_n_iters() const { return nIter; }

	/** number of cutting planes */
	uint32_t nCP;

	/** number of active cutting planes */
	uint32_t nzA;

	/** primal objective value  */
	float64_t Fp;

	/** reduced (dual) objective value */
	float64_t Fd;

	/** exitflag from the last call of the inner QP solver  */
	int8_t qp_exitflag;

	/** 1 .. bmrm.Q_P - bmrm.Q_D <= TolRel*ABS(bmrm.Q_P)
	 *  2 .. bmrm.Q_P - bmrm.Q_D <= TolAbs
	 * -1 .. bmrm.nCutPlanes >= BufSize
	 * -2 .. not enough memory for the solver
	 */
	int8_t exitflag;

	/** Track of Fp values in individual iterations */
	SGVector< float64_t > hist_Fp;

	/** getter for hist_Fp */
	SGVector< float64_t > get_hist_Fp_vector() const { return hist_Fp; };

	/** Track of Fd values in individual iterations */
	SGVector< float64_t > hist_Fd;

	/** getter for hist_Fd */
	SGVector< float64_t > get_hist_Fd_vector() const { return hist_Fd; };

	/** Track of w_dist values in individual iterations */
	SGVector< float64_t > hist_wdist;

	/** getter for hist_wdist */
	SGVector< float64_t > get_hist_wdist_vector() const { return hist_wdist; };
};

}
#endif
