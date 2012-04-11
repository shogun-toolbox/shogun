/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef LARS_H_WSBLKNVU
#define LARS_H_WSBLKNVU

#include <shogun/machine/LinearMachine.h>

class CFeatures;

namespace shogun 
{

// TODO: rename LARS to CLARS
class CLARS: public CLinearMachine
{
public:
	CLARS():m_lasso(true)
	{
	}
	virtual ~CLARS()
	{
	}

	/** load regression from file
	 *
	 * @param srcfile file to load from
	 * @return if loading was successful
	 */
	virtual bool load(FILE* srcfile);

	/** save regression to file
	 *
	 * @param dstfile file to save to
	 * @return if saving was successful
	 */
	virtual bool save(FILE* dstfile);

	/** get classifier type
	 *
	 * @return classifier type LinearRidgeRegression
	 */
	inline virtual EClassifierType get_classifier_type()
	{
		return CT_LARS;
	}

	/** @return object name */
	inline virtual const char* get_name() const { return "LARS"; }

protected:
	virtual bool train_machine(CFeatures* data=NULL);

private:
	bool m_lasso;
}; // class LARS

} // namespace shogun

#endif /* end of include guard: LARS_H_WSBLKNVU */

