/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2013 Roman Votyakov
 */

#ifndef _GAUSSIANPROCESSBINARYCLASSIFICATION_H_
#define _GAUSSIANPROCESSBINARYCLASSIFICATION_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/GaussianProcessMachine.h>

namespace shogun
{

/** TODO: insert documentation
 *
 *
 */
class CGaussianProcessBinaryClassification : public CGaussianProcessMachine
{
public:
	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_BINARY);

	/** default constructor */
	CGaussianProcessBinaryClassification();

	/** constructor
	 *
	 * @param method inference method
	 */
	CGaussianProcessBinaryClassification(CInferenceMethod* method);

	virtual ~CGaussianProcessBinaryClassification();

	/** apply machine to data in means of binary classification problem
	 *
	 * @param data (test) data to be classified
	 *
	 * @return classified labels
	 */
	virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);

	/** get classifier type
	 *
	 * @return classifier type GAUSSIANPROCESSBINARY
	 */
	virtual EMachineType get_classifier_type()
	{
		return CT_GAUSSIANPROCESSBINARY;
	}

	/** return name of the classifier
	 *
	 * @return name GaussianProcessBinaryClassification
	 */
	virtual const char* get_name() const
	{
		return "GaussianProcessBinaryClassification";
	}

	/** load from file
	 *
	 * @param srcfile file to load from
	 * @return if loading was successful
	 */
	virtual bool load(FILE* srcfile);

	/** save to file
	 *
	 * @param dstfile file to save to
	 * @return if saving was successful
	 */
	virtual bool save(FILE* dstfile);

protected:
	/** train classifier
	 *
	 * @param data training data
	 *
	 * @return whether training was successful
	 */
	virtual bool train_machine(CFeatures* data=NULL);

	/** check wether training labels are valid for classification
	 *
	 * @param lab training labels
	 *
	 * @return wether training labels are valid for classification
	 */
	virtual bool is_label_valid(CLabels *lab) const
	{
		return (lab->get_label_type()==LT_BINARY);
	}
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _GAUSSIANPROCESSCLASSIFICATION_H_ */
