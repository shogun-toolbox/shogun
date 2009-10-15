/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Widmer
 * Copyright (C) 2007-2009 Max-Planck-Society
 */

#ifndef _DA_SVM_H___
#define _DA_SVM_H___


#include "lib/common.h"
#include "classifier/svm/SVM.h"
#include "classifier/svm/SVM_libsvm.h"
#include "classifier/svm/LibSVM.h"
#include "classifier/svm/SVM_light.h"

#include <stdio.h>

class CDA_SVM : public CSVMLight
{
	private:

/*
#ifdef HAVE_BOOST_SERIALIZATION
		friend class boost::serialization::access;
		// When the class Archive corresponds to an output archive, the
		// & operator is defined similar to <<.  Likewise, when the class Archive
		// is a type of input archive the & operator is defined similar to >>.
		template<class Archive>
        void serialize(Archive & ar, const unsigned int archive_version)
        {
            SG_DEBUG("archiving CDA_SVM\n");

            ar & boost::serialization::base_object<CLibSVM>(*this);

            ar & presvm;

            ar & B;

            ar & trainFactor;
            SG_DEBUG("done archiving CDA_SVM\n");

        }
#endif //HAVE_BOOST_SERIALIZATION
*/
	public:
    CDA_SVM();
	CDA_SVM(float64_t C, CKernel* k, CLabels* lab, CSVM* presvm, float64_t B);
	//CDA_SVM(std::string presvm_fn, float64_t B);

    void init(CSVM* presvm, float64_t B);

	virtual ~CDA_SVM();

	virtual bool train();
    virtual inline EClassifierType get_classifier_type() { return CT_DASVM; }

	virtual CLabels* classify(CFeatures* data);


    virtual CSVM* get_presvm();
    virtual float64_t get_B();


	protected:

    CSVM* presvm;
    float64_t B;
    float64_t trainFactor;

};
#endif

