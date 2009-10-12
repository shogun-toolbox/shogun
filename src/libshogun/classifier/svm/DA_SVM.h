/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Widmer
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
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
	//virtual CLabels* classify(CLabels* labels=NULL);
	//virtual float64_t classify_example(int32_t num);

    virtual CSVM* get_presvm();
    virtual float64_t get_B();

    /*
    virtual void toFile(std::string filename) const
    {

      //std::ofstream os(filename.c_str(), std::ios::binary);
      //boost::archive::binary_oarchive oa(os);
      std::ofstream os(filename.c_str());
      boost::archive::text_oarchive oa(os);

      oa << *this;

    }

    virtual void fromFile(std::string filename)
    {

      //std::ifstream is(filename.c_str(), std::ios::binary);
      //boost::archive::binary_iarchive ia(is);

      std::ifstream is(filename.c_str());
      boost::archive::text_iarchive ia(is);

      ia >> *this;

    }

    virtual std::string toString() const
    {
      std::ostringstream s;

      boost::archive::text_oarchive oa(s);

      oa << *this;

      return s.str();
    }


    virtual void fromString(std::string str)
    {

      std::istringstream is(str);

      boost::archive::text_iarchive ia(is);

      ia >> *this;

    }

    */

    float64_t get_trainFactor()
    {
      return trainFactor;
    }

    void set_trainFactor(float64_t factor)
    {
      this->trainFactor = factor;

    }

    int32_t get_num_support_vectors()
    {
        int32_t current_num = CSVM::get_num_support_vectors();

        int32_t old_num = presvm->get_num_support_vectors();

        return current_num + old_num;
    }

    int32_t get_alpha(int32_t idx)
    {
        int32_t current_num = CSVM::get_num_support_vectors();

        float64_t alpha = 0.0;

        if (idx < current_num)
        {
            alpha = CSVM::get_alpha(idx);
        } else {
            alpha = presvm->get_alpha(idx);
        }

        return alpha;
        
    }

    int32_t get_support_vector(int32_t idx)
    {
        int32_t current_num = CSVM::get_num_support_vectors();

        int32_t old_num = presvm->get_num_support_vectors();

        float64_t alpha = 0.0;

        if (idx < current_num)
        {
            alpha = CSVM::get_support_vector(idx);
        } else {
            alpha = B * presvm->get_support_vector(idx);
        }

        return alpha;
    }

	protected:

    CSVM* presvm;
    float64_t B;

    float64_t trainFactor;

};
#endif

