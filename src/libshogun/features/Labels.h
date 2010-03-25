/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LABELS__H__
#define _LABELS__H__

#include "lib/common.h"
#include "lib/io.h"
#include "lib/File.h"
#include "base/SGObject.h"

namespace shogun
{

	class CFile;

/** @brief The class Labels models labels, i.e. class assignments of objects.
 *
 * Labels here are always real-valued and thus applicable to classification
 * (cf.  CClassifier) and regression (cf. CRegression) problems.
 */
class CLabels : public CSGObject
{
	public:
		/** default constructor */
		CLabels();

		/** constructor
		 *
		 * @param num_labels number of labels
		 */
		CLabels(int32_t num_labels);

		/** constructor
		 *
		 * @param src labels to set
		 * @param len number of labels
		 */
		CLabels(float64_t* src, int32_t len);

		/** constructor
		 *
		 * @param loader File object via which to load data
		 */
		CLabels(CFile* loader);
		virtual ~CLabels();

		/** load labels from file
		 *
		 * @param loader File object via which to load data
		 */
		virtual void load(CFile* loader);

		/** save labels to file
		 *
		 * @param saver File object via which to save data
		 */
		virtual void save(CFile* saver);

		/** set label
		 *
		 * @param idx index of label to set
		 * @param label value of label
		 * @return if setting was successful
		 */
		inline bool set_label(int32_t idx, float64_t label)
		{ 
			if (labels && idx<num_labels)
			{
				labels[idx]=label;
				return true;
			}
			else 
				return false;
		}

		/** set INT label
		 *
		 * @param idx index of label to set
		 * @param label INT value of label
		 * @return if setting was successful
		 */
		inline bool set_int_label(int32_t idx, int32_t label)
		{ 
			if (labels && idx<num_labels)
			{
				labels[idx]= (float64_t) label;
				return true;
			}
			else 
				return false;
		}

		/** get label
		 *
		 * @param idx index of label to get
		 * @return value of label
		 */
		inline float64_t get_label(int32_t idx)
		{
			if (labels && idx<num_labels)
				return labels[idx];
			else
				return -1;
		}

		/** get INT label
		 *
		 * @param idx index of label to get
		 * @return INT value of label
		 */
		inline int32_t get_int_label(int32_t idx)
		{
			if (labels && idx<num_labels)
			{
				ASSERT(labels[idx]== ((float64_t) ((int32_t) labels[idx])));
				return ((int32_t) labels[idx]);
			}
			else
				return -1;
		}

		/** is two-class labeling
		 *
		 * @return if this is two-class labeling
		 */
		bool is_two_class_labeling();

		/** return number of classes (for multiclass)
		 * labels have to be zero based 0,1,...C missing
		 * labels are illegal
		 *
		 * @return number of classes
		 */
		int32_t get_num_classes();

		/** get labels
		 * caller has to clean up
		 *
		 * @param len number of labels
		 * @return the labels
		 */
		float64_t* get_labels(int32_t &len);
		
		/** get labels (swig compatible)
		 *
		 * @param dst where labels will be stored in
		 * @param len where number of labels will be stored in
		 */
		void get_labels(float64_t** dst, int32_t* len);

		/** set labels
		 *
		 * @param src labels to set
		 * @param len number of labels
		 */
		void set_labels(float64_t* src, int32_t len);

		/** get INT label vector
		 * caller has to clean up
		 *
		 * @param len number of labels to get
		 * @return INT labels
		 */
		int32_t* get_int_labels(int32_t &len);

		/** set INT labels
		 * caller has to clean up
		 *
		 * @param labels INT labels
		 * @param len number of INT labels
		 */
		void set_int_labels(int32_t *labels, int32_t len) ;

		/** get number of labels
		 *
		 * @return number of labels
		 */
		inline int32_t get_num_labels() { return num_labels; }

		/** @return object name */
		inline virtual const char* get_name() const { return "Labels"; }

#ifdef HAVE_BOOST_SERIALIZATION
    private:

        // serialization needs to split up in save/load because 
        // the serialization of pointers to natives (int* & friends) 
        // requires a workaround 
        friend class ::boost::serialization::access;
        template<class Archive>
            void save(Archive & ar, const unsigned int archive_version) const
            {

                SG_DEBUG("archiving Labels\n");

                ar & ::boost::serialization::base_object<CSGObject>(*this);

                ar & num_labels;
                for (int32_t i=0; i < num_labels; ++i) 
                {
                    ar & labels[i];
                }

                SG_DEBUG("done with Labels\n");

            }

        template<class Archive>
            void load(Archive & ar, const unsigned int archive_version)
            {

                SG_DEBUG("archiving Labels\n");

                ar & ::boost::serialization::base_object<CSGObject>(*this);

                ar & num_labels;

                SG_DEBUG("num_labels: %i\n", num_labels);

                if (num_labels > 0)
                {

                    labels = new float64_t[num_labels];
                    for (int32_t i=0; i< num_labels; ++i) 
                    {
                        ar & labels[i];
                    }

                }

                SG_DEBUG("done with Labels\n");

            }

        GLOBAL_BOOST_SERIALIZATION_SPLIT_MEMBER();


    public:

        virtual std::string toString() const
        {
            std::ostringstream s;

            ::boost::archive::text_oarchive oa(s);

            oa << *this;

            return s.str();
        }


        virtual void fromString(std::string str)
        {

            std::istringstream is(str);

            ::boost::archive::text_iarchive ia(is);

            ia >> *this;

        }
#endif //HAVE_BOOST_SERIALIZATION

	protected:
		/** number of labels */
		int32_t num_labels;
		/** the labels */
		float64_t* labels;
};
}
#endif
