/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CLASSIFIER_H__
#define _CLASSIFIER_H__

#include "lib/common.h"
#include "base/SGObject.h"
#include "lib/Mathematics.h"
#include "features/Labels.h"
#include "features/Features.h"

namespace shogun
{

class CFeatures;
class CLabels;
class CMath;

enum EClassifierType
{
	CT_NONE = 0,
	CT_LIGHT = 10,
	CT_LIGHTONECLASS = 11,
	CT_LIBSVM = 20,
	CT_LIBSVMONECLASS=30,
	CT_LIBSVMMULTICLASS=40,
	CT_MPD = 50,
	CT_GPBT = 60,
	CT_CPLEXSVM = 70,
	CT_PERCEPTRON = 80,
	CT_KERNELPERCEPTRON = 90,
	CT_LDA = 100,
	CT_LPM = 110,
	CT_LPBOOST = 120,
	CT_KNN = 130,
	CT_SVMLIN=140,
	CT_KRR = 150,
	CT_GNPPSVM = 160,
	CT_GMNPSVM = 170,
	CT_SUBGRADIENTSVM = 180,
	CT_SUBGRADIENTLPM = 190,
	CT_SVMPERF = 200,
	CT_LIBSVR = 210,
	CT_SVRLIGHT = 220,
	CT_LIBLINEAR = 230,
	CT_KMEANS = 240,
	CT_HIERARCHICAL = 250,
	CT_SVMOCAS = 260,
	CT_WDSVMOCAS = 270,
	CT_SVMSGD = 280,
	CT_MKLMULTICLASS = 290,
	CT_MKLCLASSIFICATION = 300,
	CT_MKLONECLASS = 310,
	CT_MKLREGRESSION = 320,
	CT_SCATTERSVM = 330,
	CT_DASVM = 340,
	CT_LARANK = 350,
	CT_DASVMLINEAR = 360,
	CT_GAUSSIANNAIVEBAYES = 370,
	CT_AVERAGEDPERCEPTRON = 380,
	CT_SGDQN = 390,

};

enum ESolverType
{
	ST_AUTO=0,
	ST_CPLEX=1,
	ST_GLPK=2,
	ST_NEWTON=3,
	ST_DIRECT=4,
	ST_ELASTICNET=5
};

/** @brief A generic classifier interface.
 *
 * A classifier takes as input CLabels. Later subclasses may specialize the
 * classifier to require labels and a kernel or labels and (real-valued)
 * features.
 *
 * A classifier needs to override the train() function for training,
 * the function classify_example() (optionally classify() to predict on the
 * whole set of examples) and the load and save routines.
 *
 */
class CClassifier : public CSGObject
{
	public:
		/** constructor */
		CClassifier();
		virtual ~CClassifier();

		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL)
		{
			SG_NOTIMPLEMENTED;
			return false;
		}

		/** classify objects using the currently set features
		 *
		 * @return classified labels
		 */
		virtual CLabels* classify()=0;

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* classify(CFeatures* data)=0;

		/** classify one example
		 *
		 * abstract base method
		 *
		 * @param num which example to classify
		 * @return infinite float value
		 */
		virtual float64_t classify_example(int32_t num)
		{
			SG_NOTIMPLEMENTED;
			return CMath::INFTY;
		}

		/** load Classifier from file
		 *
		 * abstract base method
		 *
		 * @param srcfile file to load from
		 * @return failure
		 */
		virtual bool load(FILE* srcfile) { ASSERT(srcfile); return false; }

		/** save Classifier to file
		 *
		 * abstract base method
		 *
		 * @param dstfile file to save to
		 * @return failure
		 */
		virtual bool save(FILE* dstfile) { ASSERT(dstfile); return false; }

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual inline void set_labels(CLabels* lab)
		{
			SG_UNREF(labels);
			SG_REF(lab);
			labels=lab;
		}

		/** get labels
		 *
		 * @return labels
		 */
		virtual inline CLabels* get_labels() { SG_REF(labels); return labels; }

		/** get one specific label
		 *
		 * @param i index of label to get
		 * @return value of label at index i
		 */
		virtual inline float64_t get_label(int32_t i)
		{
			if (!labels)
				SG_ERROR("No Labels assigned\n");

			return labels->get_label(i);
		}

		/** set maximum training time
		 *
		 * @param t maximimum training time
		 */
		inline void set_max_train_time(float64_t t) { max_train_time=t; }

		/** get maximum training time
		 *
		 * @return maximum training time
		 */
		inline float64_t get_max_train_time() { return max_train_time; }

		/** get classifier type
		 *
		 * @return classifier type NONE
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_NONE; }

		/** set solver type
		 *
		 * @param st solver type
		 */
		inline void set_solver_type(ESolverType st) { solver_type=st; }

		/** get solver type
		 *
		 * @return solver
		 */
		inline ESolverType get_solver_type() { return solver_type; }

	protected:
		/** maximum training time */
		float64_t max_train_time;

		/** labels */
		CLabels* labels;

		/** solver type */
		ESolverType solver_type;
};
}
#endif // _CLASSIFIER_H__
