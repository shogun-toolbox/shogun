/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MACHINE_H__
#define _MACHINE_H__

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/Labels.h>
#include <shogun/features/Features.h>

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
	ST_ELASTICNET=5,
	ST_BLOCK_NORM=6
};

/** @brief A generic learning machine interface.
 *
 * A machine takes as input CFeatures and (optionally) CLabels.
 * Later subclasses may specialize the machine to e.g. require labels
 * and a kernel or labels and (real-valued) features.
 *
 * A machine needs to override the train() function for training,
 * the functions apply(idx) (optionally apply() to predict on the
 * whole set of examples) and the load and save routines.
 *
 */
class CMachine : public CSGObject
{
	public:
		/** constructor */
		CMachine();
		virtual ~CMachine();

		/** train machine
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data).
		 * If flag is set, model features will be stored after training.
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL)
		{
			bool result=train_machine(data);

			if (m_store_model_features)
				store_model_features();

			return false;
		}

		/** apply machine to the currently set features
		 *
		 * @return output 'labels'
		 */
		virtual CLabels* apply()=0;

		/** apply machine to data
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data)=0;

		/** apply machine to one example
		 *
		 * abstract base method
		 *
		 * @param num which example to apply machine to
		 * @return infinite float value
		 */
		virtual float64_t apply(int32_t num)
		{
			SG_NOTIMPLEMENTED;
			return CMath::INFTY;
		}

		/** load Machine from file
		 *
		 * abstract base method
		 *
		 * @param srcfile file to load from
		 * @return failure
		 */
		virtual bool load(FILE* srcfile) { ASSERT(srcfile); return false; }

		/** save Machine to file
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

		/** Setter for store-model-features-after-training flag
		 *
		 * @param store_model_features whether model should be stored after
		 * training
		 */
		virtual void set_store_model_features(bool store_model_features)
		{
			m_store_model_features=store_model_features;
		}

		/** Stores feature data of underlying model.
		 *
		 * NOT IMPLEMENTED!
		 */
		virtual void store_model_features()
		{
			SG_NOTIMPLEMENTED;
		}

	protected:
		/** train machine
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL)
		{
			SG_NOTIMPLEMENTED;
			return false;
		}

	protected:
		/** maximum training time */
		float64_t max_train_time;

		/** labels */
		CLabels* labels;

		/** solver type */
		ESolverType solver_type;

		/** whether model features should be stored after training */
		bool m_store_model_features;
};
}
#endif // _MACHINE_H__
