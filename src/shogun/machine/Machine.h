/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _MACHINE_H__
#define _MACHINE_H__

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/Labels.h>
#include <shogun/features/Features.h>

namespace shogun
{

class CFeatures;
class CLabels;
class CMath;

/** classifier type */
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
	CT_KERNELRIDGEREGRESSION = 150,
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
	CT_CONJUGATEINDEX = 400,
	CT_LINEARRIDGEREGRESSION = 410,
	CT_LEASTSQUARESREGRESSION = 420,
	CT_NEWTONSVM = 30
};

/** solver type */
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
 * TODO say something about locking
 *
 */
class CMachine : public CSGObject
{
	public:
		/** constructor */
		CMachine();

		/** destructor */
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
		virtual bool train(CFeatures* data=NULL);

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
		virtual float64_t apply(int32_t num);

		/** load Machine from file
		 *
		 * abstract base method
		 *
		 * @param srcfile file to load from
		 * @return failure
		 */
		virtual bool load(FILE* srcfile);

		/** save Machine to file
		 *
		 * abstract base method
		 *
		 * @param dstfile file to save to
		 * @return failure
		 */
		virtual bool save(FILE* dstfile); 

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(CLabels* lab);

		/** get labels
		 *
		 * @return labels
		 */
		virtual CLabels* get_labels();

		/** get one specific label
		 *
		 * @param i index of label to get
		 * @return value of label at index i
		 */
		virtual float64_t get_label(int32_t i);

		/** set maximum training time
		 *
		 * @param t maximimum training time
		 */
		void set_max_train_time(float64_t t);

		/** get maximum training time
		 *
		 * @return maximum training time
		 */
		float64_t get_max_train_time();

		/** get classifier type
		 *
		 * @return classifier type NONE
		 */
		virtual EClassifierType get_classifier_type();

		/** set solver type
		 *
		 * @param st solver type
		 */
		void set_solver_type(ESolverType st);

		/** get solver type
		 *
		 * @return solver
		 */
		ESolverType get_solver_type();

		/** Setter for store-model-features-after-training flag
		 *
		 * @param store_model whether model should be stored after
		 * training
		 */
		virtual void set_store_model_features(bool store_model);

		/** TODO */
		virtual bool train_locked(SGVector<index_t> indices)
		{
			SG_ERROR("train_locked(SGVector<index_t>) is not yet implemented "
					"for %s\n", get_name());
			return false;
		}

		/** TODO doc */
		virtual CLabels* apply_locked(SGVector<index_t> indices)
		{
			SG_ERROR("apply_locked(SGVector<index_t>) is not yet implemented "
					"for %s\n", get_name());
			return false;
		}

		/** TODO */
		virtual void data_lock(CLabels* labs, CFeatures* features);

		/** TODO */
		virtual void data_unlock();

		/** TODO */
		virtual bool supports_locking() const { return false; }

		/** TODO */
		bool is_data_locked() const { return m_data_locked; }

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
			SG_ERROR("train_machine is not yet implemented for %s!\n",
					get_name());
			return false;
		}

		/** Stores feature data of underlying model.
		 * After this method has been called, it is possible to change
		 * the machine's feature data and call apply(), which is then performed
		 * on the training feature data that is part of the machine's model.
		 *
		 * Base method, has to be implemented in order to allow cross-validation
		 * and model selection.
		 *
		 * NOT IMPLEMENTED! Has to be done in subclasses
		 */
		virtual void store_model_features()
		{
			SG_ERROR("Model storage and therefore unlocked Cross-Validation and"
					" Model-Selection is not supported for %s. Locked may"
					" work though.\n", get_name());
		}

	protected:
		/** maximum training time */
		float64_t m_max_train_time;

		/** labels */
		CLabels* m_labels;

		/** solver type */
		ESolverType m_solver_type;

		/** whether model features should be stored after training */
		bool m_store_model_features;

		/** TODO */
		bool m_data_locked;
};
}
#endif // _MACHINE_H__
