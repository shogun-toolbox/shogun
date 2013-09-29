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
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/labels/LatentLabels.h>
#include <shogun/features/Features.h>

namespace shogun
{

class CFeatures;
class CLabels;
class CMath;

/** classifier type */
enum EMachineType
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
	CT_QDA = 430,
	CT_NEWTONSVM = 440,
	CT_GAUSSIANPROCESSREGRESSION = 450,
	CT_LARS = 460,
	CT_MULTICLASS = 470,
	CT_DIRECTORLINEAR = 480,
	CT_DIRECTORKERNEL = 490,
	CT_LIBQPSOSVM = 500,
	CT_PRIMALMOSEKSOSVM = 510,
	CT_CCSOSVM = 520,
	CT_GAUSSIANPROCESSBINARY = 530,
	CT_GAUSSIANPROCESSMULTICLASS = 540,
	CT_STOCHASTICSOSVM = 550,
	CT_BAGGING
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

/** problem type */
enum EProblemType
{
	PT_BINARY = 0,
	PT_REGRESSION = 1,
	PT_MULTICLASS = 2,
	PT_STRUCTURED = 3,
	PT_LATENT = 4
};

#define MACHINE_PROBLEM_TYPE(PT) \
	/** returns default problem type machine solves \
	 * @return problem type\
	 */ \
	virtual EProblemType get_machine_problem_type() const { return PT; }

/** @brief A generic learning machine interface.
 *
 * A machine takes as input CFeatures and CLabels (by default).
 * Later subclasses may specialize the machine to e.g. require labels
 * and a kernel or labels and (real-valued) features.
 *
 * A machine needs to override the train() function for training,
 * the functions apply(idx) (optionally apply() to predict on the
 * whole set of examples) and the load and save routines.
 *
 * Machines may support locking. This means that given some data, the machine
 * can be locked on this data to speed up computations. E.g. a kernel machine
 * may precompute its kernel. Only train_locked and apply_locked are available
 * when locked. There are methods for checking whether a machine supports
 * locking.
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

		/** apply machine to data
		 * if data is not specified apply to the current features
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data=NULL);

		/** apply machine to data in means of binary classification problem */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);
		/** apply machine to data in means of regression problem */
		virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);
		/** apply machine to data in means of multiclass classification problem */
		virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);
		/** apply machine to data in means of SO classification problem */
		virtual CStructuredLabels* apply_structured(CFeatures* data=NULL);
		/** apply machine to data in means of latent problem */
		virtual CLatentLabels* apply_latent(CFeatures* data=NULL);

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
		virtual EMachineType get_classifier_type();

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

		/** Trains a locked machine on a set of indices. Error if machine is
		 * not locked
		 *
		 * NOT IMPLEMENTED
		 *
		 * @param indices index vector (of locked features) that is used for training
		 * @return whether training was successful
		 */
		virtual bool train_locked(SGVector<index_t> indices)
		{
			SG_ERROR("train_locked(SGVector<index_t>) is not yet implemented "
					"for %s\n", get_name());
			return false;
		}

		/** applies to one vector */
		virtual float64_t apply_one(int32_t i)
		{
			SG_NOTIMPLEMENTED
			return 0.0;
		}

		/** Applies a locked machine on a set of indices. Error if machine is
		 * not locked
		 *
		 * @param indices index vector (of locked features) that is predicted
		 */
		virtual CLabels* apply_locked(SGVector<index_t> indices);

		/** applies a locked machine on a set of indices for binary problems */
		virtual CBinaryLabels* apply_locked_binary(
				SGVector<index_t> indices);
		/** applies a locked machine on a set of indices for regression problems */
		virtual CRegressionLabels* apply_locked_regression(
				SGVector<index_t> indices);
		/** applies a locked machine on a set of indices for multiclass problems */
		virtual CMulticlassLabels* apply_locked_multiclass(
				SGVector<index_t> indices);
		/** applies a locked machine on a set of indices for structured problems */
		virtual CStructuredLabels* apply_locked_structured(
				SGVector<index_t> indices);
		/** applies a locked machine on a set of indices for latent problems */
		virtual CLatentLabels* apply_locked_latent(
				SGVector<index_t> indices);

		/** Locks the machine on given labels and data. After this call, only
		 * train_locked and apply_locked may be called
		 *
		 * Only possible if supports_locking() returns true
		 *
		 * @param labs labels used for locking
		 * @param features features used for locking
		 */
		virtual void data_lock(CLabels* labs, CFeatures* features);

		/** post lock */
		virtual void post_lock(CLabels* labs, CFeatures* features) { };

		/** Unlocks a locked machine and restores previous state */
		virtual void data_unlock();

		/** @return whether this machine supports locking */
		virtual bool supports_locking() const { return false; }

		/** @return whether this machine is locked */
		bool is_data_locked() const { return m_data_locked; }

		/** returns type of problem machine solves */
		virtual EProblemType get_machine_problem_type() const
		{
			SG_NOTIMPLEMENTED
			return PT_BINARY;
		}

		virtual const char* get_name() const { return "Machine"; }

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

		/** check whether the labels is valid.
		 *
		 * Subclasses can override this to implement their check of label types.
		 *
		 * @param lab the labels being checked, guaranteed to be non-NULL
		 */
		virtual bool is_label_valid(CLabels *lab) const
		{
			return true;
		}

		/** returns whether machine require labels for training */
		virtual bool train_require_labels() const { return true; }

	protected:
		/** maximum training time */
		float64_t m_max_train_time;

		/** labels */
		CLabels* m_labels;

		/** solver type */
		ESolverType m_solver_type;

		/** whether model features should be stored after training */
		bool m_store_model_features;

		/** whether data is locked */
		bool m_data_locked;
};
}
#endif // _MACHINE_H__
