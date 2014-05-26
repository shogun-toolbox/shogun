/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#ifndef _DUALLIBQPBMSOSVM__H__
#define _DUALLIBQPBMSOSVM__H__

#include <shogun/lib/config.h>

#include <shogun/machine/LinearStructuredOutputMachine.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/structure/BmrmStatistics.h>

namespace shogun
{

/**
 * Enum
 * Training method selection
 */
enum ESolver
{
	BMRM=1,		/**< Standard BMRM algorithm. */
	PPBMRM=2,	/**< Proximal Point BMRM (BMRM with prox-term) */
	P3BMRM=3,	/**< Proximal Point P-BMRM (multiple cutting plane models) */
	NCBM=4
};

/**
 * @brief Class DualLibQPBMSOSVM that uses Bundle Methods for Regularized Risk
 * Minimization algorithms for structured output (SO) problems [1] presented
 * in [2].
 *
 * [1] Tsochantaridis, I., Hofmann, T., Joachims, T., Altun, Y.
 *	   Support Vector Machine Learning for Interdependent and Structured Ouput
 *	   Spaces.
 *	   http://www.cs.cornell.edu/People/tj/publications/tsochantaridis_etal_04a.pdf
 *
 * [2] Teo, C.H., Vishwanathan, S.V.N, Smola, A. and Quoc, V.Le.
 *     Bundle Methods for Regularized Risk Minimization
 *     http://users.cecs.anu.edu.au/~chteo/pub/TeoVisSmoLe10.pdf
 */
class CDualLibQPBMSOSVM : public CLinearStructuredOutputMachine
{
	public:
		/** default constructor */
		CDualLibQPBMSOSVM();

		/** constructor
		 *
		 * @param model		Structured Model
		 * @param labs			Structured labels
		 * @param _lambda		Regularization constant
		 * @param W				initial solution of weight vector
		 */
		CDualLibQPBMSOSVM(
				CStructuredModel*		model,
				CStructuredLabels*		labs,
				float64_t				_lambda,
				SGVector< float64_t >	W=0);

		/** destructor */
		virtual ~CDualLibQPBMSOSVM();

		/** @return name of SGSerializable */
		virtual const char* get_name() const { return "DualLibQPBMSOSVM"; }

		/** set lambda
		 *
		 * @param _lambda	Regularization constant
		 */
		inline void set_lambda(float64_t _lambda) { m_lambda=_lambda; }

		/** get lambda
		 *
		 * @return Regularization constant
		 */
		inline float64_t get_lambda() { return m_lambda; }

		/** set relative tolerance
		 *
		 * @param TolRel	Relative tolerance
		 */
		inline void set_TolRel(float64_t TolRel) { m_TolRel=TolRel; }

		/** get relative tolerance
		 *
		 * @return Relative tolerance
		 */
		inline float64_t get_TolRel() { return m_TolRel; }

		/** set absolute tolerance
		 *
		 * @param TolAbs	Absolute tolerance
		 */
		inline void set_TolAbs(float64_t TolAbs) { m_TolAbs=TolAbs; }

		/** get absolute tolerance
		 *
		 * @return Absolute tolerance
		 */
		inline float64_t get_TolAbs() { return m_TolAbs; }

		/** set size of cutting plane buffer
		 *
		 * @param BufSize	Size of the cutting plane buffer (i.e. maximal number of
		 *					iterations)
		 */
		inline void set_BufSize(uint32_t BufSize) { m_BufSize=BufSize; }

		/** get size of cutting plane buffer
		 *
		 * @return Size of the cutting plane buffer
		 */
		inline uint32_t get_BufSize() { return m_BufSize; }

		/** set ICP removal flag
		 *
		 * @param cleanICP	Flag that enables/disables inactive cutting plane removal
		 *					feature
		 */
		inline void set_cleanICP(bool cleanICP) { m_cleanICP=cleanICP; }

		/** get ICP removal flag
		 *
		 * @return Status of inactive cutting plane removal feature (enabled/disabled)
		 */
		inline bool get_cleanICP() { return m_cleanICP; }

		/** set number of iterations for cleaning ICP
		 *
		 * @param cleanAfter	Specifies number of iterations that inactive cutting
		 *						planes has to be inactive for to be removed
		 */
		inline void set_cleanAfter(uint32_t cleanAfter) { m_cleanAfter=cleanAfter; }

		/** get number of iterations for cleaning ICP
		 *
		 * @return Number of iterations that inactive cutting planes has to be
		 *			inactive for to be removed
		 */
		inline uint32_t get_cleanAfter() { return m_cleanAfter; }

		/** set K
		 *
		 * @param K	Parameter K
		 */
		inline void set_K(float64_t K) { m_K=K; }

		/** get K
		 *
		 * @return K
		 */
		inline float64_t get_K() { return m_K; }

		/** set Tmax
		 *
		 * @param Tmax Parameter Tmax
		 */
		inline void set_Tmax(uint32_t Tmax) { m_Tmax=Tmax; }

		/** get Tmax
		 *
		 * @return Tmax
		 */
		inline uint32_t get_Tmax() { return m_Tmax; }

		/** set number of cutting plane models
		 *
		 * @param cp_models	Number of cutting plane models
		 */
		inline void set_cp_models(uint32_t cp_models) { m_cp_models=cp_models; }

		/** get number of cutting plane models
		 *
		 * @return Number of cutting plane models
		 */
		inline uint32_t get_cp_models() { return m_cp_models; }

		/** get bmrm result
		 *
		 * @return Result returned from Bundle Method algorithm
		 */
		inline BmrmStatistics get_result() { return m_result; }

		/** get training algorithm
		 *
		 * @return Type of Bundle Method solver used for training
		 */
		inline ESolver get_solver() { return m_solver; }

		/** set training algorithm
		 *
		 * @param solver	Type of Bundle Method solver used for training
		 */
		inline void set_solver(ESolver solver) { m_solver=solver; }

		/** set initial value of weight vector w
		 *
		 * @param W     initial weight vector
		 */
		inline void set_w(SGVector< float64_t > W)
		{
			REQUIRE(W.vlen == m_model->get_dim(), "Dimension of the initial "
					"solution must match the model's dimension!\n");
			m_w=W;
		}
		
		/** set enableing/disabling storing training information
		 *
		 * @param store_train_info		Flag enabling/disabling storing training information,
		 * 								Storing training information requires extra computational costs.
		 */
		inline void set_store_train_info(bool store_train_info)
		{
			m_store_train_info=store_train_info;
		}
		
		/** get classifier type
		 *
		 * @return classifier type CT_LIBQPSOSVM
		 */
		virtual EMachineType get_classifier_type();

	protected:
		/** train dual SO-SVM
		 *
		 */
		bool train_machine(CFeatures* data=NULL);

	private:
		/** init parameters
		 *
		 */
		void init();

	private:

		/** lambda */
		float64_t m_lambda;

		/** TolRel */
		float64_t m_TolRel;

		/** TolAbs */
		float64_t m_TolAbs;

		/** BufSize */
		uint32_t m_BufSize;

		/** Clean ICP */
		bool m_cleanICP;

		/** Clean ICP after n-th iteration */
		uint32_t m_cleanAfter;

		/** K */
		float64_t m_K;

		/** Tmax */
		uint32_t m_Tmax;

		/** number of cutting plane models */
		uint32_t m_cp_models;

		/** BMRM result */
		BmrmStatistics m_result;

		/** training algorithm */
		ESolver m_solver;

		/** store training information*/
		bool m_store_train_info;

}; /* class CDualLibQPBMSOSVM */

} /* namespace shogun */

#endif /* _DUALLIBQPBMSOSVM__H__ */
