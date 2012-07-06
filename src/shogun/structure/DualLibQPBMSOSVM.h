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

#include <shogun/machine/LinearStructuredOutputMachine.h>
#include <shogun/structure/RiskFunction.h>
#include <shogun/structure/libbmrm.h>

namespace shogun
{

class CDualLibQPBMSOSVM : public CLinearStructuredOutputMachine
{
	public:
		/** default constructor */
		CDualLibQPBMSOSVM();

		/** standard constructor
		 *
		 */
		CDualLibQPBMSOSVM(CStructuredModel* model, CLossFunction* loss, CStructuredLabels* labs, CDotFeatures* features, float64_t lambda, CRiskFunction* risk_function);

		/** destructor */
		~CDualLibQPBMSOSVM();

		/** set lambda */
		inline void set_lambda(float64_t lambda) { m_lambda=lambda; }

		/** get lambda */
		inline float64_t get_lambda() { return m_lambda; }

		/** set relative tolerance */
		inline void set_TolRel(float64_t TolRel) { m_TolRel=TolRel; }

		/** get relative tolerance */
		inline float64_t get_TolRel() { return m_TolRel; }

		/** set absolute tolerance */
		inline void set_TolAbs(float64_t TolAbs) { m_TolAbs=TolAbs; }

		/** get absolute tolerance */
		inline float64_t get_TolAbs() { return m_TolAbs; }

		/** set size of cutting plane buffer */
		inline void set_BufSize(uint32_t BufSize) { m_BufSize=BufSize; }

		/** get size of cutting plane buffer */
		inline uint32_t get_BufSize() { return m_BufSize; }

		/** set ICP removal flag */
		inline void set_cleanICP(bool cleanICP) { m_cleanICP=cleanICP; }

		/** get ICP removal flag */
		inline bool get_cleanICP() { return m_cleanICP; }

		/** set number of iterations for cleaning ICP */
		inline void set_cleanAfter(uint32_t cleanAfter) { m_cleanAfter=cleanAfter; }

		/** get number of iterations for cleaninng ICP */
		inline uint32_t get_cleanAfter() { return m_cleanAfter; }

		/** set K */
		inline void set_K(float64_t K) { m_K=K; }

		/** get K */
		inline float64_t get_K() { return m_K; }

		/** set Tmax */
		inline void set_Tmax(uint32_t Tmax) { m_Tmax=Tmax; }

		/** get Tmax */
		inline uint32_t get_Tmax() { return m_Tmax; }

		/** get bmrm result */
		inline bmrm_return_value_T get_bmrm_result() { return m_bmrm_result; }

	protected:
		/** train dual SO-SVM
		 *
		 */
		bool train_machine(CFeatures* data=NULL);

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

		/** Risk function */
		CRiskFunction* m_risk_function;

		/** BMRM result */
		bmrm_return_value_T m_bmrm_result;

}; /* class CDualLibQPBMSOSVM */

} /* namespace shogun */

#endif /* _DUALLIBQPBMSOSVM__H__ */
