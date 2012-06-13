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
#include <shogun/so/RiskFunction.h>

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
		CDualLibQPBMSOSVM(CStructuredModel* model, CLossFunction* loss, CStructuredLabels* labs, CFeatures* features, float64_t lambda);

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

	protected:
		/** train dual SO-SVM
		 *
		 */
		bool train_machine(CFeatures* data=NULL);

	private:
		/** register class parameters */
		void register_parameters();

	private:
		/** weight vector */
		SGVector< float64_t > m_w;

		/** lambda */
		float64_t m_lambda;

		/** TolRel */
		float64_t m_TolRel;

		/** TolAbs */
		float64_t m_TolAbs;

		/** BufSize */
		uint32_t m_BufSize;

		/** Risk function */
		CRiskFunction* m_risk_function;

}; /* class CDualLibQPBMSOSVM */

} /* namespace shogun */

#endif /* _DUALLIBQPBMSOSVM__H__ */
