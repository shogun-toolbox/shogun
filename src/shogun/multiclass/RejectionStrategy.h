/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef _REJECTIONSTRATEGY_H___
#define _REJECTIONSTRATEGY_H___

namespace shogun
{

/** @brief base rejection strategy class */
class CRejectionStrategy : public CSGObject
{
	public:
		/** default constructor  */
		CRejectionStrategy() { };

		/** destructor */
		virtual ~CRejectionStrategy() { };

		/** get name */
		virtual const char* get_name() const
		{
				return "RejectionStrategy";
		};

		/** returns true if given output set leads to rejection */
		virtual bool reject(SGVector<float64_t> outputs) const = 0;

};

/** @brief threshold based rejection strategy */
class CThresholdRejectionStrategy : public CRejectionStrategy
{
	public:

		/** constructor */
		CThresholdRejectionStrategy() :
			CRejectionStrategy(), m_threshold(0.0) { };

		/** constructor */
		CThresholdRejectionStrategy(float64_t threshold) :
			CRejectionStrategy(), m_threshold(threshold) { };

		virtual ~CThresholdRejectionStrategy() {};

		/** get name */
		virtual const char* get_name() const
		{
			return "ThresholdRejectionStrategy";
		}

		/** returns true if given output set leads to rejection */
		virtual bool reject(SGVector<float64_t> outputs) const
		{
			for (int32_t i=0; i<outputs.vlen; i++)
			{
				if (outputs[i]>m_threshold)
					return false;
			}
			return true;
		}

protected:

		/** threshold */
		float64_t m_threshold;


};

static const float64_t Q_test_statistic_values[10][8] =
{
	/* 10,20,30,40,50,60,70,80,90,100 */
	{0.713,0.683,0.637,0.597,0.551,0.477,0.409,0.325},
	{0.627,0.604,0.568,0.538,0.503,0.450,0.401,0.339},
	{0.539,0.517,0.484,0.456,0.425,0.376,0.332,0.278},
	{0.490,0.469,0.438,0.412,0.382,0.337,0.295,0.246},
	{0.460,0.439,0.410,0.384,0.355,0.312,0.272,0.226},
	{0.437,0.417,0.388,0.363,0.336,0.294,0.256,0.211},
	{0.422,0.403,0.374,0.349,0.321,0.280,0.244,0.201},
	{0.408,0.389,0.360,0.337,0.310,0.270,0.234,0.192},
	{0.397,0.377,0.350,0.326,0.300,0.261,0.226,0.185},
	{0.387,0.368,0.341,0.317,0.292,0.253,0.219,0.179}
};

/** @brief simplified version of Dixon's Q test outlier based
 * rejection strategy. Statistic values are taken from
 * http://www.vias.org/tmdatanaleng/cc_outlier_tests_dixon.html
 * */
class CDixonQTestRejectionStrategy : public CRejectionStrategy
{
	public:

		/** constructor */
		CDixonQTestRejectionStrategy() :
			CRejectionStrategy()
		{
			s_index = 3;
		}

		/** constructor
		 * @param significance_level either 0.001,0.002,0.005,
		 * 0.01,0.02,0.05,0.1 or 0.2
		 */
		CDixonQTestRejectionStrategy(float64_t significance_level) :
			CRejectionStrategy()
		{
			if (significance_level==0.001)
				s_index = 0;
			else if (significance_level==0.002)
				s_index = 1;
			else if (significance_level==0.005)
				s_index = 2;
			else if (significance_level==0.01)
				s_index = 3;
			else if (significance_level==0.02)
				s_index = 4;
			else if (significance_level==0.05)
				s_index = 5;
			else if (significance_level==0.1)
				s_index = 6;
			else if (significance_level==0.2)
				s_index = 7;
			else SG_ERROR("Given significance level is not supported")
		}

		virtual ~CDixonQTestRejectionStrategy()
		{
		}

		/** get name */
		virtual const char* get_name() const
		{
			return "DixonQTestRejectionStrategy";
		}

		/** returns true if given output set leads to rejection */
		virtual bool reject(SGVector<float64_t> outputs) const
		{
			int32_t N = outputs.vlen;
			if (N<10 || N>100)
				SG_ERROR("Given number of classes is not supported.")

			int32_t Ni = N/10 - 1;

			SGVector<float64_t> outputs_local = outputs.clone();
			CMath::qsort(outputs_local);

			float64_t Q = 0.0;
			if (N==10)
				Q = (outputs[N-1]-outputs[N-2])/(outputs[N-1]-outputs[0]);

			if (N>=20)
				Q = (outputs[N-1]-outputs[N-4])/(outputs[N-1]-outputs[2]);

			if (Q>Q_test_statistic_values[Ni][s_index])
				return false;

			return true;
		}

private:

		int32_t s_index;

};

}
#endif
