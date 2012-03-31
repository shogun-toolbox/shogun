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
class CThresholdReject : public CRejectionStrategy
{
	public:

		/** constructor */
		CThresholdReject() :
			CRejectionStrategy(), m_threshold(0.0) { };

		/** constructor */
		CThresholdReject(float64_t threshold) :
			CRejectionStrategy(), m_threshold(threshold) { };

		virtual ~CThresholdReject() {};

		/** get name */
		virtual const char* get_name() const
		{
				return "ThresholdReject";
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
}
#endif
