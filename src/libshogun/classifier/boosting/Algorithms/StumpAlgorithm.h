/*
*
*    MultiBoost - Multi-purpose boosting package
*
*    Copyright (C) 2010   AppStat group
*                         Laboratoire de l'Accelerateur Lineaire
*                         Universite Paris-Sud, 11, CNRS
*
*    This file is part of the MultiBoost library
*
*    This library is free software; you can redistribute it 
*    and/or modify it under the terms of the GNU General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    General Public License for more details.
*
*    You should have received a copy of the GNU General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 51 Franklin St, 5th Floor, Boston, MA 02110-1301 USA
*
*    Contact: Balazs Kegl (balazs.kegl@gmail.com)
*             Norman Casagrande (nova77@gmail.com)
*             Robert Busa-Fekete (busarobi@gmail.com)
*
*    For more information and up-to-date version, please visit
*        
*                       http://www.multiboost.org/
*
*/


/**
* \file StumpAlgorithm.h The Decision Stump-based algorithms.
*/

#ifndef __STUMP_ALGORITHM_H
#define __STUMP_ALGORITHM_H

#include <vector>
#include <cassert>

#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/Others/Rates.h"
#include "classifier/boosting/IO/NameMap.h"
#include "classifier/boosting/Algorithms/ConstantAlgorithm.h"

using namespace std;

namespace MultiBoost {

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	/**
	* Class specialized in solving decision stump-type algorithms.
	* A decision stump is a decision tree with a single level.
	*/
	template <typename T>
	class StumpAlgorithm
	{
	public:

		// A couple of useful typedefs
		/**
		* Iterator on Pair. The pair refers to <index, value>.
		*/
		typedef typename vector< pair<int, T> >::const_iterator       vpIterator;
		/**
		* Const iterator on Pair. The pair refers to <index, value>.
		*/
		typedef typename vector< pair<int, T> >::const_iterator cvpIterator;

		StumpAlgorithm( int numClasses )
		{
			// resize: it's done here to avoid a reallocation
			// for each dimension.

			_halfEdges.resize(numClasses);
			_constantHalfEdges.resize(numClasses);
			_bestHalfEdges.resize(numClasses);
			_halfWeightsPerClass.resize(numClasses);   
		}

		/**
		* Initilizes halfWeightsPerClass and constantHalfEdges for subsequent calls
		* to findSingleThresholdWithInit or findMultiThresholdWithInit
		* \param pData The pointer to the data class.
		* \date 03/07/2006
		*/
		void initSearchLoop(InputData* pData);

		/**
		* Find the optimal threshold that maximizes
		* the edge (or minimizes the error) on the given weighted data.
		* In case of multi-class data, the threshold will minimize the
		* error over all the classes:
		* \image html stumps_sg.gif "The optimal threshold on a simple multi class problem"
		* \image latex stumps_sg.eps "The optimal threshold on a simple multi class problem" width=10cm
		* In the figure, the vector pV represents the arrows, and pMu is the per-class
		* error.
		* \param dataBegin The iterator to the beginning of the data.
		* \param dataEnd The iterator to the end of the data.
		* \param pData The pointer to the original data class. Used to obtain the label of
		* the example from its index.
		* \param pMu The The class-wise rates to update. (if provided)
		* \param pV The alignment vector to update. (if provided)
		* \see sRates
		* \return The threshold found.
		* \remark The algorithm (suboptimally) maximizes the Edge. This 
		* is slightly different from the standard AdaBoost which maximizes
		* the energy even in the decision stump algorithm. This is the result
		* of a trade-off between speed and full adherence to the theory.
		* \date 11/11/2005
		*/
		float findSingleThreshold(const vpIterator& dataBegin,
			const vpIterator& dataEnd,
			InputData* pData,
			float halfTheta = 0.0,
			vector<sRates>* pMu = NULL, vector<float>* pV = NULL);

		/**
		* Same as findSingleThreshold, but the caller has to call initSearchLoop
		* explicitly before the loop over the dimensions (features). It is more efficient 
		* than findSingleThreshold since findSingleThreshold calls initSearchLoop 
		* in each iteration. However, it can only be used if
		* the data points and their weights do not change between initSearchLoop and subsequent
		* calls to findSingleThreshold. See SingleStumpLearner::run for an example.
		* \param dataBegin The iterator to the beginning of the data.
		* \param dataEnd The iterator to the end of the data.
		* \param pData The pointer to the original data class. Used to obtain the label of
		* the example from its index.
		* \param pMu The The class-wise rates to update. (if provided)
		* \param pV The alignment vector to update. (if provided)
		* \see sRates
		* \see findSingleThreshold
		* \return The threshold found.
		* \date 03/07/2006
		*/
		float findSingleThresholdWithInit(const vpIterator& dataBegin,
			const vpIterator& dataEnd,
			InputData* pData,
			float halfTheta,
			vector<sRates>* pMu = NULL, vector<float>* pV = NULL);

		/**
		* Find the optimal thresholds (one for each class) that maximizes
		* the edge (or minimizes the error) on the given data weighted data.
		* \image html stumps_ml.gif "The optimal thresholds (one for each class) on a simple multi class problem"
		* \image latex stumps_ml.eps "The optimal thresholds (one for each class) on a simple multi class problem" width=10cm
		* In the figure, the vector pV represents the arrows, and pMu is the per-class
		* error.
		* \param dataBegin The iterator to the beginning of the data.
		* \param dataEnd The iterator to the end of the data.
		* \param pData The pointer to the original data class. Used to obtain the label of
		* the example from its index.
		* \param thresholds The thresholds to update.
		* \param pMu The The class-wise rates to update (if provided).
		* \param pV The alignment vector to update (if provided).
		* \remark The algorithm (suboptimally) maximizes the Edge. This 
		* is slightly different from the standard AdaBoost which maximizes
		* the energy even in the decision stump algorithm. This is the result
		* of a trade-off between speed and full adherence to the theory.
		* \see sRates
		* \date 11/11/2005
		*/
		void findMultiThresholds(const vpIterator& dataBegin,
			const vpIterator& dataEnd,
			InputData* pData, vector<float>& thresholds,
			vector<sRates>* pMu = NULL, vector<float>* pV = NULL);

		/**
		* Same as findMultiThreshold, but the caller has to call initSearchLoop
		* explicitly before the loop over the dimensions (features). It is more efficient 
		* than findMultiThreshold since findMultiThreshold calls initSearchLoop 
		* in each iteration. However, it can only be used if
		* the data points and their weights do not change between initSearchLoop and subsequent
		* calls to findMultiThreshold. See MultiStumpLearner::run for an example.
		* \param dataBegin The iterator to the beginning of the data.
		* \param dataEnd The iterator to the end of the data.
		* \param pData The pointer to the original data class. Used to obtain the label of
		* the example from its index.
		* \param thresholds The thresholds to update.
		* \param pMu The The class-wise rates to update. (if provided)
		* \param pV The alignment vector to update. (if provided)
		* \see sRates
		* \see findSingleThreshold
		* \return The threshold found.
		* \date 05/07/2006
		*/
		void findMultiThresholdsWithInit(const vpIterator& dataBegin,
			const vpIterator& dataEnd,
			InputData* pData, 
			vector<float>& thresholds,
			vector<sRates>* pMu = NULL, vector<float>* pV = NULL);

	private:

		vector<float> _halfEdges; //!< half of the class-wise edges
		vector<float> _constantHalfEdges; //!< half of the class-wise edges of the constant classifier
		vector<float> _bestHalfEdges; //!< half of the edges of the best found threshold.
		vector<float> _halfWeightsPerClass; //!< The half of the total weights per class.
		vector<vpIterator> _bestSplitPoss; // the iterator of the best split
		vector<vpIterator> _bestPreviousSplitPoss; // the iterator of the example before the best split

	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	template <typename T> 
	void StumpAlgorithm<T>::initSearchLoop(InputData* pData)
	{
		ConstantAlgorithm cAlgo;
		cAlgo.findConstantWeightsEdges(pData,_halfWeightsPerClass,_constantHalfEdges);
	} // end of initSearchLoop

	//////////////////////////////////////////////////////////////////////////

	template <typename T> 
	float StumpAlgorithm<T>::findSingleThreshold(const vpIterator& dataBegin,
		const vpIterator& dataEnd,
		InputData* pData,
		float halfTheta,
		vector<sRates>* pMu, vector<float>* pV)
	{ 
		// initialize halfEdges to the constant classifier's half edges 
		initSearchLoop(pData);

		// findSingleThreshold after initialization
		return findSingleThresholdWithInit(dataBegin,dataEnd,pData,pMu,pV,halfTheta);

	} // end of findSingleThreshold

	template <typename T> 
	float StumpAlgorithm<T>::findSingleThresholdWithInit
		(const vpIterator& dataBegin,const vpIterator& dataEnd,
		InputData* pData, float halfTheta, vector<sRates>* pMu, vector<float>* pV)
	{ 
		const int numClasses = pData->getNumClasses();

		vpIterator currentSplitPos; // the iterator of the currently examined example
		vpIterator previousSplitPos; // the iterator of the example before the current example
		vpIterator bestSplitPos; // the iterator of the best split
		vpIterator bestPreviousSplitPos; // the iterator of the example before the best split

		// initialize halfEdges to the constant classifier's half edges 
		copy(_constantHalfEdges.begin(), _constantHalfEdges.end(), _halfEdges.begin());

		float currHalfEdge = 0;
		float bestHalfEdge = -numeric_limits<float>::max();
		vector<Label>::const_iterator lIt;

		// find the best threshold (cutting point)
		// at the first split we have
		// first split: x | x x x x x x x x ..
		//    previous -^   ^- current
		for( currentSplitPos = previousSplitPos = dataBegin, ++currentSplitPos;
			currentSplitPos != dataEnd; 
			previousSplitPos = currentSplitPos, ++currentSplitPos)
		{
			vector<Label>& labels = pData->getLabels(previousSplitPos->first);

			// recompute halfEdges at the next point
			////// Bottleneck BEGIN
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
				_halfEdges[ lIt->idx ] -= lIt->weight * lIt->y;
			////// Bottleneck END

			// points with the same value of data: to skip because we cannot find a cutting point here!
			// so we only do the cutting if there is a "hole":
			if ( previousSplitPos->second != currentSplitPos->second ) 
			{

				currHalfEdge = 0;

				////// Bottleneck BEGIN
				if ( nor_utils::is_zero(halfTheta) ) { // we save an "if" in the loop, 20% faster
					for (int l = 0; l < numClasses; ++l) { 
						// flip the class-wise edge if it is negative
						// but store the flipping bit only at the end (below**)
						if ( _halfEdges[l] > 0 )
							currHalfEdge += _halfEdges[l];
						else
							currHalfEdge -= _halfEdges[l];
					}
				}
				else {
					for (int l = 0; l < numClasses; ++l) { 
						// flip the class-wise edge if it is negative
						// but store the flipping bit only at the end (below**)
						if ( _halfEdges[l] > halfTheta )
							currHalfEdge += _halfEdges[l];
						else if ( _halfEdges[l] < -halfTheta )
							currHalfEdge -= _halfEdges[l];
					}
				}
				////// Bottleneck END

				// the current edge is the new maximum
				if (currHalfEdge > bestHalfEdge)
				{
					bestHalfEdge = currHalfEdge;
					bestSplitPos = currentSplitPos; 
					bestPreviousSplitPos = previousSplitPos; 

					for (int l = 0; l < numClasses; ++l)
						_bestHalfEdges[l] = _halfEdges[l];
				}
			}
		}

		// If we found a valid stump in this dimension
		if (bestHalfEdge >  -numeric_limits<float>::max()) 
		{
			float threshold = static_cast<float>( bestPreviousSplitPos->second + 
				bestSplitPos->second ) / 2;

			// Fill the mus if present. This could have been done in the threshold loop, 
			// but here is done just once
			if ( pMu ) 
			{
				for (int l = 0; l < numClasses; ++l)
				{
					// **here
					if (_bestHalfEdges[l] > 0)
						(*pV)[l] = +1;
					else
						(*pV)[l] = -1;

					(*pMu)[l].classIdx = l;

					(*pMu)[l].rPls  = _halfWeightsPerClass[l] + (*pV)[l] * _bestHalfEdges[l];
					(*pMu)[l].rMin  = _halfWeightsPerClass[l] - (*pV)[l] * _bestHalfEdges[l];
					(*pMu)[l].rZero = (*pMu)[l].rPls + (*pMu)[l].rMin; // == weightsPerClass[l]
				}
			}
			//cout << 2 * bestHalfEdge << endl << flush;
			return threshold;
		}
		else
			return numeric_limits<float>::signaling_NaN();

	} // end of findSingleThresholdWithInit

	//////////////////////////////////////////////////////////////////////////

	template <typename T> 
	void StumpAlgorithm<T>::findMultiThresholds(const vpIterator& dataBegin,
		const vpIterator& dataEnd,
		InputData* pData, vector<float>& thresholds,
		vector<sRates>* pMu, vector<float>* pV)
	{ 
		// initialize halfEdges to the constant classifier's half edges 
		initSearchLoop(pData);

		// findMultiThreshold after initialization
		return findMultiThresholdsWithInit(dataBegin,dataEnd,pData,thresholds,pMu,pV);

	} // end of findMultiThresholds

	//////////////////////////////////////////////////////////////////////////

	template <typename T> 
	void StumpAlgorithm<T>::findMultiThresholdsWithInit
		(const vpIterator& dataBegin, const vpIterator& dataEnd,
		InputData* pData, vector<float>& thresholds, 
		vector<sRates>* pMu, vector<float>* pV)
	{ 
		const int numClasses = pData->getNumClasses();

		vpIterator currentSplitPos; // the iterator of the currently examined example
		vpIterator previousSplitPos; // the iterator of the example before the current example

		// Initializing halfEdges to the constant classifier's half edges 
		copy(_constantHalfEdges.begin(), _constantHalfEdges.end(), _halfEdges.begin());

		// Initializing bestHalfEdges, thresholds, and pV to the constant classifier 
		copy(_constantHalfEdges.begin(), _constantHalfEdges.end(), _bestHalfEdges.begin());
		for (int l = 0; l < numClasses; ++l)
			thresholds[l] = -numeric_limits<float>::max(); // constant cut

		bool alignAlloc = false;
		if (pV == NULL)
		{
			pV = new vector<float>(numClasses);
			alignAlloc = true;
		}

		for (int l = 0; l < numClasses; ++l)
		{
			if (_halfEdges[l] > 0) // constant cut
				(*pV)[l] = 1;
			else
				(*pV)[l] = -1;
		}


		vector<Label>::const_iterator lIt;

		// find the best threshold (cutting point)
		// at the first split we have
		// first split: x | x x x x x x x x ..
		//    previous -^   ^- current
		for( currentSplitPos = previousSplitPos = dataBegin, ++currentSplitPos;
			currentSplitPos != dataEnd; 
			previousSplitPos = currentSplitPos, ++currentSplitPos)
		{
			// recompute halfEdges at the next point
			// this is the bottleneck
			vector<Label>& labels = pData->getLabels(previousSplitPos->first);

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
				_halfEdges[ lIt->idx ] -= lIt->weight * lIt->y;

			// points with the same value of data: to skip because we cannot find a cutting point here!
			// so we only do the cutting if there is a "hole":
			if ( previousSplitPos->second != currentSplitPos->second ) 
			{
				for (int l = 0; l < numClasses; ++l)
				{ 
					if (_halfEdges[l] > 0)
					{
						// the current edge is the new maximum
						if (_halfEdges[l] > _bestHalfEdges[l] * (*pV)[l]) 
						{
							(*pV)[l] = 1;
							_bestHalfEdges[l] = _halfEdges[l];
							thresholds[l] = static_cast<float>( previousSplitPos->second +
								currentSplitPos->second ) / 2;
						}
					}
					else
					{
						// the current edge is the new maximum
						if (-_halfEdges[l] > _bestHalfEdges[l] * (*pV)[l]) 
						{
							(*pV)[l] = -1;
							_bestHalfEdges[l] = _halfEdges[l];
							thresholds[l] = static_cast<float>( previousSplitPos->second +
								currentSplitPos->second ) / 2;
						}
					}
				}
			}
		}

		// Fill the mus if present. This could have been done in the threshold loop, 
		// but here is done just once
		if ( pMu ) 
		{
			for (int l = 0; l < numClasses; ++l)
			{	    
				(*pMu)[l].classIdx = l;

				(*pMu)[l].rPls  = _halfWeightsPerClass[l] + (*pV)[l] * _bestHalfEdges[l];
				(*pMu)[l].rMin  = _halfWeightsPerClass[l] - (*pV)[l] * _bestHalfEdges[l];
				(*pMu)[l].rZero = (*pMu)[l].rPls + (*pMu)[l].rMin; // == weightsPerClass[l]
			}
		}

		if (alignAlloc)
			delete pV;

	} // end of findMultiThresholdWithInit

	//////////////////////////////////////////////////////////////////////////

} // end of namespace MultiBoost

#endif // __STUMP_ALGORITHM_H
