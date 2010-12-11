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
* \file SparseStumpAlgorithm.h The Decision Stump-based algorithms.
*/

#ifndef __SPARSE_STUMP_ALGORITHM_H
#define __SPARSE_STUMP_ALGORITHM_H

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
	class SparseStumpAlgorithm
	{
	public:

		// A couple of useful typedefs
		/**
		* Iterator on Pair. The pair refers to <index, value>.
		*/
		typedef typename vector< pair<int, T> >::iterator       vpIterator;
		/**
		* Const iterator on Pair. The pair refers to <index, value>.
		*/
		typedef typename vector< pair<int, T> >::const_iterator cvpIterator;

		SparseStumpAlgorithm( int numClasses )
		{
			// resize: it's done here to avoid a reallocation
			// for each dimension.

			_edges.resize(numClasses);
			_constantEdges.resize(numClasses);
			_bestEdges.resize(numClasses);
			_weightsPerClass.resize(numClasses);   
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
		void findSingleThreshold(const vpIterator& dataBegin,
			const vpIterator& dataEnd,
			InputData* pData,
			vector<float>& thresholds,
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
		void findSingleThresholdWithInit(const vpIterator& dataBegin,
			const vpIterator& dataEnd,
			InputData* pData,
			vector<float>& thresholds,
			float halfTheta,			
			vector<sRates>* pMu = NULL, vector<float>* pV = NULL);

	private:

		vector<float> _edges; //!< half of the class-wise edges
		vector<float> _constantEdges; //!< half of the class-wise edges of the constant classifier
		vector<float> _bestEdges; //!< half of the edges of the best found threshold.
		vector<float> _weightsPerClass; //!< The half of the total weights per class.
		vector<vpIterator> _bestSplitPoss1; // the iterator of the best split
		vector<vpIterator> _bestPreviousSplitPoss1; // the iterator of the example before the best split
		vector<vpIterator> _bestSplitPoss2; // the iterator of the best split
		vector<vpIterator> _bestPreviousSplitPoss2; // the iterator of the example before the best split
	};

	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	template <typename T> 
	void SparseStumpAlgorithm<T>::initSearchLoop(InputData* pData)
	{
		const int numClasses = pData->getNumClasses();

		ConstantAlgorithm cAlgo;
		cAlgo.findConstantWeightsEdges(pData,_weightsPerClass,_constantEdges);

		for (int l = 0; l < numClasses; ++l)
		{
		  _weightsPerClass[l] *= 2.0;
		  _constantEdges[l] *= 2.0;
		}
	} // end of initSearchLoop

	//////////////////////////////////////////////////////////////////////////

	template <typename T> 
	void SparseStumpAlgorithm<T>::findSingleThreshold(const vpIterator& dataBegin,
			const vpIterator& dataEnd,
			InputData* pData,
			vector<float>& thresholds,
			float halfTheta,
			vector<sRates>* pMu, vector<float>* pV)
	{ 
		// initialize halfEdges to the constant classifier's half edges 
		initSearchLoop(pData);

		// findSingleThreshold after initialization
		findSingleThresholdWithInit(dataBegin,dataEnd,pData,thresholds,halfTheta,pMu,pV);

	} // end of findSingleThreshold

	template <typename T> 
	void SparseStumpAlgorithm<T>::findSingleThresholdWithInit
		(const vpIterator& dataBegin,
			const vpIterator& dataEnd,
			InputData* pData,
			vector<float>& thresholds,
			float halfTheta,
			vector<sRates>* pMu, vector<float>* pV)
	{ 
		const int numClasses = pData->getNumClasses();

		vpIterator currentSplitPos1; // the iterator of the currently examined example
		vpIterator previousSplitPos1; // the iterator of the example before the current example
		vpIterator bestSplitPos1; // the iterator of the best split
		vpIterator bestPreviousSplitPos1; // the iterator of the example before the best split

		vpIterator currentSplitPos2; // the iterator of the currently examined example
		vpIterator previousSplitPos2; // the iterator of the example before the current example
		vpIterator bestSplitPos2; // the iterator of the best split
		vpIterator bestPreviousSplitPos2; // the iterator of the example before the best split


		// initialize halfEdges to the constant classifier's half edges 
		copy(_constantEdges.begin(), _constantEdges.end(), _edges.begin());

		//store the thresholds
		thresholds.resize(2);
		thresholds[0] = thresholds[1] = 0.0;

		float currEdge = 0;
		float bestEdge = -numeric_limits<float>::max();
		vector<Label>::const_iterator lIt;
		vector<float> tmpEdges(numClasses);

		// find the best threshold (cutting point)
		// at the first split we have
		// first split: x | x x x x x x x x ..
		//    previous -^   ^- current
		for( currentSplitPos1 = previousSplitPos1 = dataBegin, ++currentSplitPos1;
			currentSplitPos1 != dataEnd; 
			previousSplitPos1 = currentSplitPos1, ++currentSplitPos1)
		{
			vector<Label>& labels1 = pData->getLabels(previousSplitPos1->first);

			// recompute edges at the next point
			////// Bottleneck BEGIN
			for (lIt = labels1.begin(); lIt != labels1.end(); ++lIt )
				_edges[ lIt->idx ] -= lIt->weight * lIt->y;
			////// Bottleneck END

			copy(_edges.begin(), _edges.end(), tmpEdges.begin());

			// points with the same value of data: to skip because we cannot find a cutting point here!
			// so we only do the cutting if there is a "hole":
			if ( previousSplitPos1->second != currentSplitPos1->second ) 
			{
				for( currentSplitPos2 = previousSplitPos2 = currentSplitPos1, ++currentSplitPos2;
					currentSplitPos2 != dataEnd; 
					previousSplitPos2 = currentSplitPos2, ++currentSplitPos2)				
				{
					vector<Label>& labels2 = pData->getLabels(previousSplitPos2->first);

					// recompute edges at the next point
					////// Bottleneck BEGIN
					for (lIt = labels2.begin(); lIt != labels2.end(); ++lIt )
						tmpEdges[ lIt->idx ] -= lIt->weight * lIt->y;
					////// Bottleneck END

					if ( previousSplitPos2->second != currentSplitPos2->second ) 
					{
						currEdge = 0;

						////// Bottleneck BEGIN
						for (int l = 0; l < numClasses; ++l) { 
							// flip the class-wise edge if it is negative
							// but store the flipping bit only at the end (below**)
							if ( tmpEdges[l] > 0 )
								currEdge += tmpEdges[l];
							else
								currEdge -= tmpEdges[l];
						}
						////// Bottleneck END

						// the current edge is the new maximum
						if (currEdge > bestEdge)
						{
							bestEdge = currEdge;
							bestSplitPos1 = currentSplitPos1; 
							bestPreviousSplitPos1 = previousSplitPos1; 
							bestSplitPos2 = currentSplitPos2; 
							bestPreviousSplitPos2 = previousSplitPos2; 

							for (int l = 0; l < numClasses; ++l)
								_bestEdges[l] = tmpEdges[l];
						}
					} //endif
				} // endfor
			} // endif
		} //endfor

		// If we found a valid stump in this dimension
		if (bestEdge >  -numeric_limits<float>::max()) 
		{
			thresholds[0] = static_cast<float>( bestPreviousSplitPos1->second + 
				bestSplitPos1->second ) / 2;

			thresholds[1] = static_cast<float>( bestPreviousSplitPos2->second + 
				bestSplitPos2->second ) / 2;

			// Fill the mus if present. This could have been done in the threshold loop, 
			// but here is done just once
			if ( pMu ) 
			{
				for (int l = 0; l < numClasses; ++l)
				{
					// **here
					if (_bestEdges[l] > 0)
						(*pV)[l] = +1;
					else
						(*pV)[l] = -1;

					(*pMu)[l].classIdx = l;

					(*pMu)[l].rPls  = _weightsPerClass[l] + (*pV)[l] * _bestEdges[l];
					(*pMu)[l].rMin  = _weightsPerClass[l] - (*pV)[l] * _bestEdges[l];
					(*pMu)[l].rZero = (*pMu)[l].rPls + (*pMu)[l].rMin; // == weightsPerClass[l]
				}
			}
			//cout << 2 * bestHalfEdge << endl << flush;
		}

	} // end of findSingleThresholdWithInit


	//////////////////////////////////////////////////////////////////////////

} // end of namespace MultiBoost

#endif // __SPARSE_STUMP_ALGORITHM_H
