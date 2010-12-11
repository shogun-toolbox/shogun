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


#include "BanditSingleSparseStump.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/IO/SortedData.h"
#include "classifier/boosting/Algorithms/StumpAlgorithmLSHTC.h"
#include "classifier/boosting/Algorithms/ConstantAlgorithmLSHTC.h"
#include "classifier/boosting/WeakLearners/SingleSparseStumpLearner.h"

#include "classifier/boosting/Bandits/Exp3G2.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id

namespace shogun {

	//REGISTER_LEARNER_NAME(SingleStump, BanditSingleSparseStump)
	REGISTER_LEARNER(BanditSingleSparseStump)

		// ------------------------------------------------------------------------------

		void BanditSingleSparseStump::init() {
			const int numClasses = _pTrainingData->getNumClasses();
			const int numColumns = _pTrainingData->getNumAttributes();
			const int armNumber = _banditAlgo->getArmNumber();

			if ( numColumns < armNumber )
			{
				cerr << "The number of colums smaller than the number of the arms!!!!!!" << endl;
				exit( -1 );
			}

			BaseLearner* pWeakHypothesisSource = 
				BaseLearner::RegisteredLearners().getLearner("SingleSparseStumpLearner");

			_banditAlgo->setArmNumber( numColumns );

			vector<double> initialValues( numColumns );

			for( int i=0; i < numColumns; i++ )
			{
				SingleSparseStumpLearner* singleStump = dynamic_cast<SingleSparseStumpLearner*>( pWeakHypothesisSource->create());

				singleStump->setTrainingData(_pTrainingData);
				double energy = singleStump->run( i );
				double edge = singleStump->getEdge();
				double reward = getRewardFromEdge( (float) edge );

				initialValues[i] = reward;

				delete singleStump;
			}

			_banditAlgo->initialize( initialValues );

	}

	//-------------------------------------------------------------------------------

	float BanditSingleSparseStump::run()
	{

		if ( ! this->_banditAlgo->isInitialized() ) {
			init();
		}

		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( (float) 1.0 / (float)_pTrainingData->getNumExamples() * (float)0.01 );

		vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions

		float tmpThreshold;
		float tmpAlpha;

		float bestEnergy = numeric_limits<float>::max();
		float tmpEnergy;

		StumpAlgorithmLSHTC<float> sAlgo(numClasses);
		sAlgo.initSearchLoop(_pTrainingData);

		/*
		StumpAlgorithm<float> sAlgo(numClasses);
		sAlgo.initSearchLoop(_pTrainingData);
		*/

		float halfTheta;
		if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
			halfTheta = _theta/(float)2.0;
		else
			halfTheta = 0;

		//chose an index accroding to the UCBK policy
		float maxReward = numeric_limits<float>::min();
		float tmpReward = 0.0;
		float bestReward = 0.0;


		_banditAlgo->getKBestAction( _K, _armsForPulling );
		_rewards.resize( _armsForPulling.size() );

		if ( this->_armsForPulling.size() == 0 )
		{
			cout << "error" << endl;
		}

		for( int i = 0; i < (int)_armsForPulling.size(); i++ ) {
			//columnIndices[i] = p.second;			


			const pair<vpReverseIterator,vpReverseIterator> dataBeginEnd = 
				static_cast<SortedData*>(_pTrainingData)->getFileteredReverseBeginEnd( _armsForPulling[i] );

			/*
			const pair<vpIterator,vpIterator> dataBeginEnd = 
			static_cast<SortedData*>(_pTrainingData)->getFileteredBeginEnd( _armsForPulling[i] );
			*/

			const vpReverseIterator dataBegin = dataBeginEnd.first;
			const vpReverseIterator dataEnd = dataBeginEnd.second;

			/*
			const vpIterator dataBegin = dataBeginEnd.first;
			const vpIterator dataEnd = dataBeginEnd.second;
			*/

			// also sets mu, tmpV, and bestHalfEdge
			tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
				halfTheta, &mu, &tmpV);

			tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);
			//update the weights in the UCT tree

			float edge = 0.0;
			for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 
			double reward = this->getRewardFromEdge( edge );
			_rewards[i] = reward;

			if ( _verbose > 3 ) {
				//cout << "\tK = " <<i << endl;
				cout << "\tTempAlpha: " << tmpAlpha << endl;
				cout << "\tTempEnergy: " << tmpEnergy << endl;
				cout << "\tUpdate weight: " << reward << endl;
			}


			if ( (i==0) || (tmpEnergy < bestEnergy && tmpAlpha > 0) )
			{
				// Store it in the current weak hypothesis.
				// note: I don't really like having so many temp variables
				// but the alternative would be a structure, which would need
				// to be inheritable to make things more consistent. But this would
				// make it less flexible. Therefore, I am still undecided. This
				// might change!

				_alpha = tmpAlpha;
				_v = tmpV;
				_selectedColumn = _armsForPulling[i];
				_threshold = tmpThreshold;

				bestEnergy = tmpEnergy;
				bestReward = (float)reward;
			}
		}

		if ( _banditAlgoName == BA_EXP3G2 )
		{
			vector<double> ePayoffs( numColumns );			
			fill( ePayoffs.begin(), ePayoffs.end(), 0.0 );

			for( int i=0; i<_armsForPulling.size(); i++ )
			{
				ePayoffs[_armsForPulling[i]] = _rewards[i];
			}		
			estimatePayoffs( ePayoffs );

			(dynamic_cast<Exp3G2*>(_banditAlgo))->receiveReward( ePayoffs );
		} else {
			for( int i=0; i<_armsForPulling.size(); i++ )
			{
				_banditAlgo->receiveReward( _armsForPulling[i], _rewards[i] );
			}		
		}

		if ( _verbose > 2 ) cout << "Column has been selected: " << _selectedColumn << endl;

		stringstream thresholdString;
		thresholdString << _threshold;
		_id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();

		_reward = bestReward;

		return bestEnergy;
	}

	// ------------------------------------------------------------------------------

	float BanditSingleSparseStump::run( int colIdx )
	{
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions

		float tmpAlpha;

		float bestEnergy = numeric_limits<float>::max();

		StumpAlgorithmLSHTC<float> sAlgo(numClasses);
		sAlgo.initSearchLoop(_pTrainingData);

		float halfTheta;
		if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
			halfTheta = _theta/2.0;
		else
			halfTheta = 0;

		int numOfDimensions = _maxNumOfDimensions;


		const pair<vpReverseIterator,vpReverseIterator> dataBeginEnd = 
			static_cast<SortedData*>(_pTrainingData)->getFileteredReverseBeginEnd( colIdx );


		const vpReverseIterator dataBegin = dataBeginEnd.first;
		const vpReverseIterator dataEnd = dataBeginEnd.second;

		// also sets mu, tmpV, and bestHalfEdge
		_threshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
			halfTheta, &mu, &tmpV);

		bestEnergy = getEnergy(mu, tmpAlpha, tmpV);

		_alpha = tmpAlpha;
		_v = tmpV;
		_selectedColumn = colIdx;

		stringstream thresholdString;
		thresholdString << _threshold;
		_id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();


		return bestEnergy;

	}

	// -----------------------------------------------------------------------

} // end of namespace shogun
