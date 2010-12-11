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


#include "UCBKSingleStumpLearner.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/IO/SortedData.h"
#include "classifier/boosting/Algorithms/StumpAlgorithm.h"
#include "classifier/boosting/Algorithms/ConstantAlgorithm.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id
#include <math.h> //for log
#include <queue>

namespace MultiBoost {

	//REGISTER_LEARNER_NAME(SingleStump, UCBKSingleStumpLearner)
	REGISTER_LEARNER(UCBKSingleStumpLearner)

	vector< int > UCBKSingleStumpLearner::_T; // the number of a feature has been selected 
	int UCBKSingleStumpLearner::_numOfCalling = 0; //number of the single stump learner had been called
	vector< float > UCBKSingleStumpLearner::_X; // the everage reward of a feature
	int UCBKSingleStumpLearner::_K = 0; // number of columns to be selected

	void UCBKSingleStumpLearner::declareArguments(nor_utils::Args& args)
	{
		BaseLearner::declareArguments(args);

		args.declareArgument("updaterule", 
			"The update weights in the UCT can be the 1-sqrt( 1- edge^2 ) [edge]\n"
			"  or the alpha [alphas]\n"
			"  Default is the first one\n",
			1, "<type>");

		args.declareArgument("rsample", 
			"Number of features to be considered\n"
			"  Default is one\n",
			1, "<K>");

	}

	// ------------------------------------------------------------------------------

	void UCBKSingleStumpLearner::initLearningOptions(const nor_utils::Args& args)
	{
		BaseLearner::initLearningOptions(args);


		string updateRule = "";
		if ( args.hasArgument( "updaterule" ) )
			args.getValue("updaterule", 0, updateRule );   

		if ( updateRule.compare( "edge" ) == 0 )
			_updateRule = EDGE_SQUARE;
		else if ( updateRule.compare( "logedge" ) == 0 )
			_updateRule = LOGEDGE;
		else if ( updateRule.compare( "alphas" ) == 0 )
			_updateRule = ALPHAS;
		else if ( updateRule.compare( "edgesquare" ) == 0 )
			_updateRule = ESQUARE;
		else {
			cerr << "Unknown update rule in ProductLearnerUCT (set to default [edge]" << endl;
			_updateRule = EDGE_SQUARE;
		}

		if ( args.hasArgument( "rsample" ) ){
			_K = args.getValue<int>("rsample", 0);
		}

	}

	//-------------------------------------------------------------------------------

	void UCBKSingleStumpLearner::init() {
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();


		if ( _K > numColumns ) {
			UCBKSingleStumpLearner::_K = numColumns;
			cout << "There is no sense to use the UCBK heuristics!!!!!" << endl;
		} 

		if ( _verbose > 2 ) {
			cout << "The number of column in each iteration to be chosen: " << UCBKSingleStumpLearner::_K << endl;
		}


		UCBKSingleStumpLearner::_numOfCalling = 1;
		UCBKSingleStumpLearner::_T.resize( _pTrainingData->getNumAttributes() );
		UCBKSingleStumpLearner::_X.resize( _pTrainingData->getNumAttributes() );

		float tmpAlpha;
		float tmpEnergy;

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions

		float tmpThreshold;

		float bestEnergy = numeric_limits<float>::max();

		StumpAlgorithm<float> sAlgo(numClasses);
		sAlgo.initSearchLoop(_pTrainingData);

		float halfTheta;
		if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
			halfTheta = _theta/2.0;
		else
			halfTheta = 0;

		for (int j = 0; j < numColumns; ++j)
		{
			const pair<vpIterator,vpIterator> dataBeginEnd = 
				static_cast<SortedData*>(_pTrainingData)->getFileteredBeginEnd(j);


			const vpIterator dataBegin = dataBeginEnd.first;
			const vpIterator dataEnd = dataBeginEnd.second;

			// also sets mu, tmpV, and bestHalfEdge
			tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
				halfTheta, &mu, &tmpV);

			//if (tmpThreshold == tmpThreshold) // tricky way to test Nan
			//{ 

			//update the weights in the UCT tree
			double updateWeight = 0.0;
			if ( _updateRule == EDGE_SQUARE ) {
				float edge = 0.0;
				for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 
				updateWeight = 1 - sqrt( 1 - ( edge * edge ) );
			} else if ( _updateRule == ALPHAS ) {
				tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);
				//double alpha = this->getAlpha();
				updateWeight = tmpAlpha;
			} else if ( _updateRule == ESQUARE ) {
				float edge = 0.0;
				for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 
				updateWeight = edge * edge;
			}




				UCBKSingleStumpLearner::_T[j] = 1;
				UCBKSingleStumpLearner::_X[j] = updateWeight;
			//} // tmpThreshold == tmpThreshold
		}
	}


	// ------------------------------------------------------------------------------

	float UCBKSingleStumpLearner::run()
	{
		if ( UCBKSingleStumpLearner::_numOfCalling == 0 ) {
			init();
		}

		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();
		
		// the K parameter corresponds to the _maxNumOfDimensions, you can set this parameter by the rsampple 
		// I place this part of code here because at the serialization the init isn't to be called
		if ( UCBKSingleStumpLearner::_K == 0 ) {
			if ( _maxNumOfDimensions == numeric_limits<int>::max() ) { 
				UCBKSingleStumpLearner::_K = 1;
			} else {
				if ( _maxNumOfDimensions > numColumns ) {
					UCBKSingleStumpLearner::_K = numColumns;
					cout << "There is no sense to use the UCBK heuristics!!!!!" << endl;
				} else {
					UCBKSingleStumpLearner::_K = _maxNumOfDimensions;
				}
			}
		}

		UCBKSingleStumpLearner::_numOfCalling++; 


		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions

		float tmpThreshold;
		float tmpAlpha;

		float bestEnergy = numeric_limits<float>::max();
		float tmpEnergy;

		StumpAlgorithm<float> sAlgo(numClasses);
		sAlgo.initSearchLoop(_pTrainingData);

		float halfTheta;
		if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
			halfTheta = _theta/2.0;
		else
			halfTheta = 0;

		//chose an index accroding to the UCBK policy
		int UCBKcolumnIndex = 0;
		float maxReward = numeric_limits<float>::min();
		float tmpReward = 0.0;
		float bestReward = 0.0;

		priority_queue< pair< float, int > > rewardArray;

		for (int j = 0; j < numColumns; ++j) {
			tmpReward = UCBKSingleStumpLearner::_X[j] / (float) UCBKSingleStumpLearner::_T[j];
			tmpReward += sqrt( ( 2 * log( (float)UCBKSingleStumpLearner::_numOfCalling ) ) / UCBKSingleStumpLearner::_T[j] );
			
			if ( tmpReward != tmpReward ) continue;

			pair< float, int > p( tmpReward, j );
			rewardArray.push( p );
			//cout << tmpReward << " " << j << endl;
		}
		/*
		while ( ! rewardArray.empty() ) {
			pair< float, int > p = rewardArray.top();
			rewardArray.pop();
			cout << p.first << "\t" << p.second << endl;
		}
		*/


		if ( rewardArray.empty() ) {
			cerr << "UCBKSingleStumpLearner::Empty the reward array!!!" << endl;
		}


		for( int i = 0; i < UCBKSingleStumpLearner::_K; i++ ) {
			pair< float, int > p = rewardArray.top();
			tmpReward = p.first;
			rewardArray.pop();

			//columnIndices[i] = p.second;			

			const pair<vpIterator,vpIterator> dataBeginEnd = 
				static_cast<SortedData*>(_pTrainingData)->getFileteredBeginEnd(p.second);


			const vpIterator dataBegin = dataBeginEnd.first;
			const vpIterator dataEnd = dataBeginEnd.second;

			// also sets mu, tmpV, and bestHalfEdge
			tmpThreshold = sAlgo.findSingleThresholdWithInit(dataBegin, dataEnd, _pTrainingData, 
				halfTheta, &mu, &tmpV);

			// small inconsistency compared to the standard algo (but a good
			// trade-off): in findThreshold we maximize the edge (suboptimal but
			// fast) but here (among dimensions) we minimize the energy.





			 //if (tmpThreshold == tmpThreshold) // tricky way to test Nan
			 //{ 
				// small inconsistency compared to the standard algo (but a good
				// trade-off): in findThreshold we maximize the edge (suboptimal but
				// fast) but here (among dimensions) we minimize the energy.
				tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);
				//update the weights in the UCT tree
				double updateWeight = 0.0;
				if ( _updateRule == EDGE_SQUARE ) {
					float edge = 0.0;
					for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 
					updateWeight = 1 - sqrt( 1 - ( edge * edge ) );
				} else if ( _updateRule == ALPHAS ) {
					//double alpha = this->getAlpha();
					updateWeight = tmpAlpha;
				} else if ( _updateRule == LOGEDGE ) {
					//double alpha = this->getAlpha();
					float edge = 0.0;
					for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 
					if ( edge < 1.0 ) {
						updateWeight = - log( 1 - ( edge * edge ) );
					} else {
						updateWeight = - log( 1 - numeric_limits< double >::epsilon() );
					}
				} else if ( _updateRule == ESQUARE ) {
					float edge = 0.0;
					for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 
					updateWeight = edge * edge;
				}


				if ( _verbose > 3 ) {
					cout << "\tK = " <<i << endl;
					cout << "\tTempAlpha: " << tmpAlpha << endl;
					cout << "\tTempEnergy: " << tmpEnergy << endl;
					cout << "\tUpdate weight: " << updateWeight << endl;
				}


				//update the estimated rewards in the badit algorithm
				
				UCBKSingleStumpLearner::_X[p.second] += updateWeight;
				UCBKSingleStumpLearner::_T[p.second]++;


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
				   _selectedColumn = p.second;
				   _threshold = tmpThreshold;
				   //if the thresold is infinity or nan we set the threshol to max
				   /*
				   if ( (tmpThreshold == tmpThreshold) && ( tmpThreshold < numeric_limits<float>::max() ) ) {
					   _threshold = tmpThreshold;
				   } else {
   					   _threshold = numeric_limits<float>::max();
				   }
				   */

				   bestEnergy = tmpEnergy;
				   bestReward = updateWeight;
				}
			 //} // tmpThreshold == tmpThreshold



		}


		if ( _verbose > 2 ) cout << "Column to be selected: " << _selectedColumn << endl;

		stringstream thresholdString;
		thresholdString << _threshold;
		_id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn) + thresholdString.str();

		_reward = bestReward;
		
		return bestEnergy;

	}

	// ------------------------------------------------------------------------------

	float UCBKSingleStumpLearner::phi(float val, int /*classIdx*/) const
	{
		if (val > _threshold)
			return +1;
		else
			return -1;
	}

	// ------------------------------------------------------------------------------

	float UCBKSingleStumpLearner::phi(InputData* pData,int pointIdx) const
	{
		return phi(pData->getValue(pointIdx,_selectedColumn),0);
	}

	// -----------------------------------------------------------------------

	void UCBKSingleStumpLearner::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class method
		FeaturewiseLearner::save(outputStream, numTabs);

		// save selectedCoulumn
		outputStream << Serialization::standardTag("threshold", _threshold, numTabs) << endl;
		
		outputStream << Serialization::standardTag("reward", _reward, numTabs) << endl;
		//outputStream << Serialization::standardTag("columnIndex", _selectedColumn, numTabs) << endl;
	}

	// -----------------------------------------------------------------------

	void UCBKSingleStumpLearner::load(nor_utils::StreamTokenizer& st)
	{
		// Calling the super-class method
		if ( UCBKSingleStumpLearner::_numOfCalling == 0 ) {
			const int numColumns = _pTrainingData->getNumAttributes();
			UCBKSingleStumpLearner::_T.resize( numColumns );
			UCBKSingleStumpLearner::_X.resize( numColumns );

			fill( UCBKSingleStumpLearner::_T.begin(), UCBKSingleStumpLearner::_T.end(), 1 );
			fill( UCBKSingleStumpLearner::_X.begin(), UCBKSingleStumpLearner::_X.end(), 0.0 );

		}

		FeaturewiseLearner::load(st);

		_threshold = UnSerialization::seekAndParseEnclosedValue<float>(st, "threshold");
	

		stringstream thresholdString;
		thresholdString << _threshold;
		_id = _id + thresholdString.str();

		_reward = UnSerialization::seekAndParseEnclosedValue<float>(st, "reward");
			UCBKSingleStumpLearner::_X[ _selectedColumn ] += _reward;
		UCBKSingleStumpLearner::_T[ _selectedColumn ]++;


		UCBKSingleStumpLearner::_numOfCalling++;
	}

	// -----------------------------------------------------------------------

	void UCBKSingleStumpLearner::subCopyState(BaseLearner *pBaseLearner)
	{
		FeaturewiseLearner::subCopyState(pBaseLearner);

		UCBKSingleStumpLearner* pUCBKSingleStumpLearner =
			dynamic_cast<UCBKSingleStumpLearner*>(pBaseLearner);

		pUCBKSingleStumpLearner->_threshold = _threshold;
		pUCBKSingleStumpLearner->_reward = _reward;
	}

	// -----------------------------------------------------------------------

	//void UCBKSingleStumpLearner::getStateData( vector<float>& data, const string& /*reason*/, InputData* pData )
	//{
	//   const int numClasses = pData->getNumClasses();
	//   const int numExamples = pData->getNumExamples();
	//
	//   // reason ignored for the moment as it is used for a single task
	//   data.resize( numClasses + numExamples );
	//
	//   int pos = 0;
	//
	//   for (int l = 0; l < numClasses; ++l)
	//      data[pos++] = _v[l];
	//
	//   for (int i = 0; i < numExamples; ++i)
	//      data[pos++] = UCBKSingleStumpLearner::phi( pData->getValue( i, _selectedColumn), 0 );
	//}

	// -----------------------------------------------------------------------

} // end of namespace MultiBoost
