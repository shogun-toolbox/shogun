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



#include "BanditSingleStumpLearner.h"

#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/IO/SortedData.h"
#include "classifier/boosting/Algorithms/StumpAlgorithm.h"
#include "classifier/boosting/Algorithms/ConstantAlgorithm.h"
#include "classifier/boosting/WeakLearners/SingleStumpLearner.h"

#include <limits> // for numeric_limits<>
#include <sstream> // for _id
#include <math.h> //for log
#include <queue>

#include "classifier/boosting/Bandits/Random.h"
#include "classifier/boosting/Bandits/UCBK.h"
#include "classifier/boosting/Bandits/UCBKV.h"
#include "classifier/boosting/Bandits/UCBKRandomized.h"
#include "classifier/boosting/Bandits/Exp3.h"
#include "classifier/boosting/Bandits/Exp3G.h"
#include "classifier/boosting/Bandits/Exp3G2.h"
#include "classifier/boosting/Bandits/Exp3P.h"

namespace shogun {

	//REGISTER_LEARNER_NAME(SingleStump, BanditSingleStumpLearner)
	REGISTER_LEARNER(BanditSingleStumpLearner)

	//vector< int > BanditSingleStumpLearner::_T; // the number of a feature has been selected 
	//int BanditSingleStumpLearner::_numOfCalling = 0; //number of the single stump learner had been called
	//vector< float > BanditSingleStumpLearner::_X; // the everage reward of a feature
	//int BanditSingleStumpLearner::_K = 0; // number of columns to be selected

	void BanditSingleStumpLearner::declareArguments(nor_utils::Args& args)
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

		args.declareArgument("banditalgo", 
			"The bandit algorithm (UCBK, UCBKRandomized, EXP3 )\n"
			"Default is UCBK\n",
			1, "<algoname>");

		args.declareArgument("percent", 
			"how many percent of database will be used for estimating the payoffs(EXP3G)\n"
			"  Default is 10%\n",
			1, "<p>");

	}

	// ------------------------------------------------------------------------------

	void BanditSingleStumpLearner::initLearningOptions(const nor_utils::Args& args)
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
			//cerr << "Unknown update rule in ProductLearnerUCT (set to default [logedge]" << endl;
			_updateRule = LOGEDGE;
		}

		if ( args.hasArgument( "percent" ) ){
			_percentage = args.getValue<double>("percent", 0);
		} else {
			_percentage = 0.1;
		}

		if ( args.hasArgument( "rsample" ) ){
			_K = args.getValue<int>("rsample", 0);
		} else {
			_K = 1;
		}


		string banditAlgoName = "";
		if ( args.hasArgument( "banditalgo" ) )
			args.getValue("banditalgo", 0, banditAlgoName ); 

		if ( banditAlgoName.compare( "Random" ) == 0 )
			_banditAlgoName = BA_RANDOM;
		else if ( banditAlgoName.compare( "UCBK" ) == 0 )
			_banditAlgoName = BA_UCBK;
		else if ( banditAlgoName.compare( "UCBKR" ) == 0 )
			_banditAlgoName = BA_UCBKR;
		else if ( banditAlgoName.compare( "UCBKV" ) == 0 )
			_banditAlgoName = BA_UCBKV;
		else if ( banditAlgoName.compare( "EXP3" ) == 0 )
			_banditAlgoName = BA_EXP3;
		else if ( banditAlgoName.compare( "EXP3G" ) == 0 )
			_banditAlgoName = BA_EXP3G;
		else if ( banditAlgoName.compare( "EXP3G2" ) == 0 )
			_banditAlgoName = BA_EXP3G2;
		else if ( banditAlgoName.compare( "EXP3P" ) == 0 )
			_banditAlgoName = BA_EXP3P;
		else {
			cerr << "Unknown bandit algo (BanditSingleStumpLearner)" << endl;
			_banditAlgoName = BA_UCBK;
		}

		if ( _banditAlgo == NULL ) {
			switch ( _banditAlgoName )
			{
				case BA_RANDOM:
					_banditAlgo =  new Random();
					break;	
				case BA_UCBK:
					_banditAlgo =  new UCBK();
					break;
				case BA_UCBKV:
					_banditAlgo =  new UCBKV();
					break;
				case BA_UCBKR:
					_banditAlgo = new UCBKRandomized();
					break;
				case BA_EXP3:
					_banditAlgo = new Exp3();
					break;
				case BA_EXP3G:
					_banditAlgo = new Exp3G();
					break;
				case BA_EXP3G2:
					_banditAlgo = new Exp3G2();
					break;
				case BA_EXP3P:
					_banditAlgo = new Exp3P();
					break;
				default:
					cerr << "There is no bandit algorithm to be given!" << endl;
					exit( -1 );
			}
			// the bandit algorithm object must be initilaized once only
			_banditAlgo->initLearningOptions( args );
		}

		
	}

	//-------------------------------------------------------------------------------

	void BanditSingleStumpLearner::init() {
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();
		const int armNumber = _banditAlgo->getArmNumber();
		
		if ( numColumns < armNumber )
		{
			cerr << "The number of colums smaller than the number of the arms!!!!!!" << endl;
			exit( -1 );
		}

		BaseLearner* pWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner("SingleStumpLearner");
		
		_banditAlgo->setArmNumber( numColumns );
		
		vector<double> initialValues( numColumns );

		for( int i=0; i < numColumns; i++ )
		{
			SingleStumpLearner* singleStump = dynamic_cast<SingleStumpLearner*>( pWeakHypothesisSource->create());
			
			singleStump->setTrainingData(_pTrainingData);
			double energy = singleStump->run( i );
			double edge = singleStump->getEdge();
			double reward = getRewardFromEdge( (float) edge );
			
			initialValues[i] = reward;
			
			delete singleStump;
		}

		_banditAlgo->initialize( initialValues );

	}

	// ------------------------------------------------------------------------------
	double BanditSingleStumpLearner::getRewardFromEdge( float edge )
	{
		double updateWeight = 0.0;
		if ( _updateRule == EDGE_SQUARE ) {
			updateWeight = 1 - sqrt( 1 - ( edge * edge ) );
		} else if ( _updateRule == LOGEDGE ) { // logedge: published our in the ICML paper
			if ( edge < ( 1.0 - numeric_limits< double >::epsilon() ) ) {
				updateWeight = - log(sqrt( 1 - ( edge * edge ) ));
			} else {
				updateWeight = - log( numeric_limits< double >::epsilon() );
			}
			if ( updateWeight > 1.0 ) updateWeight = 1.0;
		} else if ( _updateRule == ESQUARE ) {
			updateWeight = edge * edge;
		}
		
		return updateWeight;
	}


	// ------------------------------------------------------------------------------

	float BanditSingleStumpLearner::run()
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

		StumpAlgorithm<float> sAlgo(numClasses);
		sAlgo.initSearchLoop(_pTrainingData);

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

			const pair<vpIterator,vpIterator> dataBeginEnd = 
				static_cast<SortedData*>(_pTrainingData)->getFileteredBeginEnd( _armsForPulling[i] );


			const vpIterator dataBegin = dataBeginEnd.first;
			const vpIterator dataEnd = dataBeginEnd.second;

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
				cout << "\tK = " <<i << endl;
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
	void BanditSingleStumpLearner::estimatePayoffs( vector<double>& payoffs )
	{
		set<int> oldIndexSet;
		set<int> randomIndexSet;
		const int numExamples = _pTrainingData->getNumExamples();
		const int numColumns = _pTrainingData->getNumAttributes();

		_pTrainingData->getIndexSet( oldIndexSet );
		int numSubset = static_cast<int>( static_cast<double>(numExamples) * _percentage );
		
		if ( numSubset < 2 ) {
			//use the whole dataset, do nothing
		} else {
			for (int j = 0; j < numExamples; ++j)
			{
				// Tricky way to select numOfDimensions columns randomly out of numColumns
				int rest = numExamples - j;
				float r = rand()/static_cast<float>(RAND_MAX);

				if ( static_cast<float>(numSubset) / rest > r ) 
				{
					--numSubset;
					randomIndexSet.insert( j );
				}
			}
			_pTrainingData->loadIndexSet( randomIndexSet );
		}
		
		
		payoffs.resize( numColumns );

		BaseLearner* pWeakHypothesisSource = 
			BaseLearner::RegisteredLearners().getLearner("SingleStumpLearner");		
		
		for( int i=0; i < numColumns; i++ )
		{
			if ( payoffs[i] > 0.0 ) continue;

			SingleStumpLearner* singleStump = dynamic_cast<SingleStumpLearner*>( pWeakHypothesisSource->create());
			
			singleStump->setTrainingData(_pTrainingData);
			double energy = singleStump->run( i );
			double edge = singleStump->getEdge();
			double reward = getRewardFromEdge( (float) edge );
			
			payoffs[i] = reward;			
			delete singleStump;
		}

		//restore the database
		_pTrainingData->loadIndexSet( oldIndexSet );
	}

	// ------------------------------------------------------------------------------

	float BanditSingleStumpLearner::phi(float val, int /*classIdx*/) const
	{
		if (val > _threshold)
			return +1;
		else
			return -1;
	}

	// ------------------------------------------------------------------------------

	float BanditSingleStumpLearner::phi(InputData* pData,int pointIdx) const
	{
		return phi(pData->getValue(pointIdx,_selectedColumn),0);
	}

	// -----------------------------------------------------------------------

	void BanditSingleStumpLearner::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class method
		FeaturewiseLearner::save(outputStream, numTabs);

		// save selectedCoulumn
		outputStream << Serialization::standardTag("threshold", _threshold, numTabs) << endl;
		
		outputStream << Serialization::vectorTag("rewards", _armsForPulling, _rewards, "arm", numTabs) << endl;
		//outputStream << Serialization::standardTag("columnIndex", _selectedColumn, numTabs) << endl;
	}

	// -----------------------------------------------------------------------

	void BanditSingleStumpLearner::load(nor_utils::StreamTokenizer& st)
	{
		//recalculate the initial values
		//these values can be stored also avoiding the recalculation of this values (maybe this would be better solution)
		if ( ! this->_banditAlgo->isInitialized() ) {
			const int numColumns = _pTrainingData->getNumAttributes();
			const int armNumber = _banditAlgo->getArmNumber();
			
			if ( numColumns < armNumber )
			{
				cerr << "The number of colums smaller than the number of the arms!!!!!!" << endl;
				exit( -1 );
			}
			
			_banditAlgo->setArmNumber( numColumns );
			
			vector<double> initialValues(0);
			_banditAlgo->initialize( initialValues );
		}

		// Calling the super-class method
		FeaturewiseLearner::load(st);

		_threshold = UnSerialization::seekAndParseEnclosedValue<float>(st, "threshold");
	

		stringstream thresholdString;
		thresholdString << _threshold;
		_id = _id + thresholdString.str();

		//restore the rewards		
		UnSerialization::seekAndParseVectorTag( st, "rewards", "arm", _armsForPulling, _rewards );

		_reward = -numeric_limits<float>::infinity();
		for( int i=0; i<(int)_rewards.size(); i++ )
		{
			_banditAlgo->receiveReward( _armsForPulling[i], _rewards[i] );
			if ( _reward < _rewards[i] ) _reward = (float)_rewards[i];
		}
	}

	// -----------------------------------------------------------------------

	void BanditSingleStumpLearner::subCopyState(BaseLearner *pBaseLearner)
	{
		FeaturewiseLearner::subCopyState(pBaseLearner);

		BanditSingleStumpLearner* pBanditSingleStumpLearner =
			dynamic_cast<BanditSingleStumpLearner*>(pBaseLearner);

		pBanditSingleStumpLearner->_threshold = _threshold;
		pBanditSingleStumpLearner->_reward = _reward;
		pBanditSingleStumpLearner->_banditAlgo = _banditAlgo;
		pBanditSingleStumpLearner->_banditAlgoName = _banditAlgoName;
		pBanditSingleStumpLearner->_K = _K;
		pBanditSingleStumpLearner->_updateRule = _updateRule;
		pBanditSingleStumpLearner->_percentage = _percentage;
	}

	// -----------------------------------------------------------------------

	//void BanditSingleStumpLearner::getStateData( vector<float>& data, const string& /*reason*/, InputData* pData )
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
	//      data[pos++] = BanditSingleStumpLearner::phi( pData->getValue( i, _selectedColumn), 0 );
	//}

	// -----------------------------------------------------------------------

} // end of namespace shogun


