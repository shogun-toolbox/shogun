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



#include "BanditLearner.h"
#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/BanditsLS/Exp3LS.h"
#include "classifier/boosting/BanditsLS/Exp3GLS.h"
#include "classifier/boosting/BanditsLS/UCTLS.h"

namespace shogun {

	// -----------------------------------------------------------------------
	void BanditLearner::declareArguments(nor_utils::Args& args) 
	{
		BaseLearner::declareArguments(args);

		args.declareArgument("rsample", 
			"Number of features to be considered\n"
			"  Default is one\n",
			1, "<K>");

		args.declareArgument("banditalgo", 
			"The bandit algorithm (EXP3_LS )\n"
			"Default is EXP_LS\n",
			1, "<algoname>");

	}

	// -----------------------------------------------------------------------
	void BanditLearner::getArms()
	{
		string nextAction = _banditAlgo->getNextAction( "" );	
		// default action is the emty action, therefore if it gives
		// the empty string we have to chose an arm uniformly from
		// the unexplored arms
		if ( nextAction.empty() )
		{
			const int colNum = _pTrainingData->getNumAttributes();			
			_armsForPulling.resize( _K );

			for( int i=0; i < _K; i++ )
			{
				double r = rand()/static_cast<double>(RAND_MAX);
				int randomFeature = static_cast<int>((r * ( colNum - 1 ) ));

				_armsForPulling[i] = randomFeature;
			}
		} else {
			stringstream ss( nextAction );
			ss.imbue( _loc );
			_armsForPulling.resize( _K );
			for( int i=0; i < _K; i++ ) 
			{
				int tmpInt; 
				ss >> tmpInt;
				_armsForPulling[i] = tmpInt;
			}
		}
	}

	// ------------------------------------------------------------------------------	
	// set arms
	void BanditLearner::provideRewardForBanditAlgo()
	{
		string key = getKeyToString();
		_banditAlgo->receiveReward( key, _reward );
	}

	// ------------------------------------------------------------------------------	
	string BanditLearner::getKeyToString()
	{
		string key = nor_utils::int2string( _armsForPulling[0] );
		for( int i=1; i < _K; i++ )
		{
			key.append( ":" );
			key.append( nor_utils::int2string( _armsForPulling[i] ) );
		}
		return key;
	}

	// ------------------------------------------------------------------------------
	void BanditLearner::init() {
		const int numColumns = _pTrainingData->getNumAttributes();

		_banditAlgo->setArmNumber(static_cast<int>( pow( (double)numColumns, (double)_K ) ) );

		if ( _banditAlgoName == BA_UCT_LS )
		{
			static_cast<UCT<double,string>*>(_banditAlgo)->setDepth( _K );
			static_cast<UCT<double,string>*>(_banditAlgo)->setOrder( numColumns );
		}

		map<string, double > initialValues;
		_banditAlgo->initialize( initialValues );

	}

	// ------------------------------------------------------------------------------
	double BanditLearner::getRewardFromEdge( float edge )
	{
		double updateWeight = 0.0;
		if ( _updateRule == EDGE_SQUARE ) {
			updateWeight = 1 - sqrt( 1 - ( edge * edge ) );
		} else if ( _updateRule == LOGEDGE ) {
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
	void BanditLearner::subCopyState( BaseLearner *pBaseLearner )
	{
		BaseLearner::subCopyState(pBaseLearner);
		
		_banditAlgo		= dynamic_cast<BanditLearner*>( pBaseLearner)->getBanditAlgoObject();
		_banditAlgoName = dynamic_cast<BanditLearner*>( pBaseLearner)->getAlgoType();
		_K				= dynamic_cast<BanditLearner*>( pBaseLearner)->getK();
		_reward			= dynamic_cast<BanditLearner*>( pBaseLearner)->getReward();
		_updateRule		= dynamic_cast<BanditLearner*>( pBaseLearner)->getUpdateRule();

		dynamic_cast<BanditLearner*>( pBaseLearner)->getArmsForPulling( _armsForPulling );
	}


// -----------------------------------------------------------------------
	void BanditLearner::load(nor_utils::StreamTokenizer& st)
	{
		BaseLearner::load(st);

		_reward = UnSerialization::seekAndParseEnclosedValue<double>(st, "reward");
		_K = UnSerialization::seekAndParseEnclosedValue<int>(st, "K");
		string arm = UnSerialization::seekAndParseEnclosedValue<string>(st, "arm");


		if ( ! _banditAlgo->isInitialized() ) {
			init();
		}

		
		this->_banditAlgo->receiveReward( arm, _reward );

		stringstream ss( arm );
		ss.imbue( _loc );
		string token;
		_armsForPulling.resize( _K );
		for( int i = 0; i < _K; i++ ) 
		{
			getline(ss, token, ':');
			int tmpInt = atoi( token.c_str() );
			_armsForPulling[i] = tmpInt;
		}
	}

// -----------------------------------------------------------------------
	void BanditLearner::save(ofstream& outputStream, int numTabs)
	{
		BaseLearner::save(outputStream, numTabs);

		outputStream << Serialization::standardTag("reward", getReward(), numTabs) << endl;
		outputStream << Serialization::standardTag("K", getK(), numTabs) << endl;
		outputStream << Serialization::standardTag("arm", getKeyToString(), numTabs) << endl;
	}

// -----------------------------------------------------------------------
	void BanditLearner::initLearningOptions(const nor_utils::Args& args)
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
			//cerr << "Unknown update rule in ProductLearnerUCT (set to default [edge]" << endl;
			_updateRule = LOGEDGE;
		}

		if ( args.hasArgument( "rsample" ) ){
			_K = args.getValue<int>("rsample", 0);
		}

		string banditAlgoName = "";
		if ( args.hasArgument( "banditalgo" ) )
			args.getValue("banditalgo", 0, banditAlgoName ); 

		if ( banditAlgoName.compare( "Random" ) == 0 )
			_banditAlgoName = BA_RANDOM_LS;
		else if ( banditAlgoName.compare( "UCBK" ) == 0 )
			_banditAlgoName = BA_UCBK_LS;
		else if ( banditAlgoName.compare( "UCBKR" ) == 0 )
			_banditAlgoName = BA_UCBKR_LS;
		else if ( banditAlgoName.compare( "UCBKV" ) == 0 )
			_banditAlgoName = BA_UCBKV_LS;
		else if ( banditAlgoName.compare( "EXP3" ) == 0 )
			_banditAlgoName = BA_EXP3_LS;
		else if ( banditAlgoName.compare( "EXP3G" ) == 0 )
			_banditAlgoName = BA_EXP3G_LS;
		else if ( banditAlgoName.compare( "UCT" ) == 0 )
			_banditAlgoName = BA_UCT_LS;
		else {
			cerr << "Unknown bandit algo (BanditSingleStumpLearner)" << endl;
			_banditAlgoName = BA_EXP3_LS;
		}

		if ( _banditAlgo == NULL ) {
			switch ( _banditAlgoName )
			{
				case BA_RANDOM_LS:
					//_banditAlgo =  new Random();
					break;	
				case BA_UCBK_LS:
					//_banditAlgo =  new UCBK();
					break;
				case BA_UCBKV_LS:
					//_banditAlgo =  new UCBKV();
					break;
				case BA_UCBKR_LS:
					//_banditAlgo = new UCBKRandomized();
					break;
				case BA_EXP3_LS:
					_banditAlgo = dynamic_cast<GenericBanditAlgorithmLS<double,string>*>( new Exp3LS<double,string>());
					break;
				case BA_EXP3G_LS:
					_banditAlgo = dynamic_cast<GenericBanditAlgorithmLS<double,string>*>(new Exp3GLS<double,string>());
					break;
				case BA_UCT_LS:					
					_banditAlgo = dynamic_cast<GenericBanditAlgorithmLS<double,string>*>(new UCT<double,string>());
					break;
				default:
					cerr << "There is no bandit algorithm to be given!" << endl;
					exit( -1 );
			}			
		}

	}


} // end of namespace shogun
