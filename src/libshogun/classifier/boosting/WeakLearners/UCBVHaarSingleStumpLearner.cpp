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


#include "UCBVHaarSingleStumpLearner.h"

#include "classifier/boosting/IO/HaarData.h"
#include "classifier/boosting/IO/Serialization.h"
#include "classifier/boosting/WeakLearners/Haar/HaarFeatures.h" // for shortname->type and viceversa (see serialization)
#include "classifier/boosting/Algorithms/StumpAlgorithm.h"

#include <limits> // for numeric_limits
#include <ctime> // for time
#include <math.h>

namespace MultiBoost {

	REGISTER_LEARNER_NAME(UCBVHaarSingleStump, UCBVHaarSingleStumpLearner)


	map<int,FeatureDataUCBV> UCBVHaarSingleStumpLearner::_featuresData;
	int UCBVHaarSingleStumpLearner::_numOfCalling = 0;

	//-------------------------------------------------------------------------------
	void UCBVHaarSingleStumpLearner::init() {
		UCBVHaarSingleStumpLearner::_featuresData.clear();
		UCBVHaarSingleStumpLearner::_numOfCalling = 1;

		//choose randomly the feature for the UCBV
		HaarData* pHaarData = static_cast<HaarData*>(_pTrainingData);
		vector<HaarFeature*>& loadedFeatures = pHaarData->getLoadedFeatures();

		int numConf = 0;
		bool quitConfiguration;
		long numProcessed;
		FeatureDataUCBV tmpFeatureData;
		int confIdx = 0;

		tmpFeatureData.T = 0;
		tmpFeatureData.X.clear();
		
		vector<HaarFeature*>::iterator ftIt;
		for (ftIt = loadedFeatures.begin(), confIdx = 0; ftIt != loadedFeatures.end(); ++ftIt, confIdx++)
		{
			// just for readability
			HaarFeature* pCurrFeature = *ftIt;
			if (_samplingType != ST_NO_SAMPLING)
				pCurrFeature->setAccessType(AT_RANDOM_SAMPLING);

			// Reset the iterator on the configurations. For random sampling
			// this clear the visited list
			pCurrFeature->resetConfigIterator();
			quitConfiguration = false;
			numProcessed = 0;

			numConf = 0;

			if (_verbose > 1)
				cout << "Learning type " << pCurrFeature->getName() << ".." << flush;

			while ( pCurrFeature->hasConfigs() ) 
			{
				// Move to the next configuration
				pCurrFeature->moveToNextConfig();

				int key =  ( 10 * pCurrFeature->getLoadedConfigIndex() ) + confIdx;
				//cout << key << "\t" <<  pCurrFeature->getType() << endl;
				//cout << (int) (key / 10) << "\t" << (key % 10) << endl;

				UCBVHaarSingleStumpLearner::_featuresData[key] = tmpFeatureData;

				// check stopping criterion for random configurations
				switch (_samplingType)
				{
				case ST_NUM:
					++numConf;
					if (numConf >= _samplingVal)
						quitConfiguration = true;
					break;

				case ST_NO_SAMPLING:
					perror("ERROR: What? No sampling??");
					break;

				} // end switch

				if (quitConfiguration)
					break;

			} // end while
		}

		cout << "The number of the randomly chosen features:\t" << UCBVHaarSingleStumpLearner::_featuresData.size() << endl;
	}
	// ------------------------------------------------------------------------------

	float UCBVHaarSingleStumpLearner::getTthSeriesElement( int t )
	{
		float retval = 0;
		retval = (float)(((2 * log(10.0 * log((float)t)) + log((float)t)))/2.0);
		return retval;
	}

	//-------------------------------------------------------------------------------
	void UCBVHaarSingleStumpLearner::declareArguments(nor_utils::Args& args)
	{
		// call the superclasses
		HaarLearner::declareArguments(args);
		SingleStumpLearner::declareArguments(args);
	}

	// ------------------------------------------------------------------------------

	void UCBVHaarSingleStumpLearner::initLearningOptions(const nor_utils::Args& args)
	{
		// call the superclasses
		HaarLearner::initOptions(args);
		SingleStumpLearner::initLearningOptions(args);
	}

	// ------------------------------------------------------------------------------

	float UCBVHaarSingleStumpLearner::classify(InputData* pData, int idx, int classIdx)
	{
		// The integral image data from the input must be transformed into the 
		// feature's space. This is done by getValue of the selected feature.
		return _v[classIdx] *
			UCBVHaarSingleStumpLearner::phi( 
			_pSelectedFeature->getValue( 
			pData->getValues(idx), _selectedConfig ),
			//static_cast<HaarData*>(pData)->getIntImage(idx), _selectedConfig ),
			classIdx );

	}

	//-------------------------------------------------------------------------------

	float UCBVHaarSingleStumpLearner::getBValue( int key )
	{
		float retval = numeric_limits<float>::max();
		map<int,FeatureDataUCBV>::const_iterator itUCBV = UCBVHaarSingleStumpLearner::_featuresData.find( key );
		
		if ( itUCBV != UCBVHaarSingleStumpLearner::_featuresData.end() ) 
		{
			int s = itUCBV->second.X.size();
			
			if ( s == 0 ) return retval;

			float epsilonT = getTthSeriesElement( UCBVHaarSingleStumpLearner::_numOfCalling );
			retval = (3.0 * epsilonT ) / s;
			//claculate the average reward
			float avgX = 0.0;
			for( int i = 0; i < s; i++ )
			{
				avgX += itUCBV->second.X[i];
			}
			avgX /= s;
			
			//claculate the variance of rewards
			float varX = 0.0;
			for( int i = 0; i < s; i++ )
			{
				varX += ( itUCBV->second.X[i] - avgX ) * ( itUCBV->second.X[i] - avgX );
			}
			varX /= s;

			retval += ( avgX + sqrt( ( 2.0 * varX * epsilonT ) / ( s ) ) ); 
		}

		return retval;
	}
	
	// ------------------------------------------------------------------------------

	float UCBVHaarSingleStumpLearner::run()
	{
 	   if ( UCBVHaarSingleStumpLearner::_numOfCalling == 0 ) {
 			init();
 	   }

		UCBVHaarSingleStumpLearner::_numOfCalling++;
		//cout << "Num of iter:\t" << UCBVHaarSingleStumpLearner::_numOfCalling << " " << this->getTthSeriesElement( UCBVHaarSingleStumpLearner::_numOfCalling ) << flush << endl;
		const int numClasses = _pTrainingData->getNumClasses();

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> mu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions

		float tmpThreshold;
		float tmpAlpha;

		float bestEnergy = numeric_limits<float>::max();
		float tmpEnergy;

		HaarData* pHaarData = static_cast<HaarData*>(_pTrainingData);

		// get the whole data matrix
		//const vector<int*>& intImages = pHaarData->getIntImageVector();

		// The data matrix transformed into the feature's space
		vector< pair<int, float> > processedHaarData(_pTrainingData->getNumExamples());

		// I need to prepare both type of sampling

		StumpAlgorithm<float> sAlgo(numClasses);
		sAlgo.initSearchLoop(_pTrainingData);

		float halfTheta;
		if ( _abstention == ABST_REAL || _abstention == ABST_CLASSWISE )
			halfTheta = _theta/2.0;
		else
			halfTheta = 0;

		// The declared features types
		vector<HaarFeature*>& loadedFeatures = pHaarData->getLoadedFeatures();

		// for every feature type
		vector<HaarFeature*>::iterator ftIt;
		//vector<HaarFeature*>::iterator maxftIt;

		vector<float> maxV( loadedFeatures.size() ); 
		vector<int> maxKey( loadedFeatures.size() );
		vector<int> maxNum( loadedFeatures.size() );
		
		//claculate the Bk,s,t of the randomly chosen features
		int key = getKeyOfMaximalElement();
		int featureIdx = (int) (key / 10);
		int featureType = (key % 10);

		//for (i = 0, ftIt = loadedFeatures.begin(); ftIt != loadedFeatures.end(); i++ ++ftIt)
		//*ftIt = loadedFeatures[ featureType ];

		// just for readability
		//HaarFeature* pCurrFeature = *ftIt;
		HaarFeature* pCurrFeature = loadedFeatures[ featureType ];
		if (_samplingType != ST_NO_SAMPLING)
			pCurrFeature->setAccessType(AT_RANDOM_SAMPLING);

		// Reset the iterator on the configurations. For random sampling
		// this clear the visited list
		pCurrFeature->loadConfigByNum( featureIdx );


		if (_verbose > 1)
			cout << "Learning type " << pCurrFeature->getName() << ".." << flush;

		// transform the data from intImages to the feature's space
		pCurrFeature->fillHaarData( _pTrainingData->getExamples(), processedHaarData );
		//pCurrFeature->fillHaarData(intImages, processedHaarData);

		// sort the examples in the new space by their coordinate
		sort( processedHaarData.begin(), processedHaarData.end(), 
			nor_utils::comparePair<2, int, float, less<float> >() );

		// find the optimal threshold
		tmpThreshold = sAlgo.findSingleThresholdWithInit(processedHaarData.begin(), 
			processedHaarData.end(), 
			_pTrainingData, halfTheta, &mu, &tmpV);

		tmpEnergy = getEnergy(mu, tmpAlpha, tmpV);
		

		// Store it in the current weak hypothesis.
		// note: I don't really like having so many temp variables
		// but the alternative would be a structure, which would need
		// to be inheritable to make things more consistent. But this would
		// make it less flexible. Therefore, I am still undecided. This
		// might change!
		_alpha = tmpAlpha;
		_v = tmpV;

		// I need to save the configuration because it changes within the object
		_selectedConfig = pCurrFeature->getCurrentConfig();
		// I save the object because it contains the informations about the type,
		// the name, etc..
		_pSelectedFeature = pCurrFeature;
		_threshold = tmpThreshold;

		bestEnergy = tmpEnergy;

		float edge = 0.0;
		for( vector<sRates>::iterator itR = mu.begin(); itR != mu.end(); itR++ ) edge += ( itR->rPls - itR->rMin ); 
		//need to set the X value
		updateKeys( key, edge * edge );
		
		if (!_pSelectedFeature)
		{
			cerr << "ERROR: No Haar Feature found. Something must be wrong!" << endl;
			exit(1);
		}
		else
		{
			if (_verbose > 1)
				cout << "Selected type: " << _pSelectedFeature->getName() << endl;
		}

		return bestEnergy;
	}
	
	int UCBVHaarSingleStumpLearner::getKeyOfMaximalElement()
	{
		int key;
		float maxVal = numeric_limits<float>::min();
		map<int,FeatureDataUCBV>::iterator itUCBV;
		
		for( itUCBV = UCBVHaarSingleStumpLearner::_featuresData.begin(); itUCBV != _featuresData.end(); itUCBV++ )
		{
			float val = getBValue( itUCBV->first );
			if ( val > maxVal ) 
			{
				maxVal = val;
				key = itUCBV->first;

				if ( maxVal == numeric_limits<float>::max() ) break;
			}
		}
		return key;
	}

	// ------------------------------------------------------------------------------
	void UCBVHaarSingleStumpLearner::updateKeys( int key, float val )
	{		
		map<int,FeatureDataUCBV>::const_iterator itUCBV = UCBVHaarSingleStumpLearner::_featuresData.find( key );
		FeatureDataUCBV tmp;

		if ( itUCBV == UCBVHaarSingleStumpLearner::_featuresData.end() ) 
		{
			tmp.T = 1;
			tmp.X.push_back( val );
			UCBVHaarSingleStumpLearner::_featuresData[ key ] = tmp;
		} else {
			tmp = itUCBV->second;
			tmp.T++;
			tmp.X.push_back( val );
			UCBVHaarSingleStumpLearner::_featuresData[ key ] = tmp;
		}
	}
	// ------------------------------------------------------------------------------

	InputData* UCBVHaarSingleStumpLearner::createInputData()
	{ 
		return new HaarData();
	}

	// ------------------------------------------------------------------------------

	float UCBVHaarSingleStumpLearner::phi(float val, int /*classIdx*/) const
	{
		if (val > _threshold)
			return +1;
		else
			return -1;
	}

	// ------------------------------------------------------------------------------

	void UCBVHaarSingleStumpLearner::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class methods
		SingleStumpLearner::save(outputStream, numTabs);
		HaarLearner::save(outputStream, numTabs);
	}

	// -----------------------------------------------------------------------

	void UCBVHaarSingleStumpLearner::load(nor_utils::StreamTokenizer& st)
	{
		// Calling the super-class methods
		SingleStumpLearner::load(st);
		HaarLearner::load(st);
	}

	// -----------------------------------------------------------------------

	void UCBVHaarSingleStumpLearner::subCopyState(BaseLearner *pBaseLearner)
	{
		SingleStumpLearner::subCopyState(pBaseLearner);
		HaarLearner::subCopyState(dynamic_cast<HaarLearner*>(pBaseLearner));
	}

	// -----------------------------------------------------------------------

	void UCBVHaarSingleStumpLearner::getStateData( vector<float>& data, const string& /*reason*/, InputData* pData )
	{
	   const int numClasses = pData->getNumClasses();
	   const int numExamples = pData->getNumExamples();

	   // reason ignored for the moment as it is used for a single task
	   data.resize( numClasses + numExamples );
	   int pos = 0;

	   for (int l = 0; l < numClasses; ++l)
		  data[pos++] = _v[l];

	   for (int i = 0; i < numExamples; ++i)
	   {
		  data[pos++] = UCBVHaarSingleStumpLearner::phi( 
							_pSelectedFeature->getValue( 
							   pData->getValues(i), _selectedConfig ),
						//       static_cast<HaarData*>(pData)->getIntImage(i), _selectedConfig ), 
						0 );
	   }
	}

	// -----------------------------------------------------------------------

} // end of MultiBoost namespace
