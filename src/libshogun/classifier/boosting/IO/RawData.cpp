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


// Indexes: i = loop on examples
//          j = loop on columns
//          l = loop on classes

#include <iostream> // for cerr
#include <algorithm> // for sort
#include <functional> // for less
#include <fstream>

#include "classifier/boosting/IO/TxtParser.h"
#include "classifier/boosting/IO/ArffParser.h"
#include "classifier/boosting/IO/ArffParserBzip2.h"
#include "classifier/boosting/IO/LSHTCParser.h"
#include "classifier/boosting/IO/SVMLightParser.h"

#include "classifier/boosting/Utils/Utils.h" // for white_tabs
#include "classifier/boosting/IO/RawData.h"

//#include <cassert>
//#include <cmath>  //for fabs

//extern "C" void solveEquationSystem( double* X, double* b, int* N );

//
namespace MultiBoost {



	// ------------------------------------------------------------------------

	RawData* RawData::load( const string& fileName, 
		eInputType inputType, int verboseLevel )
	{
		GenericParser* pParser = NULL;

		switch ( _fileFormat )
		{
		case FF_SIMPLE:
			pParser = new TxtParser( fileName );
			static_cast<TxtParser*>(pParser)->setClassEnd( _classInLastColumn );
			static_cast<TxtParser*>(pParser)->setHasExampleName( _hasExampleName );
			static_cast<TxtParser*>(pParser)->setSepChars( _sepChars );

			break;

		case FF_ARFF:
			pParser = new ArffParser( fileName );

			break;
		case FF_ARFFBZIP:
			pParser = new ArffParserBzip2( fileName );

			break;
		case FF_LSHTC:
			pParser = new LSHTCParser( fileName );
			static_cast<LSHTCParser*>(pParser)->setHierarchyFileName( _hierarchyFile );
			static_cast<LSHTCParser*>(pParser)->setLabeling( _labelingType, _labelingParameter );
			break;
		case FF_SVMLIGHT:
			pParser = new SVMLightParser( fileName );
			static_cast<SVMLightParser*>(pParser)->setHeaderFileName( _hierarchyFile );
		}

		if (verboseLevel > 0)
			cout << "Loading file " << fileName << ":" << endl;

		pParser->readData( _data, _classMap, _enumMaps, _attributeNameMap, _attributeTypes );

		_numClasses = _classMap.getNumNames();
		_numAttributes = pParser->getNumAttributes();
		_dataRep = pParser->getDataRep();
		_labelRep = pParser->getLabelRep();

		// debug notice
		/*
		if ( _dataRep == DR_SPARSE )
		{
			cout << "\nWARNING! Sparse *data* representation is NOT ready yet!" << endl;
			cout << "The method InputData::getValue must be specialized for this case!!" << endl;
			cout << "If not all the attributes are specified, the result will likely be a crash!" << endl;
		}
		*/

		_numExamples = static_cast<int>( _data.size() );

		// Initialize weights
		if ( !pParser->hasWeightInitialized() && _labelRep == LR_SPARSE )
		{
			cerr << "ERROR: Weights were not initialized with sparse labels!" << endl;
			exit (1);
		}
		delete pParser;

		vector<Example>::const_iterator it;
		map<int, int> tmpPointsPerClass;

		for ( it = _data.begin(); it != _data.end(); ++it )
		{
			const vector<Label>& labels = it->getLabels();
			vector<Label>::const_iterator lIt;

			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
			{
				switch ( _labelRep )
				{
				case LR_DENSE:
					if ( lIt->y > 0 )
						tmpPointsPerClass[lIt->idx]++;
					break;
				case LR_SPARSE:
					if ( lIt->y > 0 )
						tmpPointsPerClass[lIt->idx]++;
					break;
				}
			}

		}


		for (int l = 0; l < _numClasses; ++l)
			_nExamplesPerClass.push_back( tmpPointsPerClass[l] );

		initWeights();
#if MB_DEBUG
		// Checks if there is no variance on the columns
		checkVariances();
#endif

		if (verboseLevel > 0)
		{
			cout << "!!Loading is done!!" << endl;

			if (verboseLevel > 1)
			{
				cout << "Num Attributes = " << _numAttributes << endl;

				for (int l = 0; l < _numClasses; ++l)
					cout << "Of class '" << _classMap.getNameFromIdx(l) << "': "
					<< _nExamplesPerClass[l] << endl;

				cout << "Total: " << _numExamples << " examples read." << endl;
			}
		}
		return this;
	}


	// ------------------------------------------------------------------------

	void RawData::initOptions(const nor_utils::Args& args)
	{
		/////////////////////////////////////////////////////////////////////
		// to be moved as they belong to txtparser only!

		// check if the input file has a filename for each example
		if ( args.hasArgument("examplename") )
			_hasExampleName = true;

		// check if the class is at the last column of the data file
		if ( args.hasArgument("classend") )
			_classInLastColumn = true;

		_sepChars = "\t\r "; // "standard" white spaces
		if ( args.hasArgument("d") )
		{
			_sepChars = args.getValue<string>("d", 0);
			_sepChars = nor_utils::getEscapeSequence( _sepChars);
		}
		/////////////////////////////////////////////////////////////////////


		if ( args.hasArgument("fileformat") )
		{
			string fileFormat = args.getValue<string>("fileformat");
			if ( fileFormat == "simple" )
				_fileFormat = FF_SIMPLE;
			else if ( fileFormat == "arff" )
				_fileFormat = FF_ARFF;
			else if ( fileFormat == "arffbzip2" )
				_fileFormat = FF_ARFFBZIP;
			else if ( fileFormat == "lshtc" )
				_fileFormat = FF_LSHTC;
			else if ( fileFormat == "svmlight" )
				_fileFormat = FF_SVMLIGHT;
			else
			{
				cerr << "ERROR: Unrecognized --fileformat option!!" << endl;
				exit(1);
			}
		}

		/////////////////////////////////////////////////////////////////////
		// for the libsvm file format, the header can be stored in a separate file
		if ( args.hasArgument("headerfile") )
		{
			_hierarchyFile = args.getValue<string>("headerfile");
		} else {
			_hierarchyFile = "";
		}


		_weightInitType = WIT_SHARE_POINT; // default
		if ( args.hasArgument("weightpolicy") )
		{
			string weightInitString = args.getValue<string>("weightpolicy");

			if ( weightInitString == "sharepoints" )
				_weightInitType = WIT_SHARE_POINT;
			else if ( weightInitString == "sharelabels" )
				_weightInitType = WIT_SHARE_LABEL;
			else if ( weightInitString == "proportional" )
				_weightInitType = WIT_PROP_ONLY;
			else if ( weightInitString == "balanced" )
				_weightInitType = WIT_BALANCED;
			else
			{
				cerr << "ERROR: Invalid value (" << weightInitString << ") for option --weightpolicy!" << endl;
				exit(1);
			}
		}

		// LSHTC: set the labelling
		if ( args.hasArgument("labelsetting") ) {
			_labelingType = args.getValue<string>( "labelsetting", 0 );
			_labelingParameter = args.getValue<int>( "labelsetting", 1 );
		} else {
			_labelingType = "full";
			//-1 causes warning message
			_labelingParameter = -1;
		}
	}

	// ------------------------------------------------------------------------

	void RawData::initWeights()
	{
		vector<Example>::iterator eIt;
		vector<Label>::iterator lIt;
		float sumWeight; // sum of the weight for each example

		//switch ( _labelRep )
		//{
		//case LR_BINARY:
		//case LR_MULTI_DENSE:
		//   {
		//      int numLabels = ClassMappings::getNumClasses();
		//      float    posClassWeightDiv = 2.0*_numExamples;
		//      float nonPosClassWeightDiv = 2.0*_numExamples*(numLabels-1);

		//      int numPos;
		//      int numNeg;

		//      // for each example
		//      for ( eIt = _data.begin(); eIt != _data.end(); ++eIt )
		//      {
		//         vector<Label>& labels = eIt->getLabels();
		//         vector<Label>::iterator lIt;

		//         // for dense labels the size MUST always be the same!
		//         assert (labels.size() == numLabels );

		//         numPos = 0; numNeg = 0;
		//         // first count the number of positives y
		//         for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
		//            if ( lIt->y > 0)
		//               numPos += lIt->y;

		//         for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
		//         {
		//            if (lIt->y > 0) // +1
		//               lIt->weight = numLabels / posClassWeightDiv;
		//            else // -1
		//               lIt->weight = numLabels / nonPosClassWeightDiv;
		//         }

		//      }
		//   }

		//   break;

		switch ( _weightInitType )
		{
		case WIT_SHARE_POINT:
			{
				double oneDiv2n;
				// for each example
				for ( eIt = _data.begin(); eIt != _data.end(); ++eIt )
				{
					vector<Label>& labels = eIt->getLabels();

					double sumPos = 0;
					double sumNeg = 0;

					// first find the sum of the weights
					for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
					{
						if ( lIt->y > 0 )
							sumPos += lIt->weight;
						else if ( lIt->y < 0 )
							sumNeg += lIt->weight;
					}

					if ( nor_utils::is_zero(sumPos) || nor_utils::is_zero(sumNeg) )
						oneDiv2n = 1.0/(_numExamples);
					else
						oneDiv2n = 1.0/(2.0*_numExamples);

					// for each label
					for ( lIt = labels.begin(); lIt != labels.end(); ++lIt )
					{
						// clearly this can be optimized, but the current format is
						// just for clarity
						if ( lIt->y > 0 )
							lIt->weight = oneDiv2n * ( lIt->weight / sumPos );
						else if ( lIt->y < 0 )
							lIt->weight = oneDiv2n * ( lIt->weight / sumNeg );
						else
							lIt->weight = 0; // should never happen!
					}
				}
			}

			break;

		case WIT_SHARE_LABEL:

			// for each example
			for ( eIt = _data.begin(); eIt != _data.end(); ++eIt )
			{
				vector<Label>& labels = eIt->getLabels();
				sumWeight = 0;

				// first find the sum of the weights
				for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
					sumWeight += lIt->weight;

				// now set the weights
				for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
					lIt->weight /= (sumWeight * _numExamples);
			}

			break;

		case WIT_PROP_ONLY:

			sumWeight = 0;

			// first compute the whole sum of weights-examples
			for ( eIt = _data.begin(); eIt != _data.end(); ++eIt )
			{
				vector<Label>& labels = eIt->getLabels();
				for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
					sumWeight += lIt->weight;
			}

			// now re-weight 
			for ( eIt = _data.begin(); eIt != _data.end(); ++eIt )
			{
				vector<Label>& labels = eIt->getLabels();

				// now set the weights
				for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
					lIt->weight /= sumWeight;
			}

			break;

		case WIT_BALANCED:
			{
				/*
				double* equationCoeffM = new double[ numOfClasses * numOfClasses ];
				double* wiCommas = new double[ numOfClasses ];
				int* dim = new int;
				*dim = numOfClasses;
				

				for( int i = 0; i < numOfClasses; i ++ ) {
					for( int j = 0; j < numOfClasses; j ++ ) {
						if ( i != j ) {
							equationCoeffM[ (i* numOfClasses ) + j ] = numPerClasses[ j ];
						} else {
							equationCoeffM[ (i* numOfClasses ) + j ] = numPerClasses[ i ] * ( numOfClasses - 1.0 );
						}
					}
					wiCommas[i] = 1.0 / numOfClasses;
				}


				solveEquationSystem( equationCoeffM, wiCommas, dim );
				
				vector< double > wi( numOfClasses );
				vector< double > wic( numOfClasses );

				
				for( int i = 0; i < numOfClasses; i ++ ) {
					cout << wiCommas[i] << "\t";
					wic[i] = wiCommas[i];
					wi[i] = wiCommas[i] * ( numOfClasses - 1.0 );
				}
				cout << endl;

				delete [] wiCommas;
				delete [] equationCoeffM;
				delete dim;
				*/
				
				int numOfClasses = this->getNumClasses();
				vector< int > numPerClasses = this->getExamplesPerClass();
				int numOfSamples = this->getNumExample();
				vector< double > wi( numOfClasses );
				vector< double > wic( numOfClasses );

				// we assume pl = 1/K
				for( int i = 0; i < numOfClasses; i ++ ) {
					wi[i] =  (1.0 / numOfClasses) / (2.0 * numPerClasses[i]);
					wic[i] = (1.0 / numOfClasses) / (2.0 * ( numOfSamples - numPerClasses[i] ));
					
					//cout << wi[i] << "\t" << wic[i] << endl;
				}
				//cout << endl;


				//this->_nExamplesPerClass
				// for each example
				for ( eIt = _data.begin(); eIt != _data.end(); ++eIt )
				{
					vector<Label>& labels = eIt->getLabels();

					float sumPos = 0;
					float sumNeg = 0;

					// first find the sum of the weights					
					int i = 0;
					for ( lIt = labels.begin(); lIt != labels.end(); ++lIt, i++ )
					{
						if ( lIt->y > 0 )
							lIt->weight = wi[lIt->idx];
						else if ( lIt->y < 0 )
							lIt->weight = wic[lIt->idx];
					}

				}
			}

			break;

		}

		// check for the sum of weights!
		sumWeight = 0;
		for ( eIt = _data.begin(); eIt != _data.end(); ++eIt )
		{
			vector<Label>& labels = eIt->getLabels();
			vector<Label>::iterator lIt;

			// first find the sum of the weights
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
				sumWeight += lIt->weight;
		}

		if ( !nor_utils::is_zero(sumWeight-1.0, 1E-3 ) )
		{
			cerr << "\nERROR: Sum of weights (" << sumWeight << ") != 1!" << endl;
			cerr << "Try a different weight policy (--weightpolicy under 'Basic Algorithm Options')!" << endl;
			//exit(1);
		}

		// set the initial weights needed to calculate the initial weighted error (11)
		for ( eIt = _data.begin(); eIt != _data.end(); ++eIt )
		{
			vector<Label>& labels = eIt->getLabels();
#ifndef NOTIWEIGHT
			for (lIt = labels.begin(); lIt != labels.end(); ++lIt )
				lIt->initialWeight = lIt->weight;
#endif
		}

	}

	// ------------------------------------------------------------------------

#if MB_DEBUG
	// Print a warning if there is no variance in a column.
	// IMPORTANT: TO BE REWRITTEN!
	void InputData::checkVariances()
	{
		//// for each column
		//for (int j = 0; j < _numColumns; ++j)
		//{
		//   float valueChk = getValue(0, j); // get value of the first example..
		//   bool hasVariance = false;

		//   for (int i = 1; i < _numExamples; ++i)
		//   {
		//      if ( getValue(i, j) != valueChk)
		//      {
		//         hasVariance = true; // the value has changed.. thus variance > 0
		//         break;
		//      }
		//   }

		//   if (!hasVariance)
		//      cerr << "WARNING!! Column " << j << " has no variance!" << endl;
	}
}
#endif // MB_DEBUG


} //end namespace MultiBoost


