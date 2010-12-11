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


#include "EnumLearner4.h"
#include <limits>

#include "classifier/boosting/IO/Serialization.h"

namespace shogun {

	//REGISTER_LEARNER_NAME(SingleStump, EnumLearner4)
	REGISTER_LEARNER(EnumLearner4)

		// ------------------------------------------------------------------------------

		float EnumLearner4::run()
	{
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();
		const int numExamples = _pTrainingData->getNumExamples();

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> vMu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions
		vector<float> previousTmpV(numClasses); // The class-wise votes/abstentions

		float tmpAlpha,previousTmpAlpha, previousEnergy;
		float bestEnergy = numeric_limits<float>::max();

		int numOfDimensions = _maxNumOfDimensions;
		for (int j = 0; j < numColumns; ++j)
		{
			// Tricky way to select numOfDimensions columns randomly out of numColumns
			int rest = numColumns - j;
			float r = rand()/static_cast<float>(RAND_MAX);

			if ( static_cast<float>(numOfDimensions) / rest > r ) 
			{
				--numOfDimensions;

				if (_verbose > 2)
					cout << "    --> trying attribute = "
					<<_pTrainingData->getAttributeNameMap().getNameFromIdx(j)
					<< endl << flush;

				const int numIdxs = _pTrainingData->getEnumMap(j).getNumNames();

				// Create and initialize the numIdxs x numClasses gamma matrix
				vector<vector<float> > tmpGammasPls(numIdxs);
				vector<vector<float> > tmpGammasMin(numIdxs);
				for (int io = 0; io < numIdxs; ++io) {
					vector<float> tmpGammaPls(numClasses);
					vector<float> tmpGammaMin(numClasses);
					fill(tmpGammaPls.begin(), tmpGammaPls.end(), 0.0);
					fill(tmpGammaMin.begin(), tmpGammaMin.end(), 0.0);
					tmpGammasPls[io] = tmpGammaPls;
					tmpGammasMin[io] = tmpGammaMin;
				}

				// Compute the elements of the gamma plus and minus matrices
				float entry;
				for (int i = 0; i < numExamples; ++i) {
					const vector<Label>& labels = _pTrainingData->getLabels(i);
					int io = static_cast<int>(_pTrainingData->getValue(i,j));	    
					for (int l = 0; l < numClasses; ++l) {
						entry = labels[l].weight * labels[l].y;
						if (entry > 0)
							tmpGammasPls[io][l] += entry;
						else if (entry < 0)
							tmpGammasMin[io][l] += -entry;
					}
				}

				// Initialize the u vector to random +-1
				vector<sRates> uMu(numIdxs); // The idx-wise rates
				vector<float> tmpU(numIdxs);// The idx-wise votes/abstentions
				vector<float> previousTmpU(numIdxs);// The idx-wise votes/abstentions

				/*
				for (int io = 0; io < numIdxs; ++io) {
					uMu[io].classIdx = io;	    
					if ( rand()/static_cast<float>(RAND_MAX) > 0.5 )
						tmpU[io] = +1;
					else
						tmpU[io] = -1;
				}
				*/

				vector<float> tmpV(numClasses); // The label-wise votes/abstentions
				int breakPoint = (int) ( rand()/static_cast<float>(RAND_MAX) * (numClasses - 1) );
				for (int io = 0; io < numClasses; ++io) {
					if ( io < breakPoint )
						tmpV[io] = -1;
					else
						tmpV[io] = +1;
				}


				vector<sRates> vMu(numClasses); // The label-wise rates
				for (int l = 0; l < numClasses; ++l)
					vMu[l].classIdx = l;
				
				float tmpEnergy = numeric_limits<float>::max();
				float tmpVal;
				tmpAlpha = 0.0;

				while (1) {
					previousEnergy = tmpEnergy;
					previousTmpU = tmpU;
					previousTmpAlpha = tmpAlpha;

					//filling out tmpU and uMu
					for (int io = 0; io < numIdxs; ++io) {
						uMu[io].rPls = uMu[io].rMin = uMu[io].rZero = 0; 
						for (int l = 0; l < numClasses; ++l) {
							if (tmpV[l] > 0) {
								uMu[io].rPls += tmpGammasPls[io][l];
								uMu[io].rMin += tmpGammasMin[io][l];
							}
							else if (tmpV[l] < 0) {
								uMu[io].rPls += tmpGammasMin[io][l];
								uMu[io].rMin += tmpGammasPls[io][l];
							}
						}
						if (uMu[io].rPls >= uMu[io].rMin) {
							tmpU[io] = +1;
						}
						else {
							tmpU[io] = -1;
							tmpVal = uMu[io].rPls;
							uMu[io].rPls = uMu[io].rMin;
							uMu[io].rMin = tmpVal;
						}
					}

					tmpEnergy = AbstainableLearner::getEnergy(uMu, tmpAlpha, tmpU);

					if (_verbose > 2)
						cout << "        --> energy U = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

					if (tmpEnergy >= previousEnergy) {
						tmpU = previousTmpU;
						break;
					}


					previousEnergy = tmpEnergy;
					previousTmpV = tmpV;
					previousTmpAlpha = tmpAlpha;

					//filling out tmpV and vMu
					for (int l = 0; l < numClasses; ++l) {
						vMu[l].rPls = vMu[l].rMin = vMu[l].rZero = 0; 
						for (int io = 0; io < numIdxs; ++io) {
							if (tmpU[io] > 0) {
								vMu[l].rPls += tmpGammasPls[io][l];
								vMu[l].rMin += tmpGammasMin[io][l];
							}
							else if (tmpU[io] < 0) {
								vMu[l].rPls += tmpGammasMin[io][l];
								vMu[l].rMin += tmpGammasPls[io][l];
							}
						}
						if (vMu[l].rPls >= vMu[l].rMin) {
							tmpV[l] = +1;
						}
						else {
							tmpV[l] = -1;
							tmpVal = vMu[l].rPls;
							vMu[l].rPls = vMu[l].rMin;
							vMu[l].rMin = tmpVal;
						}
					}

					tmpEnergy = AbstainableLearner::getEnergy(vMu, tmpAlpha, tmpV);

					if (_verbose > 2)
						cout << "        --> energy V = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

					if (tmpEnergy >= previousEnergy) {
						tmpV = previousTmpV;
						break;
					}


				}

				if ( previousEnergy < bestEnergy && previousTmpAlpha > 0 ) {
					_alpha = previousTmpAlpha;
					_v = tmpV;
					_u = tmpU;
					_selectedColumn = j;
					bestEnergy = previousEnergy;
				}
			}
		}


		_id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn);
		return bestEnergy;

	}

		// ------------------------------------------------------------------------------

		float EnumLearner4::run( int colIdx )
	{
		const int numClasses = _pTrainingData->getNumClasses();
		const int numColumns = _pTrainingData->getNumAttributes();
		const int numExamples = _pTrainingData->getNumExamples();

		// set the smoothing value to avoid numerical problem
		// when theta=0.
		setSmoothingVal( 1.0 / (float)_pTrainingData->getNumExamples() * 0.01 );

		vector<sRates> vMu(numClasses); // The class-wise rates. See BaseLearner::sRates for more info.
		vector<float> tmpV(numClasses); // The class-wise votes/abstentions
		vector<float> previousTmpV(numClasses); // The class-wise votes/abstentions

		float tmpAlpha,previousTmpAlpha, previousEnergy;
		float bestEnergy = numeric_limits<float>::max();

		int numOfDimensions = _maxNumOfDimensions;

		// Tricky way to select numOfDimensions columns randomly out of numColumns
		int j = colIdx;


		if (_verbose > 2)
			cout << "    --> trying attribute = "
			<<_pTrainingData->getAttributeNameMap().getNameFromIdx(j)
			<< endl << flush;

		const int numIdxs = _pTrainingData->getEnumMap(j).getNumNames();

		// Create and initialize the numIdxs x numClasses gamma matrix
		vector<vector<float> > tmpGammasPls(numIdxs);
		vector<vector<float> > tmpGammasMin(numIdxs);
		for (int io = 0; io < numIdxs; ++io) {
			vector<float> tmpGammaPls(numClasses);
			vector<float> tmpGammaMin(numClasses);
			fill(tmpGammaPls.begin(), tmpGammaPls.end(), 0.0);
			fill(tmpGammaMin.begin(), tmpGammaMin.end(), 0.0);
			tmpGammasPls[io] = tmpGammaPls;
			tmpGammasMin[io] = tmpGammaMin;
		}

		// Compute the elements of the gamma plus and minus matrices
		float entry;
		for (int i = 0; i < numExamples; ++i) {
			const vector<Label>& labels = _pTrainingData->getLabels(i);
			int io = static_cast<int>(_pTrainingData->getValue(i,j));	    
			for (int l = 0; l < numClasses; ++l) {
				entry = labels[l].weight * labels[l].y;
				if (entry > 0)
					tmpGammasPls[io][l] += entry;
				else if (entry < 0)
					tmpGammasMin[io][l] += -entry;
			}
		}

		// Initialize the u vector to random +-1
		vector<sRates> uMu(numIdxs); // The idx-wise rates
		vector<float> tmpU(numIdxs);// The idx-wise votes/abstentions
		vector<float> previousTmpU(numIdxs);// The idx-wise votes/abstentions
		for (int io = 0; io < numIdxs; ++io) {
			uMu[io].classIdx = io;	    
			if ( rand()/static_cast<float>(RAND_MAX) > 0.5 )
				tmpU[io] = +1;
			else
				tmpU[io] = -1;
		}

		//vector<sRates> vMu(numClasses); // The label-wise rates
		for (int l = 0; l < numClasses; ++l)
			vMu[l].classIdx = l;
		//vector<float> tmpV(numClasses); // The label-wise votes/abstentions

		float tmpEnergy = numeric_limits<float>::max();
		float tmpVal;
		tmpAlpha = 0.0;

		while (1) {
			previousEnergy = tmpEnergy;
			previousTmpV = tmpV;
			previousTmpAlpha = tmpAlpha;

			//filling out tmpV and vMu
			for (int l = 0; l < numClasses; ++l) {
				vMu[l].rPls = vMu[l].rMin = vMu[l].rZero = 0; 
				for (int io = 0; io < numIdxs; ++io) {
					if (tmpU[io] > 0) {
						vMu[l].rPls += tmpGammasPls[io][l];
						vMu[l].rMin += tmpGammasMin[io][l];
					}
					else if (tmpU[io] < 0) {
						vMu[l].rPls += tmpGammasMin[io][l];
						vMu[l].rMin += tmpGammasPls[io][l];
					}
				}
				if (vMu[l].rPls >= vMu[l].rMin) {
					tmpV[l] = +1;
				}
				else {
					tmpV[l] = -1;
					tmpVal = vMu[l].rPls;
					vMu[l].rPls = vMu[l].rMin;
					vMu[l].rMin = tmpVal;
				}
			}

			tmpEnergy = AbstainableLearner::getEnergy(vMu, tmpAlpha, tmpV);

			if (_verbose > 2)
				cout << "        --> energy V = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

			if (tmpEnergy >= previousEnergy) {
				tmpV = previousTmpV;
				break;
			}

			previousEnergy = tmpEnergy;
			previousTmpU = tmpU;
			previousTmpAlpha = tmpAlpha;

			//filling out tmpU and uMu
			for (int io = 0; io < numIdxs; ++io) {
				uMu[io].rPls = uMu[io].rMin = uMu[io].rZero = 0; 
				for (int l = 0; l < numClasses; ++l) {
					if (tmpV[l] > 0) {
						uMu[io].rPls += tmpGammasPls[io][l];
						uMu[io].rMin += tmpGammasMin[io][l];
					}
					else if (tmpV[l] < 0) {
						uMu[io].rPls += tmpGammasMin[io][l];
						uMu[io].rMin += tmpGammasPls[io][l];
					}
				}
				if (uMu[io].rPls >= uMu[io].rMin) {
					tmpU[io] = +1;
				}
				else {
					tmpU[io] = -1;
					tmpVal = uMu[io].rPls;
					uMu[io].rPls = uMu[io].rMin;
					uMu[io].rMin = tmpVal;
				}
			}

			tmpEnergy = AbstainableLearner::getEnergy(uMu, tmpAlpha, tmpU);

			if (_verbose > 2)
				cout << "        --> energy U = " << tmpEnergy << "\talpha = " << tmpAlpha << endl << flush;

			if (tmpEnergy >= previousEnergy) {
				tmpU = previousTmpU;
				break;
			}

			if ( previousEnergy < bestEnergy && previousTmpAlpha > 0 ) {
				_alpha = previousTmpAlpha;
				_v = tmpV;
				_u = tmpU;
				_selectedColumn = j;
				bestEnergy = previousEnergy;
			}
		}
		

		_id = _pTrainingData->getAttributeNameMap().getNameFromIdx(_selectedColumn);
		return bestEnergy;

	}


	// ------------------------------------------------------------------------------

	float EnumLearner4::phi(float val, int /*classIdx*/) const
	{
		return _u[static_cast<int>(val)];
	}

	// -----------------------------------------------------------------------

	void EnumLearner4::save(ofstream& outputStream, int numTabs)
	{
		// Calling the super-class method
		FeaturewiseLearner::save(outputStream, numTabs);

		// save the _u vector
		outputStream << Serialization::vectorTag("uArray", _u, 
			_pTrainingData->getEnumMap(_selectedColumn), 
			"idx", (float) 0.0, numTabs) << endl;
	}

	// -----------------------------------------------------------------------

	void EnumLearner4::load(nor_utils::StreamTokenizer& st)
	{
		// Calling the super-class method
		FeaturewiseLearner::load(st);

		// load phiArray data
		UnSerialization::seekAndParseVectorTag(st, "uArray", _pTrainingData->getEnumMap(_selectedColumn), 
			"idx", _u);
	}

	// -----------------------------------------------------------------------

	void EnumLearner4::subCopyState(BaseLearner *pBaseLearner)
	{
		FeaturewiseLearner::subCopyState(pBaseLearner);

		EnumLearner4* pEnumLearner4 =
			dynamic_cast<EnumLearner4*>(pBaseLearner);

		pEnumLearner4->_u = _u;
	}

} // end of namespace shogun
