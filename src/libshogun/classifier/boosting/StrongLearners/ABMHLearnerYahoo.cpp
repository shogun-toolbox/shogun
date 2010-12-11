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


#include <ctime> // for time
#include <cmath> // for exp
#include <fstream> // for ofstream of the step-by-step data
#include <limits>
#include <iomanip> // setprecision

#include "classifier/boosting/Utils/Utils.h" // for addAndCheckExtension
#include "classifier/boosting/Defaults.h" // for defaultLearner
#include "classifier/boosting/IO/OutputInfo.h"
#include "classifier/boosting/IO/InputData.h"
#include "classifier/boosting/IO/Serialization.h" // to save the found strong hypothesis

#include "classifier/boosting/WeakLearners/BaseLearner.h"

#include "classifier/boosting/StrongLearners/ABMHLearnerYahoo.h"
#include "classifier/boosting/Classifiers/ABMHClassifierYahoo.h"

namespace MultiBoost {

	// -------------------------------------------------------------------------

	void ABMHLearnerYahoo::classify(const nor_utils::Args& args)
	{
		ABMHClassifierYahoo classifier(args, _verbose);

		// -test <dataFile> <shypFile>
		string testFileName = args.getValue<string>("test", 0);
		string shypFileName = args.getValue<string>("test", 1);
		int numIterations = args.getValue<int>("test", 2);

		string outResFileName;
		if ( args.getNumValues("test") > 3 )
			args.getValue("test", 3, outResFileName);

		classifier.run(testFileName, shypFileName, numIterations, outResFileName);
	}

	// -------------------------------------------------------------------------

} // end of namespace MultiBoost
