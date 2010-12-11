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


/*! \mainpage MultiBoost Doxygen Documentation
*
* \section intro_sec Introduction
*
* <a href="http://www.iro.umontreal.ca/~casagran/multiboost.html">MultiBoost</a> is a multi-class AdaBoost learner, very flexible and 
* well commented. Feel free to use it and to modify it!
*
* Copyright (C) 2005-2006 Norman Casagrande, Balazs Kegl and James Bergstra.
* This software is covered by the 
* <a href="http://www.gnu.org/copyleft/lesser.html">LGPL</a> licence.
*
* \section install_sec Installation
*
* Just run this simple two commands to compile:
*
* \code
* % tmake multiboost.pro -o Makefile
* % make
* \endcode
* 
* Note: you need <a href="http://www.trolltech.com/download/freebies.html">tmake</a> to create
* the Makefile. If you don't have it, and you are using g++ on a linux machine machine, you can use
* <a href="http://cvs.sourceforge.net/viewcvs.py/multiboost/MultiBoost/linux_g%2B%2B_makefile/">this Makefile</a>.
*
* To run the program, just type:
* \code
* % multiboost
* \endcode
*
* To get some help, type:
* \code 
* % multiboost -help
* \endcode
*
* \section Basic Coding Informations
*
* MultiBoost was built with Boosting algorithms in mind, in particular
* for multi-class AdaBoost.MH.
*
* The concept has been centered on the base learner (aka weak learner), 
* which holds the methods to find the best function h(x) at each boosting
* iteration. Each base function can also decide the type of data to process,
* and even the strong learner that is calling it.
*
* Here is how it works:\n
* \li Each base learner (i.e. 
* \link MultiBoost::SingleStumpLearner SingleStumpLearner\endlink) 
* is registered in a static list, that is a class factory, using just a simple macro 
* (see #REGISTER_LEARNER for details).
* \li The users selects the base learner either by providing an argument on the 
* command line or by loading a strong hypothesis file.
* \li If the base learner is in the static list, it will inform the framework
* what are its InputData and 
* \link MultiBoost::GenericStrongLearner GenericStrongLearner\endlink. 
* If the default data format 
* (\link MultiBoost::InputData InputData\endlink) and strong learner 
* (\link MultiBoost::AdaBoostMHLearner AdaBoostMHLearner\endlink) are not enough 
* they can be overrided with two methods of 
* \link MultiBoost::BaseLearner BaseLearner\endlink: 
* \link MultiBoost::BaseLearner::createInputData() createInputData()\endlink and 
* \link MultiBoost::BaseLearner::createGenericStrongLearner() createGenericStrongLearner()\endlink.
* \li The strong learner run the boosting process by creating an object of
* type BaseLearner (any registered weak learner must implement the method
* subCreate() which returns an allocated copy of itself), that will have to
* find the best h(x) for the current boosting iteration. Once the function
* has been received it is stored in the vector of the weak learner. Note
* that the weighting factor alpha (if needed) needs to be stored in the 
* weak learner.
* \li The serialization is performed simply by overriding the proper classes
* of \link MultiBoost::BaseLearner BaseLearner\endlink.
* \li The classification depends on the strong learner chosen. 
* \link MultiBoost::ABMHClassifierYahoo ABMHClassifierYahoo\endlink 
* can classify any type of weak learner that use AdaBoost.MH
* as strong learner, as long as they implement all the abstract methods.
*
* For any other question, please refer to the documentation or contact one of
* the authors.
*
* \section References
* 
* Here's the \b bibtex reference:
\verbatim
@misc{multiboost,
      author   = {Norman Casagrande and Bal\'{a}zs K\'{e}gl},
      title    = {MultiBoost: An open source multi-class AdaBoost learner},
      note     = {http://www.iro.umontreal.ca/~casagran/multiboost.html},
      year     = {2005-2006}
}\endverbatim
*
*/

/**
* \file main.cpp
* The file that contains the main() function.
* \date 10/11/2005
*/
#pragma warning( disable : 4786 )

#include <vector>
#include <string>
#include <map>
#include <iostream>

#include <shogun/classifier/boosting/Defaults.h>
#include <shogun/classifier/boosting/Utils/Args.h>

#include <shogun/classifier/boosting/StrongLearners/GenericStrongLearner.h>
#include <shogun/classifier/boosting/WeakLearners/BaseLearner.h> // To get the list of the registered weak learners

#include <shogun/classifier/boosting/Classifiers/ABMHClassifierYahoo.h> // just for -ssfeatures

#include <shogun/classifier/boosting/IO/Serialization.h> // for unserialization

#include <shogun/classifier/boosting/IO/EncodeData.h> // for --encode
#include <shogun/classifier/boosting/IO/InputData.h> // for --encode
#include <shogun/classifier/boosting/WeakLearners/ParasiteLearner.h> // for --encode
#include <shogun/classifier/boosting/StrongLearners/AdaBoostMHLearner.h> // for --encode
#include <shogun/classifier/boosting/IO/OutputInfo.h> // for --encode
#include <shogun/classifier/boosting/Bandits/GenericBanditAlgorithm.h>

using namespace std;
using namespace shogun;

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//---------------------------------------------------------------------------

/**
* Check if a given base learner has been registered. If not it will give an error
* and exit.
* \param baseLearnerName The name of the base learner to be checked.
* \date 21/3/2006
*/
void checkBaseLearner(const string& baseLearnerName)
{
   if ( !BaseLearner::RegisteredLearners().hasLearner(baseLearnerName) )
   {
      // Not found in the registered!
      cerr << "ERROR: learner <" << baseLearnerName << "> not found in the registered learners!" << endl;
      exit(1);
   }
}

//---------------------------------------------------------------------------

/**
* Show the basic output. Called when no argument is provided.
* \date 11/11/2005
*/
void showBase()
{
   cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
   cout << "---------------------------------------------------------------------------" << endl;
   cout << "Build: " << __DATE__ << " (" << __TIME__ << ") (C) Norman Casagrande 2005-2006" << endl << endl;
   cout << "===> Type --help for help or --static to show the static options" << endl;

   exit(0);
}

//---------------------------------------------------------------------------

/**
* Show the help. Called when -h argument is provided.
* \date 11/11/2005
*/
void showHelp(nor_utils::Args& args, const vector<string>& learnersList)
{
   cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
   cout << "------------------------ HELP SECTION --------------------------" << endl;

	args.printGroup("Parameters");

   cout << endl;
   cout << "For specific help options type:" << endl;
   cout << "   --h general: General options" << endl;
   cout << "   --h io: I/O options" << endl;
   cout << "   --h algo: Basic algorithm options" << endl;
   cout << "   --h bandits: Bandit algorithm options" << endl;

   cout << endl;
   cout << "For weak learners specific options type:" << endl;
   
   vector<string>::const_iterator it;
   for (it = learnersList.begin(); it != learnersList.end(); ++it)
      cout << "   --h " << *it << endl;

   exit(0);
}

//---------------------------------------------------------------------------

/**
* Show the help for the options.
* \param args The arguments structure.
* \date 28/11/2005
*/
void showOptionalHelp(nor_utils::Args& args)
{
   string helpType = args.getValue<string>("h", 0);

   cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
   cout << "---------------------------------------------------------------------------" << endl;

   if (helpType == "general")
      args.printGroup("General Options");
   else if (helpType == "io")
      args.printGroup("I/O Options");
   else if (helpType == "algo")
      args.printGroup("Basic Algorithm Options");
   else if (helpType == "bandits")
      args.printGroup("Bandit Algorithm Options");
   else if ( BaseLearner::RegisteredLearners().hasLearner(helpType) )
      args.printGroup(helpType + " Options");
   else
      cerr << "ERROR: Unknown help section <" << helpType << ">" << endl;
}

//---------------------------------------------------------------------------

/**
* Show the default values.
* \date 11/11/2005
*/
void showStaticConfig()
{
   cout << "MultiBoost (v" << CURRENT_VERSION << "). An obvious name for a multi-class AdaBoost learner." << endl;
   cout << "------------------------ STATIC CONFIG -------------------------" << endl;

   cout << "- Sort type = ";
#if CONSERVATIVE_SORT
   cout << "CONSERVATIVE (slow)" << endl;
#else
   cout << "NON CONSERVATIVE (fast)" << endl;
#endif

   cout << "Comment: " << COMMENT << endl;
#ifndef NDEBUG
   cout << "Important: NDEBUG not active!!" << endl;
#endif

#if MB_DEBUG
   cout << "MultiBoost debug active (MB_DEBUG=1)!!" << endl;
#endif

   exit(0);  
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

/**
* The main function. Everything starts here!
* \param argc The number of arguments.
* \param argv The arguments.
* \date 11/11/2005
*/
int main(int argc, const char* argv[])
{
   init_shogun();

   // initializing the random number generator
   srand ( time(NULL) );

   // no need to synchronize with C style stream
   std::ios_base::sync_with_stdio(false);

#if STABLE_SORT
   cerr << "WARNING: Stable sort active! It might be slower!!" << endl;
#endif

   //////////////////////////////////////////////////////////////////////////
   // Standard arguments
   nor_utils::Args args;

   args.setArgumentDiscriminator("--");

   args.declareArgument("help");
   args.declareArgument("static");

   args.declareArgument("h", "Help", 1, "<optiongroup>");

   //////////////////////////////////////////////////////////////////////////
   // Basic Arguments

   args.setGroup("Parameters");

   args.declareArgument("train", "Performs training.", 2, "<dataFile> <nInterations>");
   args.declareArgument("traintest", "Performs training and test at the same time.", 3, "<trainingDataFile> <testDataFile> <nInterations>");
   args.declareArgument("test", "Test the model.", 3, "<dataFile> <numIters> <shypFile>");
   args.declareArgument("test", "Test the model and output the results", 4, "<datafile> <shypFile> <numIters> <outFile>");
   args.declareArgument("cmatrix", "Print the confusion matrix for the given model.", 2, "<dataFile> <shypFile>");
   args.declareArgument("cmatrixfile", "Print the confusion matrix with the class names to a file.", 3, "<dataFile> <shypFile> <outFile>");
   args.declareArgument("posteriors", "Output the posteriors for each class, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
   args.declareArgument("cposteriors", "Output the calibrated posteriors for each class, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");
   
   args.declareArgument("likelihood", "Output the likelihoof of data for each iteration, that is the vector-valued discriminant function for the given dataset and model.", 4, "<dataFile> <shypFile> <outFile> <numIters>");

   args.declareArgument("encode", "Save the coefficient vector of boosting individually on each point using ParasiteLearner", 6, "<inputDataFile> <autoassociativeDataFile> <outputDataFile> <nIterations> <poolFile> <nBaseLearners>");
   args.declareArgument("roc", "Print out the ROC curve (it calculate the ROC curve for the first class)", 4, "<dataFile> <shypFile> <outFile> <numIters>" );

   args.declareArgument("ssfeatures", "Print matrix data for SingleStump-Based weak learners (if numIters=0 it means all of them).", 4, "<dataFile> <shypFile> <outFile> <numIters>");

   args.declareArgument( "fileformat", "Defines the type of intput file. Available types are:\n" 
                                       "* simple: each line has attributes separated by whitespace and class at the end (DEFAULT!)\n"
                                       "* arff: arff filetype. The header file can be specified using --arffheader option\n"
									   "* arffbzip: bziped arff filetype. The header file can be specified using --arffheader option\n"
									   "* svmlight: \n"
                                       "(Example: --fileformat simple)",
                         1, "<fileFormat>" );

   args.declareArgument("headerfile", "The filename of the header file (SVMLight).", 1, "header.txt");
   //yahoo challenge
   args.declareArgument("queryfile", "The filename of the query file (Yahoo).", 1, "queries");
   args.declareArgument("labelsetting", "The labeling of the dataset (LSHTC)."
						"full, subcat, depth, ndepth, leaf, children",
						2, "<type> <par>");
   args.declareArgument("scoring", "Calculation of scoring (Yahoo).", 1, "expweight");
   //yahoo challenge

   args.declareArgument("constant", "Check constant learner in each iteration.", 0, "");
   args.declareArgument("timelimit", "Time limit in minutes", 1, "<minutes>" );
   args.declareArgument("stronglearner", "Strong learner. Available strong learners:\n"
										 "AdaBoost (default)\n"
										 "BrownBoost\n", 1, "<stronglearner>" );

   args.declareArgument("slowresumeprocess", "Compute the results in each iteration (slow resume)\n"
										 "Compute only the data of the last iteration (fast resume, default)\n", 0, "" );
   args.declareArgument("weights", "Outputs the weights of instances at the end of the learning process", 1, "<filename>" );
   //// ignored for the moment!
   //args.declareArgument("arffheader", "Specify the arff header.", 1, "<arffHeaderFile>");

   //////////////////////////////////////////////////////////////////////////
   // Options

   args.setGroup("I/O Options");

   /////////////////////////////////////////////
   // these are valid only for .txt input!
   // they might be removed!
   args.declareArgument("d", "The separation characters between the fields (default: whitespaces).\nExample: -d \"\\t,.-\"\nNote: new-line is always included!", 1, "<separators>");
   args.declareArgument("classend", "The class is the last column instead of the first (or second if -examplelabel is active).");
   args.declareArgument("examplename", "The data file has an additional column (the very first) which contains the 'name' of the example.");
   /////////////////////////////////////////////

   args.setGroup("Basic Algorithm Options");
   args.declareArgument("weightpolicy", "Specify the type of weight initialization. The user specified weights (if available) are used inside the policy which can be:\n"
                                        "* sharepoints Share the weight equally among data points and between positiv and negative labels (DEFAULT)\n"
                                        "* sharelabels Share the weight equally among data points\n"
                                        "* proportional Share the weights freely", 1, "<weightType>");


   args.setGroup("General Options");

   args.declareArgument("verbose", "Set the verbose level 0, 1 or 2 (0=no messages, 1=default, 2=all messages).", 1, "<val>");
   args.declareArgument("outputinfo", "Output informations on the algorithm performances during training, on file <filename>.", 1, "<filename>");
   args.declareArgument("seed", "Defines the seed for the random operations.", 1, "<seedval>");

   //////////////////////////////////////////////////////////////////////////
   // Shows the list of available learners
   string learnersComment = "Available learners are:";

   vector<string> learnersList;
   BaseLearner::RegisteredLearners().getList(learnersList);
   vector<string>::const_iterator it;
   for (it = learnersList.begin(); it != learnersList.end(); ++it)
   {
      learnersComment += "\n ** " + *it;
      // defaultLearner is defined in Defaults.h
      if ( *it == defaultLearner )
         learnersComment += " (DEFAULT)";
   }

   args.declareArgument("learnertype", "Change the type of weak learner. " + learnersComment, 1, "<learner>");

   //////////////////////////////////////////////////////////////////////////
   //// Declare arguments that belongs to all weak learners
   BaseLearner::declareBaseArguments(args);

   ////////////////////////////////////////////////////////////////////////////
   //// Weak learners (and input data) arguments
   for (it = learnersList.begin(); it != learnersList.end(); ++it)
   {
      args.setGroup(*it + " Options");
      // add weaklearner-specific options
      BaseLearner::RegisteredLearners().getLearner(*it)->declareArguments(args);
   }

   //////////////////////////////////////////////////////////////////////////
   //// Declare arguments that belongs to all bandit learner
   GenericBanditAlgorithm::declareBaseArguments(args);


   //////////////////////////////////////////////////////////////////////////////////////////  
   //////////////////////////////////////////////////////////////////////////////////////////

   switch ( args.readArguments(argc, argv) )
   {
   case nor_utils::AOT_NO_ARGUMENTS:
      showBase();
      break;

   case nor_utils::AOT_UNKOWN_ARGUMENT:
      exit(1);
      break;

   case nor_utils::AOT_INCORRECT_VALUES_NUMBER:
      exit(1);
      break;

   case nor_utils::AOT_OK:
      break;
   }

   //////////////////////////////////////////////////////////////////////////////////////////  
   //////////////////////////////////////////////////////////////////////////////////////////

   if ( args.hasArgument("help") )
      showHelp(args, learnersList);
   if ( args.hasArgument("static") )
      showStaticConfig();

   //////////////////////////////////////////////////////////////////////////////////////////  
   //////////////////////////////////////////////////////////////////////////////////////////

   if ( args.hasArgument("h") )
      showOptionalHelp(args);

   //////////////////////////////////////////////////////////////////////////////////////////  
   //////////////////////////////////////////////////////////////////////////////////////////

   int verbose = 1;

   if ( args.hasArgument("verbose") )
      args.getValue("verbose", 0, verbose);

   //////////////////////////////////////////////////////////////////////////////////////////  
   //////////////////////////////////////////////////////////////////////////////////////////

   // defines the seed
   if (args.hasArgument("seed"))
   {
      unsigned int seed = args.getValue<unsigned int>("seed", 0);
      srand(seed);
   }

   //////////////////////////////////////////////////////////////////////////////////////////  
   //////////////////////////////////////////////////////////////////////////////////////////

   GenericStrongLearner* pModel = NULL;

   if ( args.hasArgument("train") ||
        args.hasArgument("traintest") )
   {

      // get the name of the learner
      string baseLearnerName = defaultLearner;
      if ( args.hasArgument("learnertype") )
          args.getValue("learnertype", 0, baseLearnerName);

      checkBaseLearner(baseLearnerName);
      if (verbose > 1)    
         cout << "--> Using learner: " << baseLearnerName << endl;

      // This hould be changed: the user decides the strong learner
      BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
      pModel = pWeakHypothesisSource->createGenericStrongLearner( args );

      pModel->run(args);
   }
   //////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   else if ( args.hasArgument("test") )
   {
      // -test <dataFile> <shypFile> <numIters>
      string shypFileName = args.getValue<string>("test", 1);

      string baseLearnerName = UnSerialization::getWeakLearnerName(shypFileName);
      BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
      pModel = pWeakHypothesisSource->createGenericStrongLearner( args );

      pModel->classify(args);
   }
   //////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   else if ( args.hasArgument("cmatrix") )
   {
      // -cmatrix <dataFile> <shypFile>

      string shypFileName = args.getValue<string>("cmatrix", 1);

      string baseLearnerName = UnSerialization::getWeakLearnerName(shypFileName);
      BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
      pModel = pWeakHypothesisSource->createGenericStrongLearner( args );

      pModel->doConfusionMatrix(args);
   }
   //////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   else if ( args.hasArgument("likelihood") )
   {
      // -posteriors <dataFile> <shypFile> <outFileName>
      string shypFileName = args.getValue<string>("likelihood", 1);

      string baseLearnerName = UnSerialization::getWeakLearnerName(shypFileName);
      BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
      pModel = pWeakHypothesisSource->createGenericStrongLearner( args );

      pModel->doLikelihoods(args);
   }   
   //////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   else if ( args.hasArgument("roc") )
   {
      // -posteriors <dataFile> <shypFile> <outFileName>
      string shypFileName = args.getValue<string>("roc", 1);

      string baseLearnerName = UnSerialization::getWeakLearnerName(shypFileName);
      BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
      pModel = pWeakHypothesisSource->createGenericStrongLearner( args );

      pModel->doROC(args);
   }   
   //////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   else if ( args.hasArgument("posteriors") )
   {
      // -posteriors <dataFile> <shypFile> <outFileName>
      string shypFileName = args.getValue<string>("posteriors", 1);

      string baseLearnerName = UnSerialization::getWeakLearnerName(shypFileName);
      BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
      pModel = pWeakHypothesisSource->createGenericStrongLearner( args );

      pModel->doPosteriors(args);
   }   
   //////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   else if ( args.hasArgument("cposteriors") )
   {
      // -posteriors <dataFile> <shypFile> <outFileName>
      string shypFileName = args.getValue<string>("cposteriors", 1);

      string baseLearnerName = UnSerialization::getWeakLearnerName(shypFileName);
      BaseLearner*  pWeakHypothesisSource = BaseLearner::RegisteredLearners().getLearner(baseLearnerName);
      pModel = pWeakHypothesisSource->createGenericStrongLearner( args );

	  pModel->doCalibratedPosteriors(args);
   }   
   //////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   else if ( args.hasArgument("ssfeatures") )
   {
      // ONLY for AdaBoostMH classifiers

      // -ssfeatures <dataFile> <shypFile> <outFile> <numIters>
      string testFileName = args.getValue<string>("ssfeatures", 0);
      string shypFileName = args.getValue<string>("ssfeatures", 1);
      string outFileName = args.getValue<string>("ssfeatures", 2);
      int numIterations = args.getValue<int>("ssfeatures", 3);

      cerr << "ERROR: ssfeatures has been deactivated for the moment!" << endl;

      //ABMHClassifierYahoo classifier(args, verbose);
      //classifier.saveSingleStumpFeatureData(testFileName, shypFileName, outFileName, numIterations);
   }

   //////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////
   else if ( args.hasArgument("encode") )
   {

      // --encode <inputDataFile> <outputDataFile> <nIterations> <poolFile> <nBaseLearners>
      string labelsFileName = args.getValue<string>("encode", 0);
      string autoassociativeFileName = args.getValue<string>("encode", 1);
      string outputFileName = args.getValue<string>("encode", 2);
      int numIterations = args.getValue<int>("encode", 3);
      string poolFileName = args.getValue<string>("encode", 4);
      int numBaseLearners = args.getValue<int>("encode", 5);
      string outputInfoFile;
      const char* tmpArgv1[] = {"bla", // for ParasiteLearner
			       "--pool",
			       args.getValue<string>("encode", 4).c_str(),
			       args.getValue<string>("encode", 5).c_str()};
      args.readArguments(4,tmpArgv1);

      InputData* pAutoassociativeData = new InputData();
      pAutoassociativeData->initOptions(args);
      pAutoassociativeData->load(autoassociativeFileName,IT_TRAIN,verbose);

      // for the original labels
      InputData* pLabelsData = new InputData();
      pLabelsData->initOptions(args);
      pLabelsData->load(labelsFileName,IT_TRAIN,verbose);

      // set up all the InputData members identically to pAutoassociativeData
      EncodeData* pOnePoint = new EncodeData();
      pOnePoint->initOptions(args);
      pOnePoint->load(autoassociativeFileName,IT_TRAIN,verbose);

      const int numExamples = pAutoassociativeData->getNumExamples();
      BaseLearner* pWeakHypothesisSource = 
	  BaseLearner::RegisteredLearners().getLearner("ParasiteLearner");
      pWeakHypothesisSource->declareArguments(args);

      ParasiteLearner* pWeakHypothesis;

      ofstream outFile(outputFileName.c_str());
      if (!outFile.is_open())
      {
	  cerr << "ERROR: Cannot open strong hypothesis file <" << outputFileName << ">!" << endl;
	  exit(1);
      }

      for (int i = 0; i < numExamples ; ++i)
      {
	 vector<float> alphas;
	 alphas.resize(numBaseLearners);
	 fill(alphas.begin(), alphas.end(), 0);

	 if (verbose >= 1)
	    cout << "--> Encoding example no " << (i+1) << endl;
         pOnePoint->resetData();
	 pOnePoint->addExample( pAutoassociativeData->getExample(i) );
	 float energy = 1;

	 OutputInfo* pOutInfo = NULL;
	 if ( args.hasArgument("outputinfo") ) 
	 {
	    args.getValue("outputinfo", 0, outputInfoFile);
	    pOutInfo = new OutputInfo(outputInfoFile);
	    pOutInfo->initialize(pOnePoint);
	 }


	 for (int t = 0; t < numIterations; ++t)
	 {
	    pWeakHypothesis = (ParasiteLearner*)pWeakHypothesisSource->create();
	    pWeakHypothesis->initLearningOptions(args);
	    pWeakHypothesis->setTrainingData(pOnePoint);
 	    energy *= pWeakHypothesis->run();
 // 	    if (verbose >= 2)
//  	       cout << "energy = " << energy << endl << flush;
	    AdaBoostMHLearner adaBoostMHLearner;

	    if (i == 0 && t == 0)
	    {
	       if ( pWeakHypothesis->getBaseLearners().size() < numBaseLearners )
		  numBaseLearners = pWeakHypothesis->getBaseLearners().size();
	       outFile << "%Hidden representation using autoassociative boosting" << endl << endl;
	       outFile << "@RELATION " << outputFileName << endl << endl;
	       outFile << "% numBaseLearners" << endl;
	       for (int j = 0; j < numBaseLearners; ++j) 
	          outFile << "@ATTRIBUTE " << j << "_" <<
		     pWeakHypothesis->getBaseLearners()[j]->getId() << " NUMERIC" << endl;
	       outFile << "@ATTRIBUTE class {" << pLabelsData->getClassMap().getNameFromIdx(0);
	       for (int l = 1; l < pLabelsData->getClassMap().getNumNames(); ++l)
		  outFile << ", " << pLabelsData->getClassMap().getNameFromIdx(l);
	       outFile << "}" << endl<< endl<< "@DATA" << endl;
	    }
	    alphas[pWeakHypothesis->getSelectedIndex()] += 
	       pWeakHypothesis->getAlpha() * pWeakHypothesis->getSignOfAlpha();
	    if ( pOutInfo )
	       adaBoostMHLearner.printOutputInfo(pOutInfo, t, pOnePoint, NULL, pWeakHypothesis);
	    adaBoostMHLearner.updateWeights(pOnePoint,pWeakHypothesis);
	 }
	 float sumAlphas = 0;
	 for (int j = 0; j < numBaseLearners; ++j)
	    sumAlphas += alphas[j];

	 for (int j = 0; j < numBaseLearners; ++j)
	    outFile << alphas[j]/sumAlphas << ",";
	 const vector<Label>& labels = pLabelsData->getLabels(i);
	 for (int l = 0; l < labels.size(); ++l)
	    if (labels[l].y > 0)
	       outFile << pLabelsData->getClassMap().getNameFromIdx(labels[l].idx) << endl;
	 delete pOutInfo;
      }
      outFile.close();
   }

   if (pModel)
      delete pModel;

   exit_shogun();
   return 0;
}

// -----------------------------------------------------------------------------
