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
* \file LogitBoostClassifier.h Performs the classification with LogitBoost.
*/
#pragma warning( disable : 4786 )

#ifndef __LOGITBOOST_CLASSIFIER_H
#define __LOGITBOOST_CLASSIFIER_H

#include "classifier/boosting/Utils/Args.h"

#include <string>
#include <cassert>

using namespace std;

namespace shogun {

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

// Forward declarations.
class ExampleResults;
class InputData;
class BaseLearner;

/**
* Classify a dataset with AdaBoost.MH learner.
* Using the strong hypothesis file (shyp.xml by default) it builds the
* list of weak hypothesis (or weak learners), and use them to perform a classification over 
* the given data set. The strong hypothesis is the linear combination of the weak 
* hypotheses and their confidence alpha, and is defined as:
* \f[
* {\bf g}(x) = \sum_{t=1}^T \alpha^{(t)} {\bf h}^{(t)}(x),
* \f]
* where the bold defines a vector as returned value.
* To obtain a single class, we simply take the winning class that receives 
* the "most vote", that is:
* \f[
* f(x) = \mathop{\rm arg\, max}_{\ell} g_\ell(x).
* \f]
* \date 15/11/2005
*/
class LogitBoostClassifier
{
public:

   /**
   * The constructor. It initializes the variable and set them using the
   * information provided by the arguments passed. They are parsed
   * using the helpers provided by class Args.
   * \param args The arguments defined by the user in the command line.
   * \param verbose The level of verbosity
   * \see _verbose
   * \date 16/11/2005
   */
   LogitBoostClassifier(const nor_utils::Args& args, int verbose = 1);

   /**
   * Starts the classification process. 
   * \param dataFileName The file name of the data to be classified.
   * \param shypFileName The strong hypothesis filename. It is the xml file containing the
   * list of weak hypotheses that form the strong hypothesis.
   * \param outResFileName The name of the file in which the results of the classification
   * will be saved.
   * \param numRanksEnclosed This parameter defines the number of ranks to be printed.
   * \remark If \a numRanksEnclosed=1, the only error displayed will be the one in which the
   * \f$\mathop{\rm arg\, max}_{\ell} g_\ell(x)\f$ is \b not the correct class.
   * If \a numRanksEnclosed=2, in addition to the standard error, there will be also
   * the error in which the actual class is not the max, nor the second 
   * biggest of \f${\bf g}(x)\f$. With larger values of \a numRanksEnclosed it displays
   * also the other values.
   * This is useful for multi-class problems, when if the "first guess" was wrong
   * a "second guess" is allowed.
   * \remark If \a outResFileName is provided, the result
   * of the classification will be saved in a file with the following format:
   * \verbatim
   1 className
   2 className
   3 className
   ...\endverbatim
   * If \a --examplelabel is active the first column of the data file will be used instead
   * of the number of example.
   * \date 16/11/2005
   */
   void run(const string& dataFileName, const string& shypFileName, 
            int numIterations, const string& outResFileName = "", 
	    int numRanksEnclosed = 2);

   /**
   * Print to stdout a nicely formatted confusion matrix.
   * \param dataFileName The file name of the data to be classified.
   * \param shypFileName The strong hypothesis filename. It is the xml file containing the
   * list of weak hypotheses that form the strong hypothesis.
   * \date 10/2/2006
   */
   void printConfusionMatrix(const string& dataFileName, const string& shypFileName);

   /**
   * Output to a file a confusion matrix with every element separated by a tab.
   * \param dataFileName The file name of the data to be classified.
   * \param shypFileName The strong hypothesis filename. It is the xml file containing the
   * list of weak hypotheses that form the strong hypothesis.
   * \param outFileName The name of the file in which the confusion matrix will be saved.
   * \param numIterations The number of weak learners to use
   * \date 10/2/2006
   */
   void saveConfusionMatrix(const string& dataFileName, const string& shypFileName,
                            const string& outFileName);


   void saveCalibratedPosteriors(const string& dataFileName, const string& shypFileName,
                       const string& outFileName, int numIterations);


   void savePosteriors(const string& dataFileName, const string& shypFileName,
                       const string& outFileName, int numIterations);


   void saveLikelihoods(const string& dataFileName, const string& shypFileName,
                       const string& outFileName, int numIterations);

   /**
   * Save the data generated by using the strong hypothesis file of 
   * type SingleStumpLearner, and a given input data file. This kind of data
   * is intended for research, and the output has the following format:
   * \f[
   * \begin{array}({ccccc})
   * 0 & \alpha^{(1)} & \alpha^{(2)} & \cdots & \alpha^{(T)}\\& & & &\\
   * 0 & v^{(1)}_1 & v^{(2)}_1 & \cdots & v^{(T)}_1 \\& & & &\\
   * 0 & v^{(1)}_2 & v^{(2)}_2 & \cdots & v^{(T)}_2 \\& & & &\\
   * \vdots & \vdots & \vdots & \ddots & \vdots \\& & & &\\
   * 0 & v^{(1)}_K & v^{(2)}_K & \cdots & v^{(T)}_K \\& & & &\\
   * y_1 & \phi^{(1)}({\bf x}_1) & \phi^{(2)}({\bf x}_1) & \cdots & \phi^{(T)}({\bf x}_1) \\& & & &\\
   * y_2 & \phi^{(1)}({\bf x}_2) & \phi^{(2)}({\bf x}_2) & \cdots & \phi^{(T)}({\bf x}_2) \\& & & &\\
   * \vdots & \vdots & \vdots & \ddots & \vdots \\& & & &\\
   * y_n & \phi^{(1)}({\bf x}_n) & \phi^{(2)}({\bf x}_n) & \cdots & \phi^{(T)}({\bf x}_n)
   * \end{array}
   * \f]
   * where \f$v\f$ is the alignment vector and \f$\phi\f$ is the discriminative function.
   * \param dataFileName The name of the input data file.
   * \param shypFileName The name of the strong hypothesis file to load.
   * \param outFileName The name of the file in which the matrix will be saved.
   * \param numIterations The number of iterations up to which the computation should go (the "width"
   * of the matrix). If this parameter is 0, \b all the iterations will be used.
   * \remark Although we are referring to the output as a "matrix" the component of \f$y_i\f$ will
   * be output as a string, which refers to the original definition of the class in the data file.
   * \see StumpLearner::phi
   * \see SutmpLearner::_v
   * \date 10/2/2006
   * \remark TEMPORARLY OFF!!
   */
   //void saveSingleStumpFeatureData(const string& dataFileName, const string& shypFileName,
   //                                const string& outFileName, int numIterations = 0);

protected:

   /**
   * Loads the data. It needs the Strong Hypothesis file because it needs
   * the information about the weak learner used to generate it. The weak
   * learner might have associated a special InputData derived class,
   * which is returned by BaseLearner::createInputData() once the weak
   * learner has been identified.
   * \param dataFileName The file name of the data to be classified.
   * \param shypFileName The strong hypothesis filename. It is the xml file containing the
   * \warning The returned object must be destroyed by the caller.
   * \date 21/11/2005
   */
   InputData* loadInputData(const string& dataFileName, const string& shypFileName);

   /**
   * Compute the results using the weak hypotheses.
   * This method is the one that effectively computes \f${\bf g}(x)\f$.
   * \param pData A pointer to the data to be classified.
   * \param weakHypotheses The list of weak hypotheses.
   * \param results The vector where the results will be stored.
   * \see ExampleResults
   * \date 16/11/2005
   */
   void computeResults(InputData* pData, vector<BaseLearner*>& weakHypotheses, 
                       vector< ExampleResults* >& results, int numIterations );

   /**
   * Compute the overall error on the data.
   * \param pData A pointer to the data. Needed to get the actual class of 
   * the example.
   * \param results The vector where the results are hold.
   * \param atLeastRank The maximum rank in which the classification will not be considered
   * an error. If \a atLeastRank = 0, no errors are allowed. If it is 1, the second "guess"
   * will be taken into consideration, among the first, and so on.
   * \return The error.
   * \see ExampleResults
   * \date 16/11/2005
   */
   float getOverallError( InputData* pData, const vector< ExampleResults* >& results, 
                           int atLeastRank = 0 );

   /**
   * Compute the error per class.
   * \param pData A pointer to the data. Needed to get the actual class of 
   * the example.
   * \param results The vector where the results are hold.
   * \param classError The returned per class errors.
   * \param atLeastRank The maximum rank in which the classification will not be considered
   * an error. If \a atLeastRank = 0, no errors are allowed. If it is 1, the second "guess"
   * will be taken into consideration, among the first, and so on.
   * \see ExampleResults
   * \date 16/11/2005
   */
   void getClassError( InputData* pData,  const vector< ExampleResults* >& results, 
                       vector<float>& classError, int atLeastRank = 0  );

   /**
   * Defines the level of verbosity:
   * - 0 = no messages
   * - 1 = basic messages
   * - 2 = show all messages
   */
   int      _verbose;

   const nor_utils::Args&  _args;  //!< The arguments defined by the user.
   string   _outputInfoFile; //!< The filename of the step-by-step information file that will be updated 

private:

   /**
   * Fake assignment operator to avoid warning.
   * \date 6/12/2005
   */
   LogitBoostClassifier& operator=( const LogitBoostClassifier& ) {return *this;}

};

} // end of namespace shogun

#endif // __ADABOOST_MH_CLASSIFIER_H
