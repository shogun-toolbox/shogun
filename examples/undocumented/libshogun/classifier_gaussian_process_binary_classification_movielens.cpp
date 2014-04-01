/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 pl8787
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
* either expressed or implied, of the Shogun Development Team.
*/

#include <shogun/lib/config.h>

// Eigen3 is required for working with this example
#ifdef HAVE_EIGEN3

#include <shogun/base/init.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/classifier/GaussianProcessBinaryClassification.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/io/CSVFile.h>
#include <shogun/io/LineReader.h>
#include <shogun/io/Parser.h>
#include <shogun/lib/DelimiterTokenizer.h>

using namespace shogun;

// files with training data
const char* fname_ratings_train="../data/ml-100k/u1.base";
const char* fname_items="../data/ml-100k/u.item";

// file with testing data
const char* fname_ratings_test="../data/ml-100k/u1.test";

// Read movielens ratings data (uid \t mid \t ratting \t timestamp)
SGMatrix<float64_t> read_ratings_data(const char* fname)
{
	SGMatrix<float64_t> x;
	CCSVFile* file_ratings=new CCSVFile(fname);
	file_ratings->set_delimiter('\t');
	x.load(file_ratings);
	SG_UNREF(file_ratings);

	SG_SPRINT("x:%dx%d\n", x.num_rows, x.num_cols);
	return x;
}

// Read movielens movie's features
// (Movie gener, ignore string features such as movie name)
SGMatrix<float64_t> read_items_data(const char* fname)
{
	FILE* file_items=fopen(fname,"rb");
	int32_t num_lines=0;
	int32_t num_tokens=20;
	int32_t current_line_idx=0;
	SGVector<char> line;

	CDelimiterTokenizer *line_tokenizer=new CDelimiterTokenizer(true);
	line_tokenizer->delimiters['\n'] = 1;
	CDelimiterTokenizer *tokenizer=new CDelimiterTokenizer(false);
	tokenizer->delimiters['|'] = 1;
	CLineReader *line_reader=new CLineReader(file_items,line_tokenizer);
	SG_REF(line_reader);
	CParser *parser=new CParser();
	SG_REF(parser);
	parser->set_tokenizer(tokenizer);

	while (line_reader->has_next())
	{
		num_lines++;
		line_reader->skip_line();
	}

	line_reader->reset();

	SGMatrix<float64_t> x(num_tokens, num_lines);

	while (line_reader->has_next())
	{
		line=line_reader->read_line();
		parser->set_text(line);

		int current_token_idx=0;
		for (int32_t i=0; i<num_tokens+4; i++)
		{
			if (i>=1 && i<=4)
			{
				parser->skip_token();
				continue;
			}
			x(current_token_idx, current_line_idx)=parser->read_real();
			current_token_idx++;
		}
		current_line_idx++;
	}

	fclose(file_items);
	SG_UNREF(parser);
	SG_UNREF(line_reader);
	return x;
}

// Generate feature matrix and target vector for one user specific by uid
// start_idx is the start line for searching in the rating file
void generate_features(SGMatrix<float64_t> &ratings, SGMatrix<float64_t> &items,
		int &uid, int &start_idx,
		SGMatrix<float64_t> &feature, SGVector<float64_t> &target)
{
	int item_len = 0;
	int next_start_idx = 0;
	for (int i=start_idx;i<ratings.num_cols; i++)
	{
		if (ratings(0,i)==uid && (ratings(2,i)==1 || ratings(2,i)==5))
			item_len++;

		if (ratings(0,i)!=uid)
		{
			next_start_idx = i;
			break;
		}
	}

	feature = SGMatrix<float64_t>(items.num_rows-1,item_len);
	target = SGVector<float64_t>(item_len);

	for (int i=start_idx, k=0; i<ratings.num_cols && k<item_len; i++)
	{
		if (ratings(0,i)==uid && (ratings(2,i)==1 || ratings(2,i)==5))
		{
			for (int j=1; j<items.num_rows; j++)
				feature(j-1, k)=items(j, (int)ratings(1,i)-1);
			target[k]=ratings(2,i)==1 ? -1:1;
			k++;
		}
	}

	start_idx = next_start_idx;
}

// Do the GP binary classification on movielens dataset
void gp_regression_movielens(float64_t &accuracy_train, float64_t &accuracy_test)
{
	// Basic information
	int user_cnt = 943;
	//int item_cnt = 1682;

	int pred_user_count = 0;
	int loss_user_count = 0;
	int pred_pair_count_train = 0;
	int pred_pair_count_test = 0;

	// Read movielens ratings data from u1.base and u1.test
	// where uid is in increased order
	SGMatrix<float64_t> M_train_ratings=read_ratings_data(fname_ratings_train);
	SGMatrix<float64_t> M_test_ratings=read_ratings_data(fname_ratings_test);

	// Read movielens movie genre data from u.item
	SGMatrix<float64_t> M_items=read_items_data(fname_items);

	// Allocate our mean function
	CZeroMean* mean = new CZeroMean();
	SG_REF(mean);

	// Allocate our likelihood function
	CLogitLikelihood* lik = new CLogitLikelihood();
	SG_REF(lik);

	// Allocate our Kernel
	float64_t kernel_sigma = 2;
	CGaussianKernel* kernel = new CGaussianKernel(10, kernel_sigma);
	SG_REF(kernel);

	// Calculate accuracy of train and test
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation(ACCURACY);
	SG_REF(eval);

	int uid = 1;
	int start_idx_train = 0;
	int start_idx_test = 0;

	for (uid = 1;uid<=user_cnt;uid++)
	{
		SGMatrix<float64_t> M_train;
		SGMatrix<float64_t> M_test;
		SGVector<float64_t> V_train;
		SGVector<float64_t> V_test;

		// Generate train and test data from u1.base and u1.test
		generate_features(M_train_ratings, M_items, uid, start_idx_train, M_train, V_train);
		generate_features(M_test_ratings, M_items, uid, start_idx_test, M_test, V_test);

		if (V_train.vlen==0 && V_test.vlen!=0)
			loss_user_count++;

		if (V_train.vlen==0 || V_test.vlen==0)
			continue;

		// Convert training and testing data into shogun representation
		CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(M_train);
		CBinaryLabels* lab_train=new CBinaryLabels(V_train);
		CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(M_test);
		CBinaryLabels* lab_test=new CBinaryLabels(V_test);
		SG_REF(feat_train);
		SG_REF(lab_train);
		SG_REF(feat_test);
		SG_REF(lab_test);

		// specify EP approximation inference method
		CEPInferenceMethod* inf=new CEPInferenceMethod(kernel,
							feat_train, mean, lab_train, lik);
		SG_REF(inf);
		inf->set_scale(1.0);	// Parameter of kernel scale

		// Finally use these to allocate the Gaussian Process Object
		CGaussianProcessBinaryClassification* gpc = new CGaussianProcessBinaryClassification(inf);
		SG_REF(gpc);
		gpc->train();

		pred_user_count++;
		pred_pair_count_train+=V_train.vlen;
		pred_pair_count_test+=V_test.vlen;

		// perform inference on train
		CBinaryLabels* predictions_train=gpc->apply_binary(feat_train);
		SG_REF(predictions_train);

		// perform inference on test
		CBinaryLabels* predictions_test=gpc->apply_binary(feat_test);
		SG_REF(predictions_test);

		accuracy_train += eval->evaluate(predictions_train, lab_train) * V_train.vlen;
		accuracy_test += eval->evaluate(predictions_test, lab_test) * V_test.vlen;

		SG_SPRINT("Processing User:%d\n", uid);
		SG_SPRINT("Train Vlen: %d\n", V_train.vlen);
		SG_SPRINT("Test Vlen: %d\n", V_test.vlen);
		SG_SPRINT("TP+TN on Train: %lf\n", accuracy_train);
		SG_SPRINT("TP+TN on Test: %lf\n", accuracy_test);
		SG_SPRINT("Accuracy on Train:%lf\n",accuracy_train/pred_pair_count_train);
		SG_SPRINT("Accuracy on Test:%lf\n",accuracy_test/pred_pair_count_test);
		SG_SPRINT("\n");

		SG_UNREF(predictions_train);
		SG_UNREF(predictions_test);
		SG_UNREF(feat_train);
		SG_UNREF(lab_train);
		SG_UNREF(feat_test);
		SG_UNREF(lab_test);
		SG_UNREF(inf);
		SG_UNREF(gpc);
	}

	SG_SPRINT("Train Pairs: %d\n",pred_pair_count_train);
	SG_SPRINT("Test Pairs: %d\n",pred_pair_count_test);

	accuracy_train/=pred_pair_count_train;
	accuracy_test/=pred_pair_count_test;

	SG_SPRINT("Loss User Count in Test:%d\n", loss_user_count);

	SG_UNREF(mean);
	SG_UNREF(lik);
	SG_UNREF(kernel);
	SG_UNREF(eval);
}

// Main of this GP on movielens example
int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	float64_t accuracy_test = 0.0;
	float64_t accuracy_train = 0.0;

	gp_regression_movielens(accuracy_train, accuracy_test);

	SG_SPRINT("Accuracy on Train: %lf%%\n", accuracy_train*100);
	SG_SPRINT("Accuracy on Test: %lf%%\n", accuracy_test*100);

	exit_shogun();
	return 0;
}

#endif /* HAVE_EIGEN3 */
