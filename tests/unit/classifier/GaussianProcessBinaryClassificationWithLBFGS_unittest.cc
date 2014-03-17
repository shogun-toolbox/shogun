/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/LaplacianInferenceMethodWithLBFGS.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/classifier/GaussianProcessBinaryClassification.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>



using namespace shogun;
float64_t get_abs_tolorance_classifier(float64_t true_value, float64_t rel_tolorance){
  rel_tolorance = CMath::abs(rel_tolorance);
  return true_value==0.0 ? rel_tolorance : CMath::abs(true_value * rel_tolorance);
}

TEST(GaussianProcessBinaryClassificationWithLBFGS,get_mean_vector)
{
  float64_t abs_tolorance;
  float64_t rel_tolorance=1e-2;
	// create some easy random classification data
	index_t n=10, m1=25, i=0;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);
	SGMatrix<float64_t> feat_test(2, m1);

	feat_train(0, 0)=0.0919736;
	feat_train(0, 1)=-0.3813827;
	feat_train(0, 2)=-1.8011128;
	feat_train(0, 3)=-1.4603061;
	feat_train(0, 4)=-0.1386884;
	feat_train(0, 5)=0.7827657;
	feat_train(0, 6)=-0.1369808;
	feat_train(0, 7)=0.0058596;
	feat_train(0, 8)=0.1059573;
	feat_train(0, 9)=-1.3059609;

	feat_train(1, 0)=1.4186892;
	feat_train(1, 1)=0.2271813;
	feat_train(1, 2)=0.3451326;
	feat_train(1, 3)=0.4495962;
	feat_train(1, 4)=1.2066144;
	feat_train(1, 5)=-0.5425118;
	feat_train(1, 6)=1.3479000;
	feat_train(1, 7)=0.7181545;
	feat_train(1, 8)=0.4036014;
	feat_train(1, 9)=0.8928408;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=-1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;
	lab_train[5]=1.0;
	lab_train[6]=-1.0;
	lab_train[7]=1.0;
	lab_train[8]=1.0;
	lab_train[9]=-1.0;

	// create test features
	for (index_t x1=-2; x1<=2; x1++)
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf=new CLaplacianInferenceMethodWithLBFGS(kernel,
		features_train,	mean, labels_train, likelihood);
  
  int m = 100;
  int max_linesearch = 1000;
  int linesearch = 0;
  int max_iterations = 1000;
  float64_t delta = 1e-15;
  int past = 0;
  float64_t epsilon = 1e-15;
  bool enable_newton_if_fail = false;
  inf->set_lbfgs_parameter(m, 
                           max_linesearch,
                           linesearch,
                           max_iterations,
                           delta, 
                           past, 
                           epsilon,
                           enable_newton_if_fail
                          );

	// train Gaussian process binary classifier
	CGaussianProcessBinaryClassification* gpc=new CGaussianProcessBinaryClassification(inf);
	gpc->train();

	// compare mean vector with result form GPML
	SGVector<float64_t> mean_vector=gpc->get_mean_vector(features_test);

  /*
  mean=
  -0.023547066779433
  -0.164420972889231
  -0.447812356229495
  -0.472428809447940
  -0.205391227282142
  -0.011335213830652
  -0.131012850981580
  -0.427259580375569
  -0.527281189501774
  -0.274684117023014
   0.055529455358847
   0.152023871056183
   0.174282413372574
   0.010823181344098
  -0.072772631266962
   0.090191676357209
   0.288417744414623
   0.409275122823904
   0.281220920795101
   0.088382525159406
   0.043796091667543
   0.130461505967524
   0.170564691797896
   0.113006930991411
   0.041654120309486
  */ 

  abs_tolorance = get_abs_tolorance_classifier(-0.023547066779433, rel_tolorance);
  EXPECT_NEAR(mean_vector[0],  -0.023547066779433,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.164420972889231, rel_tolorance);
  EXPECT_NEAR(mean_vector[1],  -0.164420972889231,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.447812356229495, rel_tolorance);
  EXPECT_NEAR(mean_vector[2],  -0.447812356229495,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.472428809447940, rel_tolorance);
  EXPECT_NEAR(mean_vector[3],  -0.472428809447940,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.205391227282142, rel_tolorance);
  EXPECT_NEAR(mean_vector[4],  -0.205391227282142,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.011335213830652, rel_tolorance);
  EXPECT_NEAR(mean_vector[5],  -0.011335213830652,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.131012850981580, rel_tolorance);
  EXPECT_NEAR(mean_vector[6],  -0.131012850981580,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.427259580375569, rel_tolorance);
  EXPECT_NEAR(mean_vector[7],  -0.427259580375569,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.527281189501774, rel_tolorance);
  EXPECT_NEAR(mean_vector[8],  -0.527281189501774,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.274684117023014, rel_tolorance);
  EXPECT_NEAR(mean_vector[9],  -0.274684117023014,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.055529455358847, rel_tolorance);
  EXPECT_NEAR(mean_vector[10],  0.055529455358847,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.152023871056183, rel_tolorance);
  EXPECT_NEAR(mean_vector[11],  0.152023871056183,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.174282413372574, rel_tolorance);
  EXPECT_NEAR(mean_vector[12],  0.174282413372574,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.010823181344098, rel_tolorance);
  EXPECT_NEAR(mean_vector[13],  0.010823181344098,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(-0.072772631266962, rel_tolorance);
  EXPECT_NEAR(mean_vector[14],  -0.072772631266962,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.090191676357209, rel_tolorance);
  EXPECT_NEAR(mean_vector[15],  0.090191676357209,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.288417744414623, rel_tolorance);
  EXPECT_NEAR(mean_vector[16],  0.288417744414623,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.409275122823904, rel_tolorance);
  EXPECT_NEAR(mean_vector[17],  0.409275122823904,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.281220920795101, rel_tolorance);
  EXPECT_NEAR(mean_vector[18],  0.281220920795101,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.088382525159406, rel_tolorance);
  EXPECT_NEAR(mean_vector[19],  0.088382525159406,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.043796091667543, rel_tolorance);
  EXPECT_NEAR(mean_vector[20],  0.043796091667543,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.130461505967524, rel_tolorance);
  EXPECT_NEAR(mean_vector[21],  0.130461505967524,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.170564691797896, rel_tolorance);
  EXPECT_NEAR(mean_vector[22],  0.170564691797896,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.113006930991411, rel_tolorance);
  EXPECT_NEAR(mean_vector[23],  0.113006930991411,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.041654120309486, rel_tolorance);
  EXPECT_NEAR(mean_vector[24],  0.041654120309486,  abs_tolorance);


  SG_UNREF(gpc);
}

TEST(GaussianProcessBinaryClassificationWithLBFGS,get_variance_vector)
{
  float64_t abs_tolorance;
  float64_t rel_tolorance=1e-2;
  // create some easy random classification data
  index_t n=10, m1=25, i=0;

  SGMatrix<float64_t> feat_train(2, n);
  SGVector<float64_t> lab_train(n);
  SGMatrix<float64_t> feat_test(2, m1);

  feat_train(0, 0)=0.0919736;
  feat_train(0, 1)=-0.3813827;
  feat_train(0, 2)=-1.8011128;
  feat_train(0, 3)=-1.4603061;
  feat_train(0, 4)=-0.1386884;
  feat_train(0, 5)=0.7827657;
  feat_train(0, 6)=-0.1369808;
  feat_train(0, 7)=0.0058596;
  feat_train(0, 8)=0.1059573;
  feat_train(0, 9)=-1.3059609;

  feat_train(1, 0)=1.4186892;
  feat_train(1, 1)=0.2271813;
  feat_train(1, 2)=0.3451326;
  feat_train(1, 3)=0.4495962;
  feat_train(1, 4)=1.2066144;
  feat_train(1, 5)=-0.5425118;
  feat_train(1, 6)=1.3479000;
  feat_train(1, 7)=0.7181545;
  feat_train(1, 8)=0.4036014;
  feat_train(1, 9)=0.8928408;

  lab_train[0]=1.0;
  lab_train[1]=-1.0;
	lab_train[2]=-1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;
	lab_train[5]=1.0;
	lab_train[6]=-1.0;
	lab_train[7]=1.0;
	lab_train[8]=1.0;
	lab_train[9]=-1.0;

	// create test features
	for (index_t x1=-2; x1<=2; x1++)
	{
		for (index_t x2=-2; x2<=2; x2++)
		{
			feat_test(0, i)=(float64_t)x1;
			feat_test(1, i)=(float64_t)x2;
			i++;
		}
	}

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethodWithLBFGS* inf=new CLaplacianInferenceMethodWithLBFGS(kernel,
		features_train,	mean, labels_train, likelihood);
  
  int m = 100;
  int max_linesearch = 1000;
  int linesearch = 0;
  int max_iterations = 1000;
  float64_t delta = 1e-15;
  int past = 0;
  float64_t epsilon = 1e-15;
  bool enable_newton_if_fail = false;
  inf->set_lbfgs_parameter(m, 
                           max_linesearch,
                           linesearch,
                           max_iterations,
                           delta, 
                           past, 
                           epsilon,
                           enable_newton_if_fail
                          );

	// train gaussian process classifier
	CGaussianProcessBinaryClassification* gpc=new CGaussianProcessBinaryClassification(inf);
	gpc->train();

	// compare variance vector with result form GPML
	SGVector<float64_t> variance_vector=gpc->get_variance_vector(features_test);
  /*
  var=
   0.999445535646085
   0.972965743674159
   0.799464093608188
   0.776811020003602
   0.957814443755535
   0.999871512927413
   0.982835632877678
   0.817449250977293
   0.721974547197594
   0.924548635855287
   0.996916479587550
   0.976888742629093
   0.969625640389031
   0.999882858745593
   0.994704144138483
   0.991865461515876
   0.916815204706781
   0.832493873837478
   0.920914793707155
   0.992188529246447
   0.998081902354648
   0.982979795460686
   0.970907685911889
   0.987229433547902
   0.998264934261243
   */

  abs_tolorance = get_abs_tolorance_classifier(0.999445535646085, rel_tolorance);
  EXPECT_NEAR(variance_vector[0],  0.999445535646085,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.972965743674159, rel_tolorance);
  EXPECT_NEAR(variance_vector[1],  0.972965743674159,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.799464093608188, rel_tolorance);
  EXPECT_NEAR(variance_vector[2],  0.799464093608188,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.776811020003602, rel_tolorance);
  EXPECT_NEAR(variance_vector[3],  0.776811020003602,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.957814443755535, rel_tolorance);
  EXPECT_NEAR(variance_vector[4],  0.957814443755535,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.999871512927413, rel_tolorance);
  EXPECT_NEAR(variance_vector[5],  0.999871512927413,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.982835632877678, rel_tolorance);
  EXPECT_NEAR(variance_vector[6],  0.982835632877678,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.817449250977293, rel_tolorance);
  EXPECT_NEAR(variance_vector[7],  0.817449250977293,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.721974547197594, rel_tolorance);
  EXPECT_NEAR(variance_vector[8],  0.721974547197594,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.924548635855287, rel_tolorance);
  EXPECT_NEAR(variance_vector[9],  0.924548635855287,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.996916479587550, rel_tolorance);
  EXPECT_NEAR(variance_vector[10],  0.996916479587550,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.976888742629093, rel_tolorance);
  EXPECT_NEAR(variance_vector[11],  0.976888742629093,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.969625640389031, rel_tolorance);
  EXPECT_NEAR(variance_vector[12],  0.969625640389031,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.999882858745593, rel_tolorance);
  EXPECT_NEAR(variance_vector[13],  0.999882858745593,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.994704144138483, rel_tolorance);
  EXPECT_NEAR(variance_vector[14],  0.994704144138483,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.991865461515876, rel_tolorance);
  EXPECT_NEAR(variance_vector[15],  0.991865461515876,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.916815204706781, rel_tolorance);
  EXPECT_NEAR(variance_vector[16],  0.916815204706781,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.832493873837478, rel_tolorance);
  EXPECT_NEAR(variance_vector[17],  0.832493873837478,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.920914793707155, rel_tolorance);
  EXPECT_NEAR(variance_vector[18],  0.920914793707155,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.992188529246447, rel_tolorance);
  EXPECT_NEAR(variance_vector[19],  0.992188529246447,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.998081902354648, rel_tolorance);
  EXPECT_NEAR(variance_vector[20],  0.998081902354648,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.982979795460686, rel_tolorance);
  EXPECT_NEAR(variance_vector[21],  0.982979795460686,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.970907685911889, rel_tolorance);
  EXPECT_NEAR(variance_vector[22],  0.970907685911889,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.987229433547902, rel_tolorance);
  EXPECT_NEAR(variance_vector[23],  0.987229433547902,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.998264934261243, rel_tolorance);
  EXPECT_NEAR(variance_vector[24],  0.998264934261243,  abs_tolorance);
  

  SG_UNREF(gpc);
}

TEST(GaussianProcessBinaryClassificationWithLBFGS,get_probabilities)
{
  float64_t abs_tolorance;
  float64_t rel_tolorance=1e-2;
// create some easy random classification data
  index_t n=10, m1=25, i=0;

  SGMatrix<float64_t> feat_train(2, n);
  SGVector<float64_t> lab_train(n);
  SGMatrix<float64_t> feat_test(2, m1);

  feat_train(0, 0)=0.0919736;
  feat_train(0, 1)=-0.3813827;
  feat_train(0, 2)=-1.8011128;
  feat_train(0, 3)=-1.4603061;
  feat_train(0, 4)=-0.1386884;
  feat_train(0, 5)=0.7827657;
  feat_train(0, 6)=-0.1369808;
  feat_train(0, 7)=0.0058596;
  feat_train(0, 8)=0.1059573;
  feat_train(0, 9)=-1.3059609;

  feat_train(1, 0)=1.4186892;
  feat_train(1, 1)=0.2271813;
  feat_train(1, 2)=0.3451326;
  feat_train(1, 3)=0.4495962;
  feat_train(1, 4)=1.2066144;
  feat_train(1, 5)=-0.5425118;
  feat_train(1, 6)=1.3479000;
  feat_train(1, 7)=0.7181545;
  feat_train(1, 8)=0.4036014;
  feat_train(1, 9)=0.8928408;

  lab_train[0]=1.0;
  lab_train[1]=-1.0;
  lab_train[2]=-1.0;
  lab_train[3]=-1.0;
  lab_train[4]=-1.0;
  lab_train[5]=1.0;
  lab_train[6]=-1.0;
  lab_train[7]=1.0;
  lab_train[8]=1.0;
  lab_train[9]=-1.0;

  // create test features
  for (index_t x1=-2; x1<=2; x1++)
  {
    for (index_t x2=-2; x2<=2; x2++)
    {
      feat_test(0, i)=(float64_t)x1;
      feat_test(1, i)=(float64_t)x2;
      i++;
    }
  }

  // shogun representation of features and labels
  CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
  CBinaryLabels* labels_train=new CBinaryLabels(lab_train);
  CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

  // choose Gaussian kernel with sigma = 2 and zero mean function
  CGaussianKernel* kernel=new CGaussianKernel(10, 2);
  CZeroMean* mean=new CZeroMean();

  // probit likelihood
  CProbitLikelihood* likelihood=new CProbitLikelihood();

  // specify GP classification with Laplacian inference
  CLaplacianInferenceMethodWithLBFGS* inf=new CLaplacianInferenceMethodWithLBFGS(kernel,
    features_train,	mean, labels_train, likelihood);

  int m = 100;
  int max_linesearch = 1000;
  int linesearch = 0;
  int max_iterations = 1000;
  float64_t delta = 1e-15;
  int past = 0;
  float64_t epsilon = 1e-15;
  bool enable_newton_if_fail = false;
  inf->set_lbfgs_parameter(m, 
                           max_linesearch,
                           linesearch,
                           max_iterations,
                           delta, 
                           past, 
                           epsilon,
                           enable_newton_if_fail
                          );

  // train gaussian process classifier
  CGaussianProcessBinaryClassification* gpc=new CGaussianProcessBinaryClassification(inf);
  gpc->train();

  // compare probabilities with result form GPML
  SGVector<float64_t> probabilities=gpc->get_probabilities(features_test);

  /*
  probabilities=
   0.488226466245922
   0.417789511180478
   0.276093816652870
   0.263785590738910
   0.397304384814469
   0.494332392690781
   0.434493572227885
   0.286370205408434
   0.236359403085941
   0.362657942090414
   0.527764727350733
   0.576011934382649
   0.587141206106800
   0.505411594361785
   0.463613688406351
   0.545095837900176
   0.644208871583211
   0.704637561752594
   0.640610463004653
   0.544191265146420
   0.521898045707522
   0.565230752690983
   0.585282345866324
   0.556503466053284
   0.520827060710866
  */  

  abs_tolorance = get_abs_tolorance_classifier(0.488226466245922, rel_tolorance);
  EXPECT_NEAR(probabilities[0],  0.488226466245922,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.417789511180478, rel_tolorance);
  EXPECT_NEAR(probabilities[1],  0.417789511180478,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.276093816652870, rel_tolorance);
  EXPECT_NEAR(probabilities[2],  0.276093816652870,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.263785590738910, rel_tolorance);
  EXPECT_NEAR(probabilities[3],  0.263785590738910,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.397304384814469, rel_tolorance);
  EXPECT_NEAR(probabilities[4],  0.397304384814469,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.494332392690781, rel_tolorance);
  EXPECT_NEAR(probabilities[5],  0.494332392690781,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.434493572227885, rel_tolorance);
  EXPECT_NEAR(probabilities[6],  0.434493572227885,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.286370205408434, rel_tolorance);
  EXPECT_NEAR(probabilities[7],  0.286370205408434,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.236359403085941, rel_tolorance);
  EXPECT_NEAR(probabilities[8],  0.236359403085941,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.362657942090414, rel_tolorance);
  EXPECT_NEAR(probabilities[9],  0.362657942090414,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.527764727350733, rel_tolorance);
  EXPECT_NEAR(probabilities[10],  0.527764727350733,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.576011934382649, rel_tolorance);
  EXPECT_NEAR(probabilities[11],  0.576011934382649,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.587141206106800, rel_tolorance);
  EXPECT_NEAR(probabilities[12],  0.587141206106800,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.505411594361785, rel_tolorance);
  EXPECT_NEAR(probabilities[13],  0.505411594361785,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.463613688406351, rel_tolorance);
  EXPECT_NEAR(probabilities[14],  0.463613688406351,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.545095837900176, rel_tolorance);
  EXPECT_NEAR(probabilities[15],  0.545095837900176,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.644208871583211, rel_tolorance);
  EXPECT_NEAR(probabilities[16],  0.644208871583211,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.704637561752594, rel_tolorance);
  EXPECT_NEAR(probabilities[17],  0.704637561752594,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.640610463004653, rel_tolorance);
  EXPECT_NEAR(probabilities[18],  0.640610463004653,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.544191265146420, rel_tolorance);
  EXPECT_NEAR(probabilities[19],  0.544191265146420,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.521898045707522, rel_tolorance);
  EXPECT_NEAR(probabilities[20],  0.521898045707522,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.565230752690983, rel_tolorance);
  EXPECT_NEAR(probabilities[21],  0.565230752690983,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.585282345866324, rel_tolorance);
  EXPECT_NEAR(probabilities[22],  0.585282345866324,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.556503466053284, rel_tolorance);
  EXPECT_NEAR(probabilities[23],  0.556503466053284,  abs_tolorance);
  abs_tolorance = get_abs_tolorance_classifier(0.520827060710866, rel_tolorance);
  EXPECT_NEAR(probabilities[24],  0.520827060710866,  abs_tolorance);



  SG_UNREF(gpc);
}


#endif /* HAVE_EIGEN3 */
