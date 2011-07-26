using System;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.randn;

public class evaluation_contingencytableevaluation_modular
{
	static evaluation_contingencytableevaluation_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	static void Main(string[] argv)
	{
		modshogun.init_shogun_with_defaults();

		DoubleMatrix ground_truth = Load.load_labels("../data/label_train_twoclass.dat");
		DoubleMatrix predicted = randn(1, ground_truth.Length);

		Labels ground_truth_labels = new Labels(ground_truth);
		Labels predicted_labels = new Labels(predicted);

		ContingencyTableEvaluation base_evaluator = new ContingencyTableEvaluation();
		base_evaluator.evaluate(predicted_labels,ground_truth_labels);

		AccuracyMeasure evaluator1 = new AccuracyMeasure();
		double accuracy = evaluator1.evaluate(predicted_labels,ground_truth_labels);

		ErrorRateMeasure evaluator2 = new ErrorRateMeasure();
		double errorrate = evaluator2.evaluate(predicted_labels,ground_truth_labels);

		BALMeasure evaluator3 = new BALMeasure();
		double bal = evaluator3.evaluate(predicted_labels,ground_truth_labels);

		WRACCMeasure evaluator4 = new WRACCMeasure();
		double wracc = evaluator4.evaluate(predicted_labels,ground_truth_labels);

		F1Measure evaluator5 = new F1Measure();
		double f1 = evaluator5.evaluate(predicted_labels,ground_truth_labels);

		CrossCorrelationMeasure evaluator6 = new CrossCorrelationMeasure();
		double crosscorrelation = evaluator6.evaluate(predicted_labels,ground_truth_labels);

		RecallMeasure evaluator7 = new RecallMeasure();
		double recall = evaluator7.evaluate(predicted_labels,ground_truth_labels);

		PrecisionMeasure evaluator8 = new PrecisionMeasure();
		double precision = evaluator8.evaluate(predicted_labels,ground_truth_labels);

		SpecificityMeasure evaluator9 = new SpecificityMeasure();
		double specificity = evaluator9.evaluate(predicted_labels,ground_truth_labels);

		Console.Write("{0:F}, {1:F}, {2:F}, {3:F}, {4:F}, {5:F}, {6:F}, {7:F}, {8:F}\n", accuracy, errorrate, bal, wracc, f1, crosscorrelation, recall, precision, specificity);

		modshogun.exit_shogun();
	}
}
