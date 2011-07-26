using System;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.randn;

public class evaluation_prcevaluation_modular
{
	static evaluation_prcevaluation_modular()
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

		PRCEvaluation evaluator = new PRCEvaluation();
		evaluator.evaluate(predicted_labels, ground_truth_labels);

		Console.WriteLine(evaluator.get_PRC());
		Console.WriteLine(evaluator.get_auPRC());

		modshogun.exit_shogun();
	}
}
