using System;

using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.randn;

public class evaluation_multiclassaccuracy_modular
{
	static evaluation_multiclassaccuracy_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	static void Main(string[] argv)
	{
		modshogun.init_shogun_with_defaults();

		double mul = 2.0;
		DoubleMatrix ground_truth = Load.load_labels("../data/label_train_multiclass.dat");
		DoubleMatrix predicted = Load.load_labels("../data/label_train_multiclass.dat").mmul(mul);

		Labels ground_truth_labels = new Labels(ground_truth);
		Labels predicted_labels = new Labels(predicted);

		MulticlassAccuracy evaluator = new MulticlassAccuracy();
		double accuracy = evaluator.evaluate(predicted_labels, ground_truth_labels);

		Console.WriteLine(accuracy);

		modshogun.exit_shogun();
	}
}
