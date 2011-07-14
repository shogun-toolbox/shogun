using System;

using org.shogun;
using org.jblas;
// 'import static' statement cannot be converted to .NET:
import static org.jblas.MatrixFunctions.signum;
// 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.concatHorizontally;
// 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.ones;
// 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.randn;

public class classifier_libsvm_minimal_modular
{
	static classifier_libsvm_minimal_modular()
	{
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();

		int num = 1000;
		double dist = 1.0;
		double width = 2.1;
		double C = 1.0;

		DoubleMatrix offs =ones(2, num).mmul(dist);
		DoubleMatrix x = randn(2, num).sub(offs);
		DoubleMatrix y = randn(2, num).add(offs);
		DoubleMatrix traindata_real = concatHorizontally(x, y);

		DoubleMatrix m = randn(2, num).sub(offs);
		DoubleMatrix n = randn(2, num).add(offs);
		DoubleMatrix testdata_real = concatHorizontally(m, n);

		DoubleMatrix o = ones(1,num);
		DoubleMatrix trainlab = concatHorizontally(o.neg(), o);
		DoubleMatrix testlab = concatHorizontally(o.neg(), o);

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);
		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);
		Labels labels = new Labels(trainlab);
		LibSVM svm = new LibSVM(C, kernel, labels);
		svm.train();

		DoubleMatrix @out = svm.apply(feats_test).get_labels();

		Console.WriteLine("Mean Error = " + signum(@out).ne(testlab).mean());
		Features.exit_shogun();
	}
}
