import org.shogun.*;
import org.jblas.*;
import static org.jblas.MatrixFunctions.signum;
import static org.jblas.DoubleMatrix.concatHorizontally;
import static org.jblas.DoubleMatrix.ones;
import static org.jblas.DoubleMatrix.randn;

import static org.shogun.LabelsFactory.to_binary;

public class classifier_libsvm_minimal_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();

		int num = 1000;
		double dist = 1.0;
		double width = 2.1;
		double C = 1.0;

		DoubleMatrix offs=ones(2, num).mmul(dist);
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
		BinaryLabels labels = new BinaryLabels(trainlab);
		LibSVM svm = new LibSVM(C, kernel, labels);
		svm.train();

		DoubleMatrix out = to_binary(svm.apply(feats_test)).get_labels();

		System.out.println("Mean Error = " + signum(out).ne(testlab).mean());
	}
}
