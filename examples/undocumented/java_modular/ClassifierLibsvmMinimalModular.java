import org.shogun.*;
import org.jblas.*;
public class ClassifierLibsvmMinimalModular {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
	}
	
	public static void main(String argv[]) {
		Features.init_shogun();
		int num = 1000;
		int dist = 1;
		double width = 2.1;
		int C = 1;

		DoubleMatrix x = DoubleMatrix.randn(2, num).sub(DoubleMatrix.ones(2, num));
		DoubleMatrix y = DoubleMatrix.randn(2, num).add(DoubleMatrix.ones(2, num));
		DoubleMatrix traindata_real = DoubleMatrix.concatHorizontally(x, y);

		DoubleMatrix m = DoubleMatrix.randn(2, num).sub(DoubleMatrix.ones(2, num));
		DoubleMatrix n = DoubleMatrix.randn(2, num).add(DoubleMatrix.ones(2, num));
		DoubleMatrix testdata_real = DoubleMatrix.concatHorizontally(m, n);

		DoubleMatrix trainlab = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(num).neg(), DoubleMatrix.ones(num));
		DoubleMatrix testlab = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(num).neg(), DoubleMatrix.ones(num));

		RealFeatures feats_train = new RealFeatures();
		feats_train.set_feature_matrix(traindata_real);
		RealFeatures feats_test = new RealFeatures();
		feats_test.set_feature_matrix(testdata_real);
		GaussianKernel kernel = new GaussianKernel(feats_train, feats_train, width);

		Labels labels = new Labels(trainlab);
		LibSVM svm = new LibSVM(C, kernel, labels);
		svm.train();

		kernel.init(feats_train, feats_test);
		double[] out = svm.apply().get_labels();
		Features.exit_shogun();
	}
}
