import org.shogun.*;
import org.jblas.*;
import static org.jblas.MatrixFunctions.signum;
import static org.jblas.DoubleMatrix.concatHorizontally;
import static org.jblas.DoubleMatrix.ones;
import static org.jblas.DoubleMatrix.randn;
import java.util.ArrayList;
import java.io.*;

public class serialization_complex_example {
	static {
		System.loadLibrary("Features");
		System.loadLibrary("Classifier");
		System.loadLibrary("Kernel");
		System.loadLibrary("Library");
	}

	public static void main(String argv[]) {
		Features.init_shogun_with_defaults();

		int num = 1000;
		double dist = 1.0;
		double width = 2.1;
		double C = 1.0;

		DoubleMatrix offs=ones(2, num).mmul(dist);
		DoubleMatrix x = randn(2, num).sub(offs);
		DoubleMatrix y = randn(2, num).add(offs);
		DoubleMatrix traindata_real = concatHorizontally(x, y);

		DoubleMatrix o = ones(1,num);
		DoubleMatrix trainlab = concatHorizontally(o.neg(), o);
		DoubleMatrix testlab = concatHorizontally(o.neg(), o);

		RealFeatures feats = new RealFeatures(traindata_real);
		GaussianKernel kernel = new GaussianKernel(feats, feats, width);
		Labels labels = new Labels(trainlab);
		GMNPSVM svm = new GMNPSVM(C, kernel, labels);
		feats.add_preprocessor(new NormOne());
		feats.add_preprocessor(new LogPlusOne());
		feats.set_preprocessed(1);
		svm.train(feats);

		SerializableAsciiFile fstream = new SerializableAsciiFile("blaah.asc", 'w');
		//svm.save_serializable(fstream);

		Features.exit_shogun();
	}
}
