using org.shogun;
using org.jblas;
// This Java 'import static' statement cannot be converted to .NET:
import static org.jblas.MatrixFunctions.signum;
// This Java 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.concatHorizontally;
// This Java 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.ones;
// This Java 'import static' statement cannot be converted to .NET:
import static org.jblas.DoubleMatrix.randn;

public class serialization_complex_example
{
	static serialization_complex_example()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	static void Main(string[] argv)
	{
		modshogun.init_shogun_with_defaults();

		int num = 1000;
		double dist = 1.0;
		double width = 2.1;
		double C = 1.0;

		DoubleMatrix offs =ones(2, num).mmul(dist);
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

		modshogun.exit_shogun();
	}
}
