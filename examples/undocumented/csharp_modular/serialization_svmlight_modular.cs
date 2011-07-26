using System;
using System.Collections;

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

public class serialization_svmlight_modular
{
	static serialization_svmlight_modular()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	public static void save(string fname, object obj)
	{
		try
		{
			FileOutputStream fs = new FileOutputStream(fname);
			ObjectOutputStream @out = new ObjectOutputStream(fs);
			@out.writeObject(obj);
			@out.close();
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
		}
	}

	public static object load(string fname)
	{
		object r = null;
		try
		{
			FileInputStream fs = new FileInputStream(fname);
			ObjectInputStream @in = new ObjectInputStream(fs);
			r = @in.readObject();
			@in.close();
			return r;
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
		}
		return r;
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
		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.train();

		ArrayList result = new ArrayList();
		result.Add(svm);
		string fname = "out.txt";
		//save(fname, (Serializable)result);
		//ArrayList r = (ArrayList)load(fname);
		//SVMLight svm2 = (SVMLight)r.get(0);

		modshogun.exit_shogun();
	}
}
