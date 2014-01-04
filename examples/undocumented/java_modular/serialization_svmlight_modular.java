import org.shogun.*;
import org.jblas.*;
import static org.jblas.MatrixFunctions.signum;
import static org.jblas.DoubleMatrix.concatHorizontally;
import static org.jblas.DoubleMatrix.ones;
import static org.jblas.DoubleMatrix.randn;
import java.util.ArrayList;
import java.io.*;

public class serialization_svmlight_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void save(String fname, Object obj) {
		try {
			ObjectOutputStream out =  new ObjectOutputStream(new FileOutputStream(fname));
			out.writeObject(obj);
			out.close();
	}
		catch(Exception ex){
			ex.printStackTrace();
	}
	}

	public static Object load(String fname) {
		Object r = null;
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(fname));
			r = in.readObject();
			in.close();
			return r;
	}
		catch(Exception ex){
			ex.printStackTrace();
	}
		return r;
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
		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.train();

		String fname = "tmp.dat";
		save(fname, svm);
		SVMLight svm_readed = (SVMLight)load(fname);

	}
}
