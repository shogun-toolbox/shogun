import org.shogun.*;
import org.jblas.*;

public class kernel_io_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 1.2;

		DoubleMatrix traindata_real = Load.load_numbers("../data/fm_train_real.dat");
		DoubleMatrix testdata_real = Load.load_numbers("../data/fm_test_real.dat");

		RealFeatures feats_train = new RealFeatures(traindata_real);
		RealFeatures feats_test = new RealFeatures(testdata_real);

		GaussianKernel kernel = new GaussianKernel(feats_train, feats_test, width);
		DoubleMatrix km_train = kernel.get_kernel_matrix();
		CSVFile f=new CSVFile("gaussian_train.ascii",'w');
		kernel.save(f);

		kernel.init(feats_train, feats_test);
		DoubleMatrix km_test = kernel.get_kernel_matrix();
		CSVFile f_test=new CSVFile("gaussian_train.ascii",'w');
		kernel.save(f_test);

		System.out.println(km_train.toString());
		System.out.println(km_test.toString());

		modshogun.exit_shogun();
	}
}
