import org.shogun.*;
import org.jblas.*;

public class kernel_custom_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		int dim = 7;

		DoubleMatrix data = DoubleMatrix.rand(dim, dim);
		RealFeatures feats = new RealFeatures(data);
		DoubleMatrix data_T = data.transpose();
		DoubleMatrix symdata = data.add(data_T);

		int cols = (1 + dim) * dim / 2;
		DoubleMatrix lowertriangle = DoubleMatrix.zeros(1, cols);
		int count = 0;
		for (int i = 0; i < dim; i ++) {
			for (int j = 0; j < dim; j++) {
				if (j <= i) {
					lowertriangle.put(0, count++, symdata.get(i,j));
				}
			}
		}

		CustomKernel kernel= new CustomKernel();
		kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle);
		DoubleMatrix km_triangletriangle = kernel.get_kernel_matrix();

		kernel.set_triangle_kernel_matrix_from_full(symdata);
		DoubleMatrix km_fulltriangle=kernel.get_kernel_matrix();

		kernel.set_full_kernel_matrix_from_full(data);
		DoubleMatrix km_fullfull=kernel.get_kernel_matrix();

		modshogun.exit_shogun();
	}
}
