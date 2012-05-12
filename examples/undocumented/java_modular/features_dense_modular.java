import org.shogun.*;
import org.jblas.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class features_dense_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String[] argv) {
		modshogun.init_shogun_with_defaults();

		ArrayList<DoubleMatrix> result = new ArrayList<DoubleMatrix>(4);

		DoubleMatrix inputRealMatrix = Load.load_numbers("../data/fm_train_real.dat");
		RealFeatures realFeatures = new RealFeatures(inputRealMatrix);
		DoubleMatrix outputRealMatrix = realFeatures.get_feature_matrix();

		result.add(inputRealMatrix);
		result.add(outputRealMatrix);

		DoubleMatrix inputByteMatrix = Load.load_numbers("../data/fm_train_byte.dat");
		ByteFeatures byteFeatures = new ByteFeatures(inputByteMatrix);
		DoubleMatrix outputByteMatrix = byteFeatures.get_feature_matrix();

		result.add(inputByteMatrix);
		result.add(outputByteMatrix);

		System.out.println(result);

		modshogun.exit_shogun();
	}
}
