package shogun.examples;
import shogun.*;

public class VectorTest {
	static {
		System.loadLibrary("Features");
	}

	public static void main(String argv[]) {
		Features.init_shogun();
		Labels x = new Labels();
		
		doubleArray array = new doubleArray(4);		
		for (int i = 0; i < 4; i++) {
    		array.setitem(i, i);
		}

		sgDoubleVector label = new sgDoubleVector(array.cast(), 4);
		x.set_labels(label);
		sgDoubleVector z = x.get_labels();
		
		Features.exit_shogun();
	}
}
