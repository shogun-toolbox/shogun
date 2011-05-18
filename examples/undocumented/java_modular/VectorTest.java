import shogun.*;

public class VectorTest {
	static {
		System.loadLibrary("Features");
	}

	public static void main(String argv[]) {
		Features.init_shogun();
		Labels x = new Labels();

		double y[] = {1, 2, 3, 4};
		x.set_labels(y);
		double z[] = x.get_labels();
		for (int i = 0; i < 4; i ++) {
			System.out.println(z[i]);	
		}
		Features.exit_shogun();
	}
}
