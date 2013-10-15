import org.shogun.*;
import org.jblas.*;
import java.io.*;
import java.util.*;

public class Load {
	public static DoubleMatrix load_numbers(String filename) {
		DoubleMatrix result = null;
		try {
			DoubleMatrix temp = DoubleMatrix.loadAsciiFile(filename);
			result = temp.transpose();
		} catch(java.io.IOException e) {
			System.out.println("Unable to create matrix from " + filename + ": " + e.getMessage());
			System.exit(-1);
		}
		return result;
	}
	public static DoubleMatrix load_labels(String filename) {
		DoubleMatrix result = null;
		try {
			DoubleMatrix temp = DoubleMatrix.loadAsciiFile(filename);
			result = temp.reshape(1, temp.getLength());
		} catch(java.io.IOException e) {
			System.out.println("Unable to create matrix from " + filename + ": " + e.getMessage());
			System.exit(-1);
		}
		return result;
	}
	public static String[] load_dna(String filename) {
		ArrayList<String> list = new ArrayList<String>();
		String[] result = null;
		try {
			FileInputStream fstream = new FileInputStream(filename);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader buffer = new BufferedReader(new InputStreamReader(in));
			String line;
			while((line = buffer.readLine()) != null) {
				list.add(line);
			}
			in.close();
			result = new String[list.size()];
			for (int i = 0; i < list.size(); i++) {
				result[i] = (String)list.get(i);
			}
		} catch(java.io.IOException e) {
			System.out.println("Unable to create matrix from " + filename + ": " + e.getMessage());
			System.exit(-1);
		}
		return result;
	}

	public static String[] load_cubes(String filename) {
		ArrayList<String> list = new ArrayList<String>();
		String[] result = null;
		try {
			FileInputStream fstream = new FileInputStream(filename);
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader buffer = new BufferedReader(new InputStreamReader(in));
			String line;
			while((line = buffer.readLine()) != null) {
				list.add(line);
			}
			in.close();
			result = new String[list.size()];
			for (int i = 0; i < list.size(); i++) {
				result[i] = (String)list.get(i);
			}
		} catch(java.io.IOException e) {
			System.out.println("Unable to create matrix from " + filename + ": " + e.getMessage());
			System.exit(-1);
		}
		return result;
	}
}
