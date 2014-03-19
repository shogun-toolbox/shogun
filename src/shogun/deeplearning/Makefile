Example: Example.o NeuralNets.o DataAdapter.o
	g++ -o Example Example.o DataAdapter.o

%.o: ./%.cpp
	g++ -c $< -o $@

clean:
	rm -f *.o
	rm Example

