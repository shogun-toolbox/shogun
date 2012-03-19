CXXFLAGS=-O9 -g $(INCLUDES)
LINKFLAGS=-lshogun
TARGETS = sailfish
COMP = g++

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

%: %.cpp
	$(COMP) $(CXXFLAGS) $< $(LINKFLAGS) -o $@
