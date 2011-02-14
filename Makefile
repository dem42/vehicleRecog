CFLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`

all: vehicleRecognition.o
	g++ vehicleRecognition.o -o vehi $(LIBS)

.cpp.o:
	g++ $(CFLAGS) -c $<
