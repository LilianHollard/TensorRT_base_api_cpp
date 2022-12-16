SRC= $(wildcard *.cpp)
HEADER= $(wildcard *.h)
OBJ= $(SRC:.cpp=.o)
CVLIBS= -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_video -lopencv_videoio -lnvonnxparser -lnvcaffe_parser -lnvinfer -lopencv_features2d -lopencv_imgcodecs -lopencv_objdetect
LDLIBS= -L /usr/local/cuda-10.2/lib64/ -L /usr/local/lib -L /usr/lib/aarch64-linux-gnu/
CFLAGS= -std=c++0x -I /usr/local/cuda-10.2/include/ -I /usr/local/include/opencv4/ -I /usr/src/tensorrt/samples/common/ -I /usr/include/aarch64-linux-gnu/


LDFLAGS= $(LDLIBS) -lm -lstdc++ $(CVLIBS) -lcuda -lcublas -lcurand -lcudart -opencv4

all: Out

Out: $(OBJ)
	g++ $^ $(LDFLAGS) -o Out

%.o: %.cpp $(HEADER)
	g++ -c $< $(CFLAGS)

clean:
	rm *.o Out
