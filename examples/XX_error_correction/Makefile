NVCC=nvcc
NVCCFLAGS=-arch=sm_80 -I../../include -I../../tools/util/include -I../common -std=c++17 -I./mateval/include -I./cutf/include --ptxas-options=-v -lcublas
NVCCFLAGS+=-DTC_COR

TARGET=ec.test


all:
	$(NVCC) ec_main.cu $(NVCCFLAGS) -o $(TARGET)

clean:
	rm -f *.test
