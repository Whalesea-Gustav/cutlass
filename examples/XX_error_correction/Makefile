NVCC=nvcc
NVCCFLAGS=-arch=sm_80 -I../../include -I../../tools/util/include -I../common -std=c++17 -I./mateval/include -I./cutf/include --ptxas-options=-v -lcublas

USE_TC_COR=YES

ifeq (${USE_TC_COR}, YES)
	NVCCFLAGS+=-DTC_COR
endif

TARGET=ampere_tf32_tensorop_gemm.test


all:
	$(NVCC) ampere_tf32_tensorop_gemm.cu $(NVCCFLAGS) -o ampere_tf32_tensorop_gemm.test

clean:
	rm -f *.test
