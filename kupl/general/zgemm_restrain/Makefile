all: clean benchmark_kupl benchmark_omp 
	@echo "Default target 'all' completed."

run_omp:
	@echo "====Running zgemm restrain test with omp parallel===="
	OMP_NUM_THREADS=1 taskset -c 304-341 ./benchmark_omp 65536 32768 64 38 0
	OMP_NUM_THREADS=1 taskset -c 304-341 ./benchmark_omp 65536 32768 128 38 0
	OMP_NUM_THREADS=1 taskset -c 304-341 ./benchmark_omp 65536 32768 256 38 0
	OMP_NUM_THREADS=1 taskset -c 304-341 ./benchmark_omp 196608 36864 64 38 0
	OMP_NUM_THREADS=1 taskset -c 304-341 ./benchmark_omp 196608 36864 128 38 0
	OMP_NUM_THREADS=1 taskset -c 304-341 ./benchmark_omp 196608 36864 256 38 0
benchmark_omp: 
	clang test_zgemm_restrain_omp.c -o benchmark_omp -fPIC -O3 -DUSE_MEM_ON_CHIP -lkblas -fopenmp -lmemkind

benchmark_kupl: 
	clang test_zgemm_restrain_kupl.c -o benchmark_kupl -fPIC -O3 -DUSE_MEM_ON_CHIP -lkblas -lkupl -lmemkind

run_kupl:
	@echo "====Running zgemm restrain test with kupl parallel===="
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 304-341 ./benchmark_kupl 65536 32768 64 38 0
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 304-341 ./benchmark_kupl 65536 32768 128 38 0
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 304-341 ./benchmark_kupl 65536 32768 256 38 0
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 304-341 ./benchmark_kupl 196608 36864 64 38 0
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 304-341 ./benchmark_kupl 196608 36864 128 38 0
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 304-341 ./benchmark_kupl 196608 36864 256 38 0
clean:
	rm -rf benchmark_kupl benchmark_omp 