CC = clang++

OBJS = mt_gemm_fp64.cpp
TARGET = mt_gemm_fp64

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ -O3 $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	KUPL_EXECUTOR_BACKEND=pthread KUPL_SCHED_POLICY=mq taskset -c 0-3 ./$(TARGET)

clean:
	rm -f $(TARGET)