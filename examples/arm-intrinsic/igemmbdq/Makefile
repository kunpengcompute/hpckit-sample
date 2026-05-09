CC = clang++

OBJS = int8_gemm_bf16_dequant.cpp
TARGET = int8_gemm_bf16_dequant
CFLAGS = -march=armv9.2-a+sme+sme-f64f64 --rtlib=compiler-rt -unwindlib=libunwind
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) 128 7168 128

clean:
	rm -f $(TARGET)