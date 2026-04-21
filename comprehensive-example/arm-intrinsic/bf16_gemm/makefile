CC = clang

OBJS = bf16gemm.c
TARGET = bf16gemm
CFLAGS = -march=armv9.2-a+sme+bf16 --rtlib=compiler-rt
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) 128 128 128 0

clean:
	rm -f $(TARGET)