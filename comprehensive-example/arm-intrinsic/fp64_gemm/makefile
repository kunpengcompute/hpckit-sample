CC = clang

OBJS = fp64gemm.c
TARGET = fp64gemm
CFLAGS = -march=armv9.2-a+sme+sme-f64f64 --rtlib=compiler-rt
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) 128 128 128 0

clean:
	rm -f $(TARGET)