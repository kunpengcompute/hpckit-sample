CC = clang

OBJS = int8gemm.c
TARGET = int8gemm
CFLAGS = -march=armv9.2-a+sme --rtlib=compiler-rt
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) 128 128 128 0

clean:
	rm -f $(TARGET)