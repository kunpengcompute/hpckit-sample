CC = clang

OBJS = fp64gemv.c
TARGET = fp64gemv
CFLAGS = -march=armv8.2-a+sve --rtlib=compiler-rt
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) 128 128 0

clean:
	rm -f $(TARGET)