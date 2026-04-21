CC = clang

OBJS = int8gemv.c
TARGET = int8gemv
CFLAGS = -march=armv9.2-a+sme+sve+sve2 --rtlib=compiler-rt
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) 128 128 128 0

clean:
	rm -f $(TARGET)