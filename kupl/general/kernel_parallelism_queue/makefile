CC = clang++

TARGET = kernel_parallelism_queue
OBJS = $(TARGET).cpp

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-15 ./$(TARGET)

clean:
	rm -f $(TARGET)