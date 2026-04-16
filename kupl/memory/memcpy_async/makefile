CC = clang++

OBJS = memcpy_async.cpp
TARGET = memcpy_async

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	KUPL_EXECUTOR_BACKEND=pthread KUPL_SCHED_POLICY=mq taskset -c 0-1 ./$(TARGET)

clean:
	rm -f $(TARGET)