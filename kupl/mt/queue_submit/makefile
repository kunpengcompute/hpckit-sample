CC = clang++

OBJS = queue_submit.cpp
TARGET = queue_submit

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	KUPL_SCHED_POLICY=static_mq KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-19 ./$(TARGET)

clean:
	rm -f $(TARGET)