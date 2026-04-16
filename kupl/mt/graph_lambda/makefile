CC = clang++

OBJS = graph_lambda.cpp
TARGET = graph_lambda

LDFLAGS = -lkupl

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ -O3 $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	KUPL_SCHED_POLICY=mq KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-1 ./$(TARGET)

clean:
	rm -f $(TARGET)