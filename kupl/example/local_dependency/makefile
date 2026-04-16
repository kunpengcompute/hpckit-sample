CC = clang++

TARGET = mt_kupl_pf_version mt_kupl_graph_version

LDFLAGS = -lkupl

all: $(TARGET)

mt_kupl_pf_version: mt_kupl_pf_version.cpp
	$(CC) -o $@ $^ $(LDFLAGS)

mt_kupl_graph_version: mt_kupl_graph_version.cpp
	$(CC) -o $@ $^ $(LDFLAGS)

run:
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-3 ./mt_kupl_pf_version
	KUPL_EXECUTOR_BACKEND=pthread KUPL_SCHED_POLICY=mq taskset -c 0-3 ./mt_kupl_graph_version

run_pf:
	KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-3 ./mt_kupl_pf_version

run_graph:
	KUPL_EXECUTOR_BACKEND=pthread KUPL_SCHED_POLICY=mq taskset -c 0-3 ./mt_kupl_graph_version

clean:
	rm -f $(TARGET)