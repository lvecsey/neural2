
CC=gcc

# MEMCHECK=-fsanitize=address,undefined -fno-omit-frame-pointer

CFLAGS=-O3 -Wall -g -pg -I$(HOME)/src/version-control/git/glfw/deps $(MEMCHECK)

LIBS=-lm

all : training_data training_data.xdr neural2 neural2-lookup show-training_data

run : NEURONS=100

run : ITERATIONS=40000

run : INPUT_VALUE=0.55

run : training_data.xdr neural2 neural2-lookup
	@./neural2 $(NEURONS) $(ITERATIONS)
	@./neural2-lookup "neural_net.xdr" 0.23 0.77 0.35

training_data : sigmoid.o set_cluster.o training_data.o
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS) $(LIBS)

training_data.xdr : NUM_ROWS=1000

training_data.xdr : training_data
	@./$^ $(NUM_ROWS) > $@.new
	mv $@.new $@

neural2 : norm.o set_cluster.o sigmoid.o sigmoid_deriv.o forward.o neural2.o
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS) $(LIBS)

neural2-lookup : LIBS=-lm

neural2-lookup : norm.o sigmoid.o forward.o neural2-lookup.o
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS) $(LIBS)

show-training_data : show-training_data.o
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS) $(LIBS)

mse.png : mse.dat mse.plot
	@gnuplot mse.plot

test-values : neural2-lookup 
	@./neural2-lookup neural_net.xdr 0.34 0.76 0.21
	@./neural2-lookup neural_net.xdr 0.33 0.78 0.23
	@./neural2-lookup neural_net.xdr 0.57 -0.37 0.79
	@./neural2-lookup neural_net.xdr 0.55 -0.39 0.77

clean :
	-rm *.o neural2 neural2-lookup
