BENCHMARKS = \
	3d7pt \
	apop \
	game-of-life \
	heat-1d \
	heat-2d \
	heat-3d 

.PHONY: all clean

all:
	@for bench in $(BENCHMARKS); do \
		$(MAKE) -C $$bench; \
	done
	
clean:
	@for bench in $(BENCHMARKS); do \
		$(MAKE) -C $$bench clean; \
	done