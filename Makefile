NVCC = nvcc
CXX = g++

CUDA_FLAGS = -arch=sm_75
FFMPEG_LIBS = -lavcodec -lavformat -lavutil -lswscale

INPUT ?= test.mp4
TITLE ?= test

cpu:
	$(CXX) src/main.cpp src/VideoLoader.cpp -o cpu.out $(FFMPEG_LIBS)
	./cpu.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(TITLE)_cpu.mp4 -y

grayscale:
	$(NVCC) $(CUDA_FLAGS) src/grayscale.cu src/VideoLoader.cpp -o grayscale.out $(FFMPEG_LIBS)
	./grayscale.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 grayscale.mp4

ascii10:
	$(NVCC) $(CUDA_FLAGS) src/ascii.cu src/VideoLoader.cpp -o ascii.out $(FFMPEG_LIBS)
	./ascii.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(TITLE)_ascii10.mp4

ascii5:
	$(NVCC) $(CUDA_FLAGS) src/ascii5.cu src/VideoLoader.cpp -o ascii5.out $(FFMPEG_LIBS)
	./ascii5.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(TITLE)_ascii5.mp4

dither:
	$(NVCC) $(CUDA_FLAGS) src/dither.cu src/VideoLoader.cpp -o dither.out $(FFMPEG_LIBS)
	./dither.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(TITLE)_dither.mp4

posterization:
	$(NVCC) $(CUDA_FLAGS) src/posterization.cu src/VideoLoader.cpp -o posterization.out $(FFMPEG_LIBS)
	./posterization.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(TITLE)_posterization.mp4

chromatic:
	$(NVCC) $(CUDA_FLAGS) src/chromatic_aberration.cu src/VideoLoader.cpp -o chromatic_aberration.out $(FFMPEG_LIBS)
	./chromatic_aberration.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(TITLE)_chromatic.mp4

benchmark:
	$(NVCC) $(CUDA_FLAGS) src/benchmark.cu src/CpuFilters.cpp src/VideoLoader.cpp -o benchmark.out $(FFMPEG_LIBS)
	./benchmark.out $(INPUT)

clean:
	rm -f *.out $(TITLE)_*.mp4

.PHONY: cpu grayscale ascii10 dither posterization chromatic clean benchmark