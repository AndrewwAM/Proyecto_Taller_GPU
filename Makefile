NVCC = nvcc
CXX = g++

CUDA_FLAGS = -arch=sm_75
FFMPEG_LIBS = -lavcodec -lavformat -lavutil -lswscale

INPUT ?= test.mp4

cpu:
	$(CXX) src/main.cpp src/VideoLoader.cpp -o cpu.out $(FFMPEG_LIBS)
	./cpu.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(INPUT)_cpu.mp4 -y

grayscale:
	$(NVCC) $(CUDA_FLAGS) src/grayscale.cu src/VideoLoader.cpp -o $(INPUT)_grayscale.out $(FFMPEG_LIBS)
	./grayscale.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 grayscale.mp4

ascii:
	$(NVCC) $(CUDA_FLAGS) src/ascii.cu src/VideoLoader.cpp -o ascii.out $(FFMPEG_LIBS)
	./ascii.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(INPUT)_ascii.mp4

dither:
	$(NVCC) $(CUDA_FLAGS) src/dither.cu src/VideoLoader.cpp -o dither.out $(FFMPEG_LIBS)
	./dither.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(INPUT)_dither.mp4

posterization:
	$(NVCC) $(CUDA_FLAGS) src/posterization.cu src/VideoLoader.cpp -o posterization.out $(FFMPEG_LIBS)
	./posterization.out $(INPUT) tmp.mp4
	ffmpeg -i tmp.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 $(INPUT)_posterization.mp4

clean:
	rm -f *.out $(INPUT)_*.mp4

.PHONY: cpu grayscale grayscale_audio ascii ascii_audio clean
