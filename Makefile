NVCC = nvcc
CXX = g++

CUDA_FLAGS = -arch=sm_75
FFMPEG_LIBS = -lavcodec -lavformat -lavutil -lswscale

INPUT ?= test.mp4

cpu:
	$(CXX) src/main.cpp src/VideoLoader.cpp -o main_cpu.out $(FFMPEG_LIBS)
	./main_cpu.out $(INPUT) out_cpu.mp4
	ffmpeg -i out_cpu.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 out_cpu_audio.mp4 -y

grayscale_audio:
	$(NVCC) $(CUDA_FLAGS) src/main_grayscale.cu src/VideoLoader.cpp -o main_grayscale.out $(FFMPEG_LIBS)
	./main_grayscale.out $(INPUT) out_grayscale.mp4
	ffmpeg -i out_grayscale.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 out_grayscale_audio.mp4

ascii_audio:
	$(NVCC) $(CUDA_FLAGS) src/main_ascii.cu src/VideoLoader.cpp -o main_ascii.out $(FFMPEG_LIBS)
	./main_ascii.out $(INPUT) out_ascii.mp4
	ffmpeg -i out_ascii.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 out_ascii_audio.mp4

dither:
	$(NVCC) $(CUDA_FLAGS) src/main_dither.cu src/VideoLoader.cpp -o main_dither.out $(FFMPEG_LIBS)
	./main_dither.out $(INPUT) out_dither.mp4
	ffmpeg -i out_dither.mp4 -i $(INPUT) -c copy -map 0:v:0 -map 1:a:0 out_dither_audio.mp4

clean:
	rm -f main_cpu.out main_grayscale.out main_ascii.out out_*.mp4

.PHONY: cpu grayscale grayscale_audio ascii ascii_audio clean
