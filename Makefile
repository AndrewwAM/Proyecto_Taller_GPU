run:
	g++ src/main.cpp src/VideoLoader.cpp -o main.out -lavcodec -lavformat -lavutil -lswscale
	./main.out test.mp4 out.mp4

