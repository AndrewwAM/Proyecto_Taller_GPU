// VideoLoader.hpp
#pragma once

#include <string>
#include <vector>
#include <cstdint>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h> 
}

class VideoLoader {
public:
    explicit VideoLoader(const std::string& filename);
    ~VideoLoader();

    VideoLoader(const VideoLoader&) = delete;
    VideoLoader& operator=(const VideoLoader&) = delete;

    // Decoder API
    bool load_next_frame(uint8_t** buffer_out, int* width, int* height);
    double get_fps() const; // Helper to sync output FPS

    // Encoder API (New)
    void init_writer(const std::string& out_filename, int width, int height, double fps);
    void write_frame(const uint8_t* rgba_data);
    void close_writer();

private:
    // Helper to flush encoder
    void flush_encoder();

    // --- DECODER MEMBERS ---
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    int video_stream_index = -1;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    SwsContext* sws_ctx = nullptr;
    std::vector<uint8_t> m_rgba_buffer;
    int m_last_width = -1;
    int m_last_height = -1;

    // --- ENCODER MEMBERS ---
    AVFormatContext* out_fmt_ctx = nullptr;
    AVCodecContext* out_codec_ctx = nullptr;
    AVStream* out_stream = nullptr;
    AVFrame* out_frame_yuv = nullptr; // Encoder needs YUV, not RGBA
    AVPacket* out_packet = nullptr;
    SwsContext* sws_ctx_encode = nullptr;
    
    int64_t m_pts_counter = 0;
    bool m_writer_initialized = false;
};
