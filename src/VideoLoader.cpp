// VideoLoader.cpp
#include "VideoLoader.hpp"
#include <iostream>
#include <stdexcept>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

VideoLoader::VideoLoader(const std::string& filename) {
    // 1. Init Decoder
    if (avformat_open_input(&fmt_ctx, filename.c_str(), nullptr, nullptr) < 0) {
        throw std::runtime_error("Failed to open video file: " + filename);
    }
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        throw std::runtime_error("Failed to retrieve stream info");
    }
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = static_cast<int>(i);
            break;
        }
    }
    if (video_stream_index == -1) throw std::runtime_error("No video stream");

    AVCodecParameters* codec_params = fmt_ctx->streams[video_stream_index]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) throw std::runtime_error("Unsupported codec");

    codec_ctx = avcodec_alloc_context3(codec);
    if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) throw std::runtime_error("Param copy failed");
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) throw std::runtime_error("Open codec failed");

    frame = av_frame_alloc();
    packet = av_packet_alloc();
}

VideoLoader::~VideoLoader() {
    close_writer(); // Ensure writer is closed

    // Cleanup Decoder
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
}

double VideoLoader::get_fps() const {
    if (video_stream_index != -1) {
        AVRational r = fmt_ctx->streams[video_stream_index]->avg_frame_rate;
        if (r.den > 0) return static_cast<double>(r.num) / r.den;
    }
    return 30.0; // Fallback
}

bool VideoLoader::load_next_frame(uint8_t** buffer_out, int* width, int* height) {
    while (av_read_frame(fmt_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) == 0) {
                if (avcodec_receive_frame(codec_ctx, frame) == 0) {
				// Decoder Color Conversion (YUV -> RGBA)
				if (!sws_ctx || frame->width != m_last_width || frame->height != m_last_height) {
					sws_freeContext(sws_ctx);
					sws_ctx = sws_getContext(frame->width, frame->height, codec_ctx->pix_fmt,
											 frame->width, frame->height, AV_PIX_FMT_RGBA,
											 SWS_BILINEAR, nullptr, nullptr, nullptr);
					m_rgba_buffer.resize(frame->width * frame->height * 4);
					m_last_width = frame->width;
					m_last_height = frame->height;
				}

				uint8_t* dest[4] = { m_rgba_buffer.data(), nullptr, nullptr, nullptr };
				int linesize[4] = { frame->width * 4, 0, 0, 0 };
				sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, dest, linesize);

				*width = frame->width;
				*height = frame->height;
				*buffer_out = m_rgba_buffer.data();

				av_packet_unref(packet);
				return true;
                }
            }
        }
        av_packet_unref(packet);
    }
    return false;
}

void VideoLoader::init_writer(const std::string& out_filename, int width, int height, double fps) {
    if (m_writer_initialized) throw std::runtime_error("Writer already initialized");

    avformat_alloc_output_context2(&out_fmt_ctx, nullptr, nullptr, out_filename.c_str());
    if (!out_fmt_ctx) throw std::runtime_error("Could not deduce output format");

    const AVCodec* encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!encoder) throw std::runtime_error("H.264 Encoder not found");

    out_stream = avformat_new_stream(out_fmt_ctx, nullptr);
    out_codec_ctx = avcodec_alloc_context3(encoder);

    AVRational fps_rational = av_d2q(fps, 100000);
    out_codec_ctx->time_base = av_inv_q(fps_rational);
    out_stream->time_base = out_codec_ctx->time_base;

    out_codec_ctx->width = width;
    out_codec_ctx->height = height;
    out_codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    
    // Configuración CRF (Calidad constante)
    if (out_codec_ctx->codec_id == AV_CODEC_ID_H264) {
        av_opt_set(out_codec_ctx->priv_data, "preset", "fast", 0);
        av_opt_set(out_codec_ctx->priv_data, "crf", "23", 0);
    } else {
        out_codec_ctx->bit_rate = 4000000;
    }

    if (out_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        out_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avcodec_open2(out_codec_ctx, encoder, nullptr) < 0) {
        throw std::runtime_error("Could not open encoder");
    }

    if (avcodec_parameters_from_context(out_stream->codecpar, out_codec_ctx) < 0) {
        throw std::runtime_error("Failed to copy stream params");
    }

    if (!(out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&out_fmt_ctx->pb, out_filename.c_str(), AVIO_FLAG_WRITE) < 0) {
            throw std::runtime_error("Could not open output file IO");
        }
    }

    if (avformat_write_header(out_fmt_ctx, nullptr) < 0) {
        throw std::runtime_error("Error writing header");
    }

    out_frame_yuv = av_frame_alloc();
    out_frame_yuv->format = out_codec_ctx->pix_fmt;
    out_frame_yuv->width = width;
    out_frame_yuv->height = height;
    av_frame_get_buffer(out_frame_yuv, 32);

    out_packet = av_packet_alloc();

    sws_ctx_encode = sws_getContext(width, height, AV_PIX_FMT_RGBA,
                                    width, height, AV_PIX_FMT_YUV420P,
                                    SWS_BILINEAR, nullptr, nullptr, nullptr);

    m_pts_counter = 0;
    m_writer_initialized = true;
    std::cout << "Writer initialized (Fixed Timebase): " << out_filename << std::endl;
}

void VideoLoader::write_frame(const uint8_t* rgba_data) {
    if (!m_writer_initialized) throw std::runtime_error("Writer not initialized");

    // 1. Make frame writable
    if (av_frame_make_writable(out_frame_yuv) < 0) {
        throw std::runtime_error("Frame not writable");
    }

    // 2. Convert RGBA (Input) -> YUV420P (Encoder)
    const uint8_t* src_slices[1] = { rgba_data };
    int src_stride[1] = { out_codec_ctx->width * 4 }; // 4 bytes per pixel

    sws_scale(sws_ctx_encode, src_slices, src_stride, 0, out_codec_ctx->height,
              out_frame_yuv->data, out_frame_yuv->linesize);

    // 3. Set Timestamp
    out_frame_yuv->pts = m_pts_counter++;

    // 4. Send to Encoder
    if (avcodec_send_frame(out_codec_ctx, out_frame_yuv) < 0) {
        throw std::runtime_error("Error sending frame to encoder");
    }

    // 5. Write Packets
    while (avcodec_receive_packet(out_codec_ctx, out_packet) >= 0) {
        av_packet_rescale_ts(out_packet, out_codec_ctx->time_base, out_stream->time_base);
        out_packet->stream_index = out_stream->index;
        av_interleaved_write_frame(out_fmt_ctx, out_packet);
        av_packet_unref(out_packet);
    }
}

void VideoLoader::flush_encoder() {
    if (!out_codec_ctx) return;

    // Send flush command
    avcodec_send_frame(out_codec_ctx, nullptr);

    while (avcodec_receive_packet(out_codec_ctx, out_packet) >= 0) {
        av_packet_rescale_ts(out_packet, out_codec_ctx->time_base, out_stream->time_base);
        out_packet->stream_index = out_stream->index;
        av_interleaved_write_frame(out_fmt_ctx, out_packet);
        av_packet_unref(out_packet);
    }
}

void VideoLoader::close_writer() {
    if (m_writer_initialized) {
        flush_encoder();
        av_write_trailer(out_fmt_ctx);

        if (!(out_fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&out_fmt_ctx->pb);
        }

        avcodec_free_context(&out_codec_ctx);
        av_frame_free(&out_frame_yuv);
        av_packet_free(&out_packet);
        sws_freeContext(sws_ctx_encode);
        avformat_free_context(out_fmt_ctx);

        m_writer_initialized = false;
        std::cout << "Writer closed and file saved." << std::endl;
    }
}
