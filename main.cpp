#include <iostream>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/avutil.h>
}

static int global_frame_count = 0;

// Procesa los frames decodificados disponibles en el contexto
void process_decoded_frames(AVCodecContext* dec_ctx, AVFrame* frame) {
    int ret = 0;
    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            std::cerr << "Error decoding frame." << std::endl;
            exit(1);
        }

        global_frame_count++;
        bool is_key = frame->flags & AV_FRAME_FLAG_KEY;

        std::cout << "Frame " << global_frame_count
                  << " | Type: " << av_get_picture_type_char(frame->pict_type)
                  << " | Resolution: " << frame->width << "x" << frame->height
                  << " | PTS: " << frame->pts
                  << " | Keyframe: " << (is_key ? "YES" : "NO") << std::endl;

        av_frame_unref(frame); // Limpiar frame para reutilización
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // 1. Abrir archivo y leer header (Demuxing)
    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, argv[1], nullptr, nullptr) < 0) {
        std::cerr << "Could not open source file." << std::endl;
        return 1;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream info." << std::endl;
        return 1;
    }

    // 2. Localizar Stream de Video y Decoder
    int vid_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1,
                                      nullptr, 0);
    if (vid_idx < 0) {
        std::cerr << "Could not find video stream." << std::endl;
        return 1;
    }

    const AVCodec* decoder = avcodec_find_decoder(
        fmt_ctx->streams[vid_idx]->codecpar->codec_id);
    AVCodecContext* codec_ctx = avcodec_alloc_context3(decoder);
    
    avcodec_parameters_to_context(codec_ctx,
                                  fmt_ctx->streams[vid_idx]->codecpar);

    if (avcodec_open2(codec_ctx, decoder, nullptr) < 0) {
        std::cerr << "Could not open codec." << std::endl;
        return 1;
    }

    // 3. Allocación de memoria
    AVFrame* frame = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc(); // API Moderna: Heap allocation

    // 4. Loop de lectura y decodificación
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == vid_idx) {
            // Enviar paquete comprimido al decodificador
            if (avcodec_send_packet(codec_ctx, pkt) < 0) {
                std::cerr << "Error sending packet." << std::endl;
                break;
            }
            // Recibir frames raw (descomprimidos)
            process_decoded_frames(codec_ctx, frame);
        }
        av_packet_unref(pkt); // Importante: Resetear paquete
    }

    // 5. Flush del decoder (procesar frames pendientes en buffer interno)
    avcodec_send_packet(codec_ctx, nullptr);
    process_decoded_frames(codec_ctx, frame);

    // 6. Cleanup
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    return 0;
}