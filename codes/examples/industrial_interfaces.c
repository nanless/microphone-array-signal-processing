/* Original interface probes; links the separately obtained upstream libraries.
 * This is synthetic protocol verification, not acoustic or standards certification.
 * Per-call SRC traces go to stderr; the compact measurement report goes to stdout.
 */
#include <ebur128.h>
#include <sndfile.h>
#include <soxr.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "check failed at %d: %s\n", __LINE__, #c); exit(2); } } while (0)
#define INPUT_FRAMES 48000
#define OUTPUT_CAPACITY 17000
#define PI 3.14159265358979323846

typedef struct {
    double samples[OUTPUT_CAPACITY * 2];
    size_t frames, before_flush, calls, flush_calls, partial_calls;
    double delay_before_flush, delay_after_flush;
} src_result;

static void src_probe(const double *input, size_t chunk, int reset,
                      const char *name, src_result *result) {
    soxr_error_t error;
    soxr_io_spec_t io = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);
    soxr_quality_spec_t quality = soxr_quality_spec(SOXR_HQ, 0);
    soxr_runtime_spec_t runtime = soxr_runtime_spec(1);
    soxr_t state = soxr_create(48000, 16000, 2, &error, &io, &quality, &runtime);
    size_t consumed = 0;
    int did_reset = 0;
    CHECK(state && !error);
    memset(result, 0, sizeof(*result));
    while (consumed < INPUT_FRAMES) {
        size_t end = consumed + chunk;
        if (end > INPUT_FRAMES) end = INPUT_FRAMES;
        if (reset && !did_reset && end > INPUT_FRAMES / 2) end = INPUT_FRAMES / 2;
        if (reset && !did_reset && consumed == INPUT_FRAMES / 2) {
            fprintf(stderr, "%s,clear,%zu,0,0,%.17g\n", name, consumed, soxr_delay(state));
            CHECK(!soxr_clear(state));
            did_reset = 1;
            continue;
        }
        while (consumed < end) {
            double output[113 * 2];
            size_t idone = 0, odone = 0, offered = end - consumed;
            CHECK(!soxr_process(state, input + 2 * consumed, offered,
                                &idone, output, 113, &odone));
            CHECK(idone <= offered && odone <= 113 && (idone || odone));
            CHECK(result->frames + odone <= OUTPUT_CAPACITY);
            fprintf(stderr, "%s,input,%zu,%zu,%zu,%.17g\n", name, consumed, idone, odone, soxr_delay(state));
            memcpy(result->samples + 2 * result->frames, output, 2 * odone * sizeof(double));
            result->frames += odone;
            result->calls++;
            result->partial_calls += idone < offered;
            consumed += idone;
        }
    }
    result->before_flush = result->frames;
    result->delay_before_flush = soxr_delay(state);
    for (;;) {
        double output[113 * 2];
        size_t odone = 0;
        CHECK(!soxr_process(state, NULL, 0, NULL, output, 113, &odone));
        CHECK(odone <= 113 && result->frames + odone <= OUTPUT_CAPACITY);
        fprintf(stderr, "%s,flush,%zu,0,%zu,%.17g\n", name, consumed, odone, soxr_delay(state));
        memcpy(result->samples + 2 * result->frames, output, 2 * odone * sizeof(double));
        result->frames += odone;
        result->flush_calls++;
        if (!odone) break;
        CHECK(result->flush_calls < OUTPUT_CAPACITY);
    }
    result->delay_after_flush = soxr_delay(state);
    soxr_delete(state);
}

static double overlap_difference(const src_result *a, const src_result *b) {
    size_t count = a->frames < b->frames ? a->frames : b->frames;
    double maximum = 0;
    for (size_t i = 0; i < 2 * count; ++i) {
        double difference = fabs(a->samples[i] - b->samples[i]);
        CHECK(isfinite(difference));
        if (difference > maximum) maximum = difference;
    }
    return maximum;
}

static void print_src_result(const src_result *r) {
    printf("{\"output_frames\":%zu,\"before_flush_frames\":%zu,"
           "\"flush_frames\":%zu,\"input_calls\":%zu,\"flush_calls\":%zu,"
           "\"partial_consumption_calls\":%zu,\"delay_before_flush_output_samples\":%.17g,"
           "\"delay_after_flush_output_samples\":%.17g}",
           r->frames, r->before_flush, r->frames-r->before_flush, r->calls,
           r->flush_calls, r->partial_calls, r->delay_before_flush, r->delay_after_flush);
}

typedef struct { double lufs, sample_peak, true_peak; } loudness_result;

static loudness_result loudness_probe(double amplitude, size_t chunk) {
    const size_t count = 480000;
    double *signal = malloc(count * sizeof(double));
    ebur128_state *state = ebur128_init(1, 48000, EBUR128_MODE_I | EBUR128_MODE_TRUE_PEAK);
    loudness_result result;
    CHECK(signal && state);
    CHECK(ebur128_set_channel(state, 0, EBUR128_CENTER) == EBUR128_SUCCESS);
    for (size_t i = 0; i < count; ++i) signal[i] = amplitude * sin(2 * PI * 1000 * i / 48000);
    for (size_t start = 0; start < count; start += chunk) {
        size_t frames = count-start < chunk ? count-start : chunk;
        CHECK(ebur128_add_frames_double(state, signal+start, frames) == EBUR128_SUCCESS);
    }
    CHECK(ebur128_loudness_global(state, &result.lufs) == EBUR128_SUCCESS);
    CHECK(ebur128_sample_peak(state, 0, &result.sample_peak) == EBUR128_SUCCESS);
    CHECK(ebur128_true_peak(state, 0, &result.true_peak) == EBUR128_SUCCESS);
    ebur128_destroy(&state);
    free(signal);
    return result;
}

static void pcm_probe(const char *path) {
    const short expected[14] = {-32768,0, 0,32767, 16384,-16384, 1,-1,
                                12345,-23456, 0,1000, 32767,-32768};
    const sf_count_t frame_counts[4] = {3,3,1,0}, item_counts[4] = {6,6,2,0};
    SF_INFO info = {0};
    info.samplerate = 16000; info.channels = 2; info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE *file = sf_open(path, SFM_WRITE, &info);
    CHECK(file && sf_writef_short(file, expected, 7) == 7);
    CHECK(sf_close(file) == 0);
    for (int items = 0; items < 2; ++items) {
        size_t position = 0;
        memset(&info, 0, sizeof(info));
        file = sf_open(path, SFM_READ, &info);
        CHECK(file && info.frames == 7 && info.channels == 2 && info.samplerate == 16000);
        CHECK((info.format & SF_FORMAT_SUBMASK) == SF_FORMAT_PCM_16);
        for (int call = 0; call < 4; ++call) {
            short buffer[6];
            sf_count_t count = items ? sf_read_short(file, buffer, 6) : sf_readf_short(file, buffer, 3);
            CHECK(count == (items ? item_counts[call] : frame_counts[call]));
            size_t valid_items = (size_t)count * (items ? 1 : 2);
            for (size_t i = 0; i < valid_items; ++i) CHECK(buffer[i] == expected[position+i]);
            position += valid_items;
        }
        CHECK(position == 14 && sf_error(file) == SF_ERR_NO_ERROR);
        CHECK(sf_close(file) == 0);
    }
    file = sf_open(path, SFM_READ, &info);
    CHECK(file);
    sf_command(file, SFC_SET_NORM_DOUBLE, NULL, SF_TRUE);
    double decoded[14];
    CHECK(sf_readf_double(file, decoded, 7) == 7);
    double max_error = 0;
    for (int i = 0; i < 14; ++i) {
        double error = fabs(decoded[i] - expected[i] / 32768.0);
        if (error > max_error) max_error = error;
    }
    CHECK(max_error == 0 && sf_close(file) == 0);
    printf("\"pcm16\":{\"sample_rate_hz\":16000,\"channels\":2,\"frames\":7,"
           "\"frame_read_counts\":[3,3,1,0],\"item_read_counts\":[6,6,2,0],"
           "\"float_max_abs_error\":%.17g,\"pcm_interleaved\":[", max_error);
    for (int i = 0; i < 14; ++i) printf("%s%d", i ? "," : "", expected[i]);
    printf("]}");
}

int main(int argc, char **argv) {
    CHECK(argc == 2);
    double *input = calloc(INPUT_FRAMES * 2, sizeof(double));
    src_result *results = calloc(4, sizeof(src_result));
    CHECK(input && results);
    input[12000 * 2] = 0.5;
    for (size_t i = 0; i < INPUT_FRAMES; ++i) input[i*2+1] = 0.1 * sin(2 * PI * 1000 * i / 48000);
    fprintf(stderr, "case,phase,input_position_frames,idone_frames,odone_frames,delay_output_samples\n");
    src_probe(input, INPUT_FRAMES, 0, "whole", &results[0]);
    src_probe(input, 127, 0, "chunk127", &results[1]);
    src_probe(input, 509, 0, "chunk509", &results[2]);
    src_probe(input, 127, 1, "clear24000", &results[3]);
    for (int i = 0; i < 3; ++i) CHECK(results[i].frames == 16000);
    double error127 = overlap_difference(&results[0], &results[1]);
    double error509 = overlap_difference(&results[0], &results[2]);
    double reset_error = overlap_difference(&results[0], &results[3]);
    CHECK(error127 < 1e-12 && error509 < 1e-12);
    CHECK(results[3].frames != 16000 || reset_error > 1e-6);
    size_t peak_index = 0;
    for (size_t i = 1; i < results[0].frames; ++i)
        if (fabs(results[0].samples[2*i]) > fabs(results[0].samples[2*peak_index])) peak_index = i;
    CHECK(peak_index == 4000);
    double sine_error = 0;
    for (size_t i = 320; i < 15680; ++i) {
        double error = fabs(results[0].samples[2*i+1] - 0.1*sin(2*PI*1000*i/16000));
        if (error > sine_error) sine_error = error;
    }
    CHECK(sine_error < 1e-5);
    loudness_result full = loudness_probe(0.1, 480000);
    loudness_result half = loudness_probe(0.05, 480000);
    loudness_result chunked = loudness_probe(0.1, 127);
    loudness_result silence = loudness_probe(0.0, 127);
    CHECK(isfinite(full.lufs) && isfinite(half.lufs) && isfinite(chunked.lufs));
    CHECK(fabs((half.lufs-full.lufs) - 20*log10(0.5)) < 1e-10);
    CHECK(fabs(chunked.lufs-full.lufs) < 1e-10);
    CHECK(fabs(half.sample_peak/full.sample_peak-0.5) < 1e-12);
    CHECK(fabs(half.true_peak/full.true_peak-0.5) < 1e-12);
    CHECK(isinf(silence.lufs) && silence.lufs < 0 && silence.sample_peak == 0 && silence.true_peak == 0);
    int major, minor, patch;
    ebur128_get_version(&major, &minor, &patch);
    printf("{\"library_versions\":{\"soxr\":\"%s\",\"ebur128\":\"%d.%d.%d\",\"sndfile\":\"%s\"},",
           soxr_version(), major, minor, patch, sf_version_string());
    printf("\"src\":{\"input_frames\":48000,\"channels\":2,\"input_rate_hz\":48000,\"output_rate_hz\":16000,"
           "\"duration_seconds\":1,\"left_impulse_amplitude\":0.5,\"left_impulse_input_frame\":12000,"
           "\"right_sine_amplitude\":0.1,\"right_sine_frequency_hz\":1000,"
           "\"alignment\":\"No fitted shift or gain; compare native sample indices after full flush\","
           "\"reset_comparison_scope\":\"Clear discards pending output; unequal lengths and equal-index differences are not aligned quality scores\","
           "\"io\":\"FLOAT64 interleaved\",\"quality\":\"HQ, flags=0\",\"threads\":1,\"output_capacity_per_call\":113,"
           "\"whole\":"); print_src_result(&results[0]);
    printf(",\"chunk127\":"); print_src_result(&results[1]);
    printf(",\"chunk509\":"); print_src_result(&results[2]);
    printf(",\"clear_at_input_frame_24000_without_flush\":"); print_src_result(&results[3]);
    printf(",\"chunk127_max_abs_error\":%.17g,\"chunk509_max_abs_error\":%.17g,"
           "\"reset_equal_index_overlap_max_abs_difference\":%.17g,\"impulse_output_peak_frame\":%zu,"
           "\"sine_max_abs_error_frames_320_to_15680\":%.17g},",error127,error509,reset_error,peak_index,sine_error);
    printf("\"loudness\":{\"sample_rate_hz\":48000,\"frames\":480000,\"frequency_hz\":1000,"
           "\"channel_role\":\"CENTER\",\"mode\":\"I | TRUE_PEAK\",\"amplitudes\":[0.1,0.05],"
           "\"full_lufs\":%.17g,\"half_lufs\":%.17g,\"half_minus_full_lu\":%.17g,"
           "\"chunk127_minus_whole_lu\":%.17g,\"full_sample_peak\":%.17g,\"half_sample_peak\":%.17g,"
           "\"full_true_peak\":%.17g,\"half_true_peak\":%.17g,"
           "\"silence_lufs\":null,\"silence_reason\":\"negative_infinity_no_gated_energy\","
           "\"silence_sample_peak\":0,\"silence_true_peak\":0},",
           full.lufs,half.lufs,half.lufs-full.lufs,chunked.lufs-full.lufs,
           full.sample_peak,half.sample_peak,full.true_peak,half.true_peak);
    pcm_probe(argv[1]);
    puts("}");
    free(input); free(results);
    return 0;
}
