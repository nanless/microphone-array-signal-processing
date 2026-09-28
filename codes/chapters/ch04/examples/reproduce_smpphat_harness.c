/* Minimal caller for the separately obtained GPL-3.0 smpphat source.
 *
 * This file contains only experiment setup, public API calls, and JSON output.
 * It does not copy the upstream SRP or SMP-PHAT implementation.  The temporary
 * executable produced by reproduce_smpphat_reference.py is not committed.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <smpphat/signal.h>
#include <smpphat/system.h>

enum {
    CHANNELS = 4,
    PAIRS = 6,
    POINTS = 24,
    FRAME_SIZE = 64,
    INTERPOLATION_RATE = 4,
    SAMPLE_RATE = 16000
};

static const float SOUND_SPEED = 343.0f;

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void read_floats(const char *path, float *values, size_t count) {
    FILE *stream = fopen(path, "rb");
    if (stream == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    if (fread(values, sizeof(float), count, stream) != count) {
        fclose(stream);
        fail("binary input has the wrong length");
    }
    if (fgetc(stream) != EOF) {
        fclose(stream);
        fail("binary input has trailing bytes");
    }
    if (fclose(stream) != 0) {
        fail("cannot close binary input");
    }
}

static void print_float_array(const float *values, unsigned int count) {
    unsigned int index;
    putchar('[');
    for (index = 0; index < count; index++) {
        if (index != 0) {
            putchar(',');
        }
        printf("%.9g", values[index]);
    }
    putchar(']');
}

static void print_uint_array(const unsigned int *values, unsigned int count) {
    unsigned int index;
    putchar('[');
    for (index = 0; index < count; index++) {
        if (index != 0) {
            putchar(',');
        }
        printf("%u", values[index]);
    }
    putchar(']');
}

static void srp_scores(const srp_obj *object, float *scores) {
    unsigned int pair_index;
    unsigned int point_index;
    for (point_index = 0; point_index < object->pdoas_count; point_index++) {
        float score = 0.0f;
        for (pair_index = 0; pair_index < object->pairs_count; pair_index++) {
            unsigned int lookup = object->lookups[
                point_index * object->pairs_count + pair_index];
            score += object->cross_correlation[lookup];
        }
        scores[point_index] = score;
    }
}

static void smp_scores(const smp_obj *object, float *scores) {
    unsigned int group_index;
    unsigned int point_index;
    for (point_index = 0; point_index < object->pdoas_count; point_index++) {
        float score = 0.0f;
        for (group_index = 0; group_index < object->groups_count; group_index++) {
            unsigned int lookup = object->lookups[
                point_index * object->groups_count + group_index];
            score += object->cross_correlation[lookup];
        }
        scores[point_index] = score;
    }
}

static unsigned int argmax(const float *values, unsigned int count) {
    unsigned int best = 0;
    unsigned int index;
    for (index = 1; index < count; index++) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return best;
}

int main(int argc, char **argv) {
    float geometry[CHANNELS * 3];
    float directions[POINTS * 3];
    float spectra[PAIRS * (FRAME_SIZE / 2 + 1) * 2];
    float srp_values[POINTS];
    float smp_values[POINTS];
    float srp_api_xyz[3];
    float smp_api_xyz[3];
    float srp_api_score;
    float smp_api_score;
    float polarities[PAIRS];
    unsigned int groups[PAIRS];
    unsigned int srp_ranges[PAIRS];
    unsigned int smp_ranges[PAIRS];
    unsigned int srp_lookups[POINTS * PAIRS];
    unsigned int smp_lookups[POINTS * PAIRS];
    unsigned int pairs_count;
    unsigned int groups_count;
    unsigned int pair_index;
    unsigned int srp_peak;
    unsigned int smp_peak;
    int srp_status;
    int smp_status;

    mics_obj microphones;
    pdoas_obj potential_directions;
    covs_obj *covariances;
    ldoas_obj *srp_result;
    ldoas_obj *smp_result;
    srp_obj *srp;
    smp_obj *smp;

    if (argc != 4) {
        fail("usage: harness GEOMETRY_F32 DIRECTIONS_F32 PHAT_F32");
    }
    read_floats(argv[1], geometry, CHANNELS * 3);
    read_floats(argv[2], directions, POINTS * 3);
    read_floats(argv[3], spectra, PAIRS * (FRAME_SIZE / 2 + 1) * 2);

    microphones.channels_count = CHANNELS;
    microphones.xyzs = geometry;
    potential_directions.points_count = POINTS;
    potential_directions.xyzs = directions;
    covariances = covs_construct(CHANNELS, FRAME_SIZE);
    srp_result = ldoas_construct(1);
    smp_result = ldoas_construct(1);
    if (covariances == NULL || srp_result == NULL || smp_result == NULL) {
        fail("upstream allocation failed");
    }
    for (pair_index = 0; pair_index < PAIRS; pair_index++) {
        memcpy(covariances->samples[pair_index],
               &spectra[pair_index * (FRAME_SIZE / 2 + 1) * 2],
               sizeof(float) * (FRAME_SIZE / 2 + 1) * 2);
    }

    srp = srp_construct(FRAME_SIZE, INTERPOLATION_RATE, SOUND_SPEED,
                        SAMPLE_RATE, &microphones, &potential_directions);
    if (srp == NULL) {
        fail("upstream constructor failed");
    }
    srp_status = srp_call(srp, covariances, srp_result);
    if (srp_status != 0) {
        srp_destroy(srp);
        covs_destroy(covariances);
        ldoas_destroy(srp_result);
        ldoas_destroy(smp_result);
        fail("upstream srp_call returned an error");
    }
    srp_scores(srp, srp_values);
    pairs_count = srp->pairs_count;
    memcpy(srp_ranges, srp->ranges, sizeof(srp_ranges));
    memcpy(srp_lookups, srp->lookups, sizeof(srp_lookups));
    memcpy(srp_api_xyz, srp_result->xyzs, sizeof(srp_api_xyz));
    srp_api_score = srp_result->es[0];
    srp_destroy(srp);

    /* Each upstream destroy function calls fftwf_cleanup().  Do not keep the
     * other algorithm's FFTW plans alive across that global cleanup. */
    smp = smp_construct(FRAME_SIZE, INTERPOLATION_RATE, SOUND_SPEED,
                        SAMPLE_RATE, &microphones, &potential_directions);
    if (smp == NULL) {
        fail("upstream constructor failed");
    }
    smp_status = smp_call(smp, covariances, smp_result);
    if (smp_status != 0) {
        smp_destroy(smp);
        covs_destroy(covariances);
        ldoas_destroy(srp_result);
        ldoas_destroy(smp_result);
        fail("upstream smp_call returned an error");
    }
    smp_scores(smp, smp_values);
    groups_count = smp->groups_count;
    memcpy(groups, smp->groups, sizeof(groups));
    memcpy(polarities, smp->polarities, sizeof(polarities));
    memcpy(smp_ranges, smp->ranges, sizeof(unsigned int) * groups_count);
    memcpy(smp_lookups, smp->lookups,
           sizeof(unsigned int) * POINTS * groups_count);
    memcpy(smp_api_xyz, smp_result->xyzs, sizeof(smp_api_xyz));
    smp_api_score = smp_result->es[0];
    srp_peak = argmax(srp_values, POINTS);
    smp_peak = argmax(smp_values, POINTS);
    smp_destroy(smp);

    printf("{\"fftw_runtime_version\":\"%s\",", fftwf_version);
    printf("\"srp_status\":%d,\"smp_status\":%d,", srp_status, smp_status);
    printf("\"pairs_count\":%u,\"groups_count\":%u,", pairs_count,
           groups_count);
    printf("\"groups\":");
    print_uint_array(groups, pairs_count);
    printf(",\"polarities\":");
    print_float_array(polarities, pairs_count);
    printf(",\"srp_ranges\":");
    print_uint_array(srp_ranges, pairs_count);
    printf(",\"smp_ranges\":");
    print_uint_array(smp_ranges, groups_count);
    printf(",\"srp_lookups\":");
    print_uint_array(srp_lookups, POINTS * pairs_count);
    printf(",\"smp_lookups\":");
    print_uint_array(smp_lookups, POINTS * groups_count);
    printf(",\"srp_scores\":");
    print_float_array(srp_values, POINTS);
    printf(",\"smp_scores\":");
    print_float_array(smp_values, POINTS);
    printf(",\"srp_peak_index\":%u,\"smp_peak_index\":%u,",
           srp_peak, smp_peak);
    printf("\"srp_api_xyz\":");
    print_float_array(srp_api_xyz, 3);
    printf(",\"smp_api_xyz\":");
    print_float_array(smp_api_xyz, 3);
    printf(",\"srp_api_score\":%.9g,\"smp_api_score\":%.9g}\n",
           srp_api_score, smp_api_score);

    covs_destroy(covariances);
    ldoas_destroy(srp_result);
    ldoas_destroy(smp_result);
    return EXIT_SUCCESS;
}
