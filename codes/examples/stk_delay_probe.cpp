// Original interface probe. STK is compiled from its separate, locked checkout.
#include "DelayL.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    constexpr int count = 512;
    std::vector<double> input(count), whole(count), blocked(count), reset(count);
    for (int n = 0; n < count; ++n)
        input[n] = .18 * std::sin(2*std::acos(-1.)*500*n/16000.)
                 + .18 * std::sin(2*std::acos(-1.)*6000*n/16000.);
    stk::DelayL a(.5, 8), b(.5, 8), c(.5, 8), zero(0., 8), one(1., 8);
    double analytic_error = 0, zero_error = 0, one_error = 0;
    for (int n = 0; n < count; ++n) {
        whole[n] = a.tick(input[n]);
        const double previous = n ? input[n-1] : 0.;
        analytic_error = std::max(analytic_error, std::abs(whole[n]-.5*(input[n]+previous)));
        zero_error = std::max(zero_error, std::abs(zero.tick(input[n])-input[n]));
        one_error = std::max(one_error, std::abs(one.tick(input[n])-previous));
    }
    for (int start = 0; start < count; start += 37) {
        c.clear(); // Intentional negative control: erase the history at each block.
        for (int n = start; n < std::min(start+37, count); ++n) {
            blocked[n] = b.tick(input[n]);
            reset[n] = c.tick(input[n]);
        }
    }
    double block_error = 0, reset_error = 0;
    for (int n = 0; n < count; ++n) {
        block_error = std::max(block_error, std::abs(blocked[n]-whole[n]));
        reset_error = std::max(reset_error, std::abs(reset[n]-whole[n]));
    }
    std::cout << std::setprecision(17)
        << "{\"analytic_max_abs_error\":" << analytic_error
        << ",\"zero_delay_max_abs_error\":" << zero_error
        << ",\"one_delay_max_abs_error\":" << one_error
        << ",\"chunk37_max_abs_error\":" << block_error
        << ",\"reset_chunk37_max_abs_difference\":" << reset_error << "}\n";
}
