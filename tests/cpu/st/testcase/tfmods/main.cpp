#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <cmath>

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TFmodsTest, MatchesScalarFmodForScalarDivisor)
{
    using TileData = Tile<TileType::Vec, float, 2, 8, BLayout::RowMajor, 2, 4>;

    TileData dst;
    TileData src;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src);

    const float values[2][4] = {{5.5f, -5.5f, 9.25f, -9.25f}, {8.0f, 7.0f, -7.0f, 3.5f}};
    constexpr float divisor = 2.5f;

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, values[r][c]);
        }
    }

    TFMODS(dst, src, divisor);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            ExpectValueEquals(GetValue(dst, r, c), std::fmod(values[r][c], divisor));
        }
    }
}

} // namespace
