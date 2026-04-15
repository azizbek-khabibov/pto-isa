#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <cmath>

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TFmodTest, MatchesScalarFmodForElementwiseInputs)
{
    using TileData = Tile<TileType::Vec, float, 2, 8, BLayout::RowMajor, 2, 4>;

    TileData dst;
    TileData src0;
    TileData src1;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src0, src1);

    const float lhs[2][4] = {{5.5f, -5.5f, 9.25f, -9.25f}, {8.0f, 7.0f, -7.0f, 3.5f}};
    const float rhs[2][4] = {{2.0f, 2.0f, 2.5f, 2.5f}, {3.0f, -3.0f, 3.0f, 1.25f}};

    for (int r = 0; r < src0.GetValidRow(); ++r) {
        for (int c = 0; c < src0.GetValidCol(); ++c) {
            SetValue(src0, r, c, lhs[r][c]);
            SetValue(src1, r, c, rhs[r][c]);
        }
    }

    TFMOD(dst, src0, src1);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            ExpectValueEquals(GetValue(dst, r, c), std::fmod(lhs[r][c], rhs[r][c]));
        }
    }
}

} // namespace
