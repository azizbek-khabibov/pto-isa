#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TDequantTest, AppliesPerRowBroadcastedScaleAndOffset)
{
    using DstTile = Tile<TileType::Vec, float, 2, 8, BLayout::RowMajor, 2, 4>;
    using SrcTile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 2, 4>;
    using ParaTile = Tile<TileType::Vec, float, 2, 8, BLayout::RowMajor, 2, 1>;

    DstTile dst;
    SrcTile src;
    ParaTile scale;
    ParaTile offset;
    std::size_t addr = 0;
    AssignTileStorage(addr, dst, src, scale, offset);

    const int srcValues[2][4] = {{3, 5, 7, 9}, {10, 12, 14, 16}};
    const float scaleValues[2] = {0.5f, 2.0f};
    const float offsetValues[2] = {1.0f, 1.5f};

    for (int r = 0; r < src.GetValidRow(); ++r) {
        SetValue(scale, r, 0, scaleValues[r]);
        SetValue(offset, r, 0, offsetValues[r]);
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, srcValues[r][c]);
        }
    }

    TDEQUANT(dst, src, scale, offset);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const float expected = (static_cast<float>(srcValues[r][c]) - offsetValues[r]) * scaleValues[r];
            ExpectValueEquals(GetValue(dst, r, c), expected);
        }
    }
}

} // namespace
