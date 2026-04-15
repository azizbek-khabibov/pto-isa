#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TSubViewTest, AliasesRowMajorSourceStorageAtOffset)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 8>;
    using DstTile = Tile<TileType::Vec, float, 4, 8, BLayout::RowMajor, 3, 6>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    FillLinear(src, 1.0f);
    TSUBVIEW(dst, src, 1, 2);

    ASSERT_EQ(dst.data(), src.data() + SrcTile::RowStride + 2 * SrcTile::ColStride);
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            ExpectValueEquals(GetValue(dst, r, c), GetValue(src, r + 1, c + 2));
        }
    }

    SetValue(dst, 2, 5, 99.0f);
    ExpectValueEquals(GetValue(src, 3, 7), 99.0f);
}

TEST(TSubViewTest, AliasesColMajorSourceStorageAtOffset)
{
    using SrcTile = Tile<TileType::Vec, float, 8, 8, BLayout::ColMajor>;
    using DstTile = Tile<TileType::Vec, float, 8, 8, BLayout::ColMajor, 2, 3>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    FillLinear(src, 10.0f);
    TSUBVIEW(dst, src, 1, 4);

    ASSERT_EQ(dst.data(), src.data() + SrcTile::RowStride + 4 * SrcTile::ColStride);
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            ExpectValueEquals(GetValue(dst, r, c), GetValue(src, r + 1, c + 4));
        }
    }
}

} // namespace
