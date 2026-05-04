/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <pto/common/fifo.hpp>
#include <thread>
#include <vector>
#include <barrier>
#include "test_common.h"

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

namespace {
using T = float;

class TPUSHTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

// Pipe and Communication
using MainPipe = TPipe<0, Direction::DIR_BOTH, 8192, 4, 4, false>;

static void main_incore_0_aic( float* v1,  float* v2,  float* v3,  void* v4, int32_t v5) {
  MainPipe v15(v4, 0, 0);
  using AccTile = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>;
  using SmallMat = Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>;
  using Matrix = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>;
  using TileLeft = Tile<TileType::Left, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>;
  using RightTile = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>;
  for (size_t i = 0; i < 4; i ++) {
    std::cout<<"Cube : "<<"; i="<<i<<std::endl;
    Matrix v17(64, 64);
    TASSIGN(v17, 32768);
    Matrix v18(64, 64);
    __cbuf__ void* v19 = v17.data();
    uint64_t v20 = reinterpret_cast<uint64_t>(v19);
    TASSIGN(v18, v20);
    using Shape = pto::Shape<1, 1, 1, 64, 64>;
    using Stride = pto::Stride<32768, 32768, 32768, 512, 1>;
    GlobalTensor<float, Shape, Stride, pto::Layout::ND> globalTensor(v3 + (((v5 * 4) + i) * 64));
    TLOAD(v18, globalTensor); 
    SmallMat v24(32, 64);
    TPOP<MainPipe, SmallMat, TileSplitAxis::TILE_LEFT_RIGHT>(v15, v24);
    std::cout<<"Cube Pop: "<<"; i="<<i<<std::endl;
    TileLeft v25(32, 64);
    TASSIGN(v25, 0);
    TileLeft v26(32, 64);
    __ca__ void* v27 = v25.data();
    uint64_t v28 = reinterpret_cast<uint64_t>(v27);
    TASSIGN(v26, v28);
    TMOV(v26, v24);
    TFREE<MainPipe, TileSplitAxis::TILE_LEFT_RIGHT>(v15);
    RightTile v29(64, 64);
    TASSIGN(v29, 0);
    RightTile v30(64, 64);
    __cb__ void* v31 = v29.data();
    uint64_t v32 = reinterpret_cast<uint64_t>(v31);
    TASSIGN(v30, v32);
    TMOV(v30, v18);
    AccTile v33(32, 64);
    TASSIGN(v33, 0);
    AccTile v34(32, 64);
    __cc__ void* v35 = v33.data();
    uint64_t v36 = reinterpret_cast<uint64_t>(v35);
    TASSIGN(v34, v36);
    TMATMUL(v34, v26, v30);
    TPUSH<MainPipe, AccTile, TileSplitAxis::TILE_LEFT_RIGHT>(v15, v34);
    std::cout<<"Cube Push: "<<"; i="<<i<<std::endl;
}

  return;
}

static void main_incore_0_aiv( float* v1,  float* v2,  float* v3,  void* v4, int32_t v5) {
  int64_t subblock_id = get_subblockid();
  MainPipe v19(v4, 0, 0);
  using VecTile = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>;
  VecTile v20(32, 32);
  TASSIGN(v20, 32768);
  VecTile v21(32, 32);
  __ubuf__ void* v22 = v20.data();
  uint64_t v23 = reinterpret_cast<uint64_t>(v22);
  TASSIGN(v21, v23);
  int32_t v24 =  subblock_id *  32;
  using Shape = pto::Shape<1, 1, 1, 32, 32>;
  using Stride = pto::Stride<2048, 2048, 2048, 64, 1>;
  GlobalTensor<float, Shape, Stride, pto::Layout::ND> v27(v2 + v24);
    
  TLOAD(v21, v27);
  
  for (size_t i = 0; i < 4; i ++) {
    std::cout<<"Vec : "<<subblock_id<<"; i="<<i<<std::endl;
    VecTile v29(32, 32);
    TASSIGN(v29, 36864);
    VecTile v30(32, 32);
    __ubuf__ void* v31 = v29.data();
    uint64_t v32 = reinterpret_cast<uint64_t>(v31);
    TASSIGN(v30, v32);
    using Stride2 = pto::Stride<16384, 16384, 16384, 512, 1>;
    GlobalTensor<float, Shape, Stride2, pto::Layout::ND> v35(v1 + ((((v5 * 4) +  i) *  64) + v24));
    
    TLOAD(v30, v35);
    VecTile v36(32, 32);
    TASSIGN(v36, 40960);
    VecTile v37(32, 32);
    __ubuf__ void* v38 = v36.data();
    uint64_t v39 = reinterpret_cast<uint64_t>(v38);
    TASSIGN(v37, v39);
    
    TADDS(v37, v21, 1.0f);
    
    
    TPUSH<MainPipe, VecTile, TileSplitAxis::TILE_LEFT_RIGHT>(v19, v37);
    std::cout<<"Vec Push: "<<subblock_id<<"; i="<<i<<std::endl;
    VecTile v40(32, 32);
    TPOP<MainPipe, VecTile, TileSplitAxis::TILE_LEFT_RIGHT>(v19, v40);
    std::cout<<"Vec Pop: "<<subblock_id<<"; i="<<i<<std::endl;
    VecTile v41(32, 32);
    TASSIGN(v41, 45056);
    VecTile v42(32, 32);
    __ubuf__ void* v43 = v41.data();
    uint64_t v44 = reinterpret_cast<uint64_t>(v43);
    TASSIGN(v42, v44);
    
    TADD(v42, v30, v40);
    
    TFREE<MainPipe, TileSplitAxis::TILE_LEFT_RIGHT>(v19);
    
    TSTORE(v35, v42);
    
  }

  return;
}

void* g_shared_storage_ptr = nullptr;
// A simple function matching GetPipeSharedStateInjectedHookFn signature
extern "C" void* GlobalPipeHook(uint64_t key, size_t size) {
    // We'll use a global pointer to store the allocated memory
    return g_shared_storage_ptr;
}



inline void LaunchTPut(T *out, T *A, T *B, T *C) {
    std::cout<<"Start"<<std::endl;
    // 1. Allocate and zero the shared synchronization state
    size_t required_size = sizeof(MainPipe::SharedState);
    void* raw_mem = malloc(required_size);
    // g_shared_storage_ptr = calloc(1, required_size); 
    g_shared_storage_ptr = new (raw_mem) MainPipe::SharedState();

    T* pipe_mem;
    aclrtMalloc((void**)&pipe_mem, 2*65536, ACL_MEM_MALLOC_HUGE_FIRST);
    
    std::barrier sync_point(3); 

    // 2. Register the hook for shared storage ONLY
    // We pass nullptr for subblock_id hook so it falls back to the thread_local context
    pto::cpu_sim::register_hooks(nullptr, (void*)GlobalPipeHook);

    auto aiv_func = [&](int32_t id) {
        // This sets the thread_local ExecutionContext
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 2);

        sync_point.arrive_and_wait(); 
        main_incore_0_aiv(C, A, B, pipe_mem, 0);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait(); 
        main_incore_0_aic(C, A, B, pipe_mem, 0);
    };

    std::thread v0(aiv_func, 0);
    std::thread v1(aiv_func, 1);
    std::thread c0(aic_func);

    v0.join();
    v1.join();
    c0.join();

    // 3. Cleanup
    // pto::cpu_sim::register_hooks(nullptr, nullptr);
    // aclrtFree(pipe_mem);
    // static_cast<MainPipe::SharedState*>(g_shared_storage_ptr)->~SharedState();
    // free(g_shared_storage_ptr);
}

void test_tpush()
{   
     size_t ARow = 32, ACol = 64, BRow = 64, BCol = 512, CRow = 32, CCol = 512;
     size_t ASize = ARow * ACol * sizeof(T);
     size_t BSize = BRow * BCol * sizeof(T);
     size_t CSize = CRow * CCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcAHost, *srcBHost, *srcCHost;
    T *dstDevice, *srcADevice, *srcBDevice, *srcCDevice;

    aclrtMallocHost((void **)(&dstHost), CSize);
    aclrtMallocHost((void **)(&srcAHost), ASize);
    aclrtMallocHost((void **)(&srcBHost), BSize);
    aclrtMallocHost((void **)(&srcCHost), CSize);

    aclrtMalloc((void **)&dstDevice, CSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, ASize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcBDevice, BSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcCDevice, CSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/a.bin", ASize, srcAHost, ASize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/b.bin", BSize, srcBHost, BSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/c.bin", CSize, srcCHost, CSize));

    aclrtMemcpy(srcADevice, ASize, srcAHost, ASize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, BSize, srcBHost, BSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcCDevice, CSize, srcCHost, CSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPut(dstDevice, srcADevice, srcBDevice, srcCDevice);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, CSize, dstDevice, CSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", srcCDevice, CSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(srcCDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(srcCHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    size_t elem_count = CSize / sizeof(T);

    std::vector<T> golden(elem_count);
    std::vector<T> devFinal(elem_count);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", CSize, golden.data(), CSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", CSize, devFinal.data(), CSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TPUSHTest, case_1)
{
    test_tpush();
}

}
