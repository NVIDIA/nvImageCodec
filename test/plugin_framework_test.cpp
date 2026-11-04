/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../src/plugin_framework.h"
#include "../src/codec_registry.h"
#include "../src/parsers/codec_file_extensions.h"
#include "mock_codec_registry.h"
#include "mock_directory_scaner.h"
#include "mock_environment.h"
#include "mock_executor.h"
#include "mock_library_loader.h"
#include "mock_logger.h"

namespace nvimgcodec { namespace test {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Const;
using ::testing::Eq;
using ::testing::Return;
using ::testing::ReturnPointee;
using ::testing::Throw;

namespace {
nvimgcodecStatus_t g_too_long_status = NVIMGCODEC_STATUS_SUCCESS;
nvimgcodecStatus_t g_max_ok_status = NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
nvimgcodecStatus_t g_too_long_codec_status = NVIMGCODEC_STATUS_SUCCESS;
nvimgcodecStatus_t g_max_ok_codec_status = NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
std::string g_too_long_ext;
std::string g_max_ok_ext;
std::string g_too_long_codec;
std::string g_max_ok_codec;
} // namespace

uint32_t testExtModuleGetVersion()
{
    return 1;
}

nvimgcodecStatus_t testExtModuleCreate(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
{
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t testExtModuleDestroy(nvimgcodecExtension_t extension)
{
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t testExtModuleEntry(nvimgcodecExtensionDesc_t* ext_desc)
{
    ext_desc->id = "discovered_test_extension";
    ext_desc->version = 1;
    ext_desc->ext_api_version = NVIMGCODEC_VER;
    ext_desc->create = &testExtModuleCreate;
    ext_desc->destroy = &testExtModuleDestroy;
    return NVIMGCODEC_STATUS_SUCCESS;
}

int unload_failure_destroy_calls = 0;

nvimgcodecStatus_t unloadFailureExtCreate(void*, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t*)
{
    *extension = reinterpret_cast<nvimgcodecExtension_t>(1);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t unloadFailureExtDestroy(nvimgcodecExtension_t)
{
    ++unload_failure_destroy_calls;
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t unloadFailureExtEntry(nvimgcodecExtensionDesc_t* ext_desc)
{
    ext_desc->id = "unload_failure_extension";
    ext_desc->version = 1;
    ext_desc->ext_api_version = NVIMGCODEC_VER;
    ext_desc->create = &unloadFailureExtCreate;
    ext_desc->destroy = &unloadFailureExtDestroy;
    return NVIMGCODEC_STATUS_SUCCESS;
}

TEST(PluginFrameworkTest, test_ext_module_discovery)
{
    MockCodecRegistry codec_registry;

    std::unique_ptr<MockEnvironment> env = std::make_unique<MockEnvironment>();
    EXPECT_CALL(*env.get(), getVariable("NVIMGCODEC_EXTENSIONS_PATH")).Times(1).WillOnce(Return(""));

    std::unique_ptr<MockDirectoryScaner> directory_scaner = std::make_unique<MockDirectoryScaner>();
    EXPECT_CALL(*directory_scaner.get(), start(_)).Times(1);
    EXPECT_CALL(*directory_scaner.get(), hasMore())
        .Times(7)
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(true))
        .WillOnce(Return(false));

#if defined(_WIN32) || defined(_WIN64)
    const std::string shared_library_ext = ".dll";
    const std::string static_library_ext = ".lib";
#else
    const std::string shared_library_ext = ".so";
    const std::string static_library_ext = ".a";
#endif
    const fs::path module0 = std::string("nvjpeg_ext_0") + shared_library_ext;
    const fs::path module1 = std::string("nvjpeg2k_ext_0") + shared_library_ext;
    const fs::path disabled_extension_module = std::string("~libtiff_ext_0") + shared_library_ext;
    const fs::path module2 = std::string("opencv_ext_0") + shared_library_ext;
    const fs::path ignored_non_module = std::string("opencv_ext") + static_library_ext;
    const fs::path ignored_backup_module = std::string("opencv_ext_0") + shared_library_ext + ".backup";

    EXPECT_CALL(*directory_scaner.get(), next())
        .Times(6)
        .WillOnce(Return(module0))
        .WillOnce(Return(module1))
        .WillOnce(Return(disabled_extension_module))
        .WillOnce(Return(module2))
        .WillOnce(Return(ignored_non_module))
        .WillOnce(Return(ignored_backup_module));
    EXPECT_CALL(*directory_scaner.get(), symlinkStatus(_))
        .Times(6)
        .WillOnce(Return(fs::file_status(fs::file_type::regular)))
        .WillOnce(Return(fs::file_status(fs::file_type::regular)))
        .WillOnce(Return(fs::file_status(fs::file_type::regular)))
        .WillOnce(Return(fs::file_status(fs::file_type::regular)))
        .WillOnce(Return(fs::file_status(fs::file_type::symlink)))
        .WillOnce(Return(fs::file_status(fs::file_type::regular)));
    EXPECT_CALL(*directory_scaner.get(), exists(_)).WillRepeatedly(Return(true));

    std::unique_ptr<MockLibraryLoader> library_loader = std::make_unique<MockLibraryLoader>();
    ILibraryLoader::LibraryHandle handle0 = reinterpret_cast<ILibraryLoader::LibraryHandle> (0x1234);
    ILibraryLoader::LibraryHandle handle1 = reinterpret_cast<ILibraryLoader::LibraryHandle>(0x4567);
    ILibraryLoader::LibraryHandle handle2 = reinterpret_cast<ILibraryLoader::LibraryHandle>(0x7654);
    EXPECT_CALL(*library_loader.get(), loadLibrary(_))
        .Times(3)
        .WillOnce(Return(handle0))
        .WillOnce(Return(handle1))
        .WillOnce(Return(handle2));
    EXPECT_CALL(*library_loader.get(), getFuncAddress(_, Eq("nvimgcodecExtensionModuleEntry")))
        .WillRepeatedly(Return((ILibraryLoader::LibraryHandle)&testExtModuleEntry));
    EXPECT_CALL(*library_loader.get(), unloadLibrary(_)).Times(3);

    MockLogger logger;
    PluginFramework framework(&logger, &codec_registry, std::move(env), std::move(directory_scaner), std::move(library_loader), "");
    framework.discoverAndLoadExtModules();

    int extension_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework.getExtensions(nullptr, &extension_count));
    ASSERT_EQ(1, extension_count);
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework.getExtensions(&extension, &extension_count));
    const nvimgcodecExtensionDesc_t* extension_desc = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework.getExtensionDesc(extension, &extension_desc));
    ASSERT_STREQ("discovered_test_extension", extension_desc->id);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework.unregisterExtension(extension));
}

TEST(PluginFrameworkTest, unload_failure_does_not_destroy_plugin_twice)
{
    MockCodecRegistry codec_registry;
    auto env = std::make_unique<MockEnvironment>();
    auto directory_scanner = std::make_unique<MockDirectoryScaner>();
    auto library_loader = std::make_unique<MockLibraryLoader>();
    auto* loader = library_loader.get();
    ILibraryLoader::LibraryHandle module_handle = reinterpret_cast<ILibraryLoader::LibraryHandle>(0x1234);

    EXPECT_CALL(*loader, loadLibrary(Eq("throwing_extension.so"))).WillOnce(Return(module_handle));
    EXPECT_CALL(*loader, getFuncAddress(module_handle, Eq("nvimgcodecExtensionModuleEntry")))
        .WillOnce(Return(reinterpret_cast<void*>(&unloadFailureExtEntry)));
    EXPECT_CALL(*loader, unloadLibrary(module_handle)).Times(3).WillRepeatedly(Throw(std::runtime_error("unload failed")));

    unload_failure_destroy_calls = 0;
    {
        MockLogger logger;
        PluginFramework framework(
            &logger, &codec_registry, std::move(env), std::move(directory_scanner), std::move(library_loader), "unused");
        framework.loadExtModule("throwing_extension.so");

        int count = 0;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework.getExtensions(nullptr, &count));
        ASSERT_EQ(1, count);
        nvimgcodecExtension_t extension = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework.getExtensions(&extension, &count));
        EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR, framework.unregisterExtension(extension));
        EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR, framework.unregisterExtension(extension));
        EXPECT_EQ(1, unload_failure_destroy_calls);
    }
    EXPECT_EQ(1, unload_failure_destroy_calls);
}

TEST(PluginFrameworkTest, test_ext_module_without_entry_func_is_unloaded)
{
    MockCodecRegistry codec_registry;

    std::unique_ptr<MockEnvironment> env = std::make_unique<MockEnvironment>();
    std::unique_ptr<MockDirectoryScaner> directory_scaner = std::make_unique<MockDirectoryScaner>();

    std::unique_ptr<MockLibraryLoader> library_loader = std::make_unique<MockLibraryLoader>();
    ILibraryLoader::LibraryHandle handle = reinterpret_cast<ILibraryLoader::LibraryHandle>(0x1234);
    EXPECT_CALL(*library_loader.get(), loadLibrary(Eq("not_an_extension.dll"))).Times(1).WillOnce(Return(handle));
    EXPECT_CALL(*library_loader.get(), getFuncAddress(handle, Eq("nvimgcodecExtensionModuleEntry"))).Times(1).WillOnce(Return(nullptr));
    EXPECT_CALL(*library_loader.get(), unloadLibrary(handle)).Times(1);

    MockLogger logger;
    PluginFramework framework(
        &logger, &codec_registry, std::move(env), std::move(directory_scaner), std::move(library_loader), "unused");
    framework.loadExtModule("not_an_extension.dll");
}

class PluginFrameworkExtensionsPathTest : public ::testing::Test
{
  public:
    PluginFrameworkExtensionsPathTest() {}

    void SetUp() override
    {
        env_ = std::make_unique<MockEnvironment>();
        directory_scaner_ = std::make_unique<MockDirectoryScaner>();
        EXPECT_CALL(*directory_scaner_.get(), exists(_)).WillRepeatedly(Return(true));
        EXPECT_CALL(*directory_scaner_.get(), hasMore()).WillRepeatedly(Return(false));
        library_loader_ = std::make_unique<MockLibraryLoader>();
        executor_ = std::make_unique<MockExecutor>();
    }

    void TearDown() override {}

    void TestExtensionsOnePath(const std::string& env_test_path, const std::string& soft_test_path, const std::string& expected_test_path)
    {
        EXPECT_CALL(*env_.get(), getVariable("NVIMGCODEC_EXTENSIONS_PATH")).WillRepeatedly(Return(env_test_path));
        EXPECT_CALL(*directory_scaner_.get(), start(fs::path(expected_test_path))).Times(1);

        PluginFramework framework(
            &logger_, &codec_registry_, std::move(env_), std::move(directory_scaner_), std::move(library_loader_), soft_test_path);
        framework.discoverAndLoadExtModules();
    }

    void TestExtensionsMultiplePaths(
        const std::string& env_test_path, const std::string& soft_test_path, const std::vector<std::string>& expected_paths)
    {
        EXPECT_CALL(*env_.get(), getVariable("NVIMGCODEC_EXTENSIONS_PATH")).WillRepeatedly(Return(env_test_path));
        for (auto p : expected_paths) {
            EXPECT_CALL(*directory_scaner_.get(), start(fs::path(p))).Times(1);
        }

        PluginFramework framework(
            &logger_, &codec_registry_, std::move(env_), std::move(directory_scaner_), std::move(library_loader_), soft_test_path);
        framework.discoverAndLoadExtModules();
    }
    MockLogger logger_;
    MockCodecRegistry codec_registry_;
    std::unique_ptr<MockEnvironment> env_;
    std::unique_ptr<MockDirectoryScaner> directory_scaner_;
    std::unique_ptr<MockLibraryLoader> library_loader_;
    std::unique_ptr<MockExecutor> executor_;
};

TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_empty_env_and_soft_return_default)
{
    std::string env_test_path{""};
    std::string soft_test_path{""};
    std::vector<std::string> expected_test_paths;
    std::stringstream ss(GetDefaultExtensionsPath());
    std::string current_path;
    while (std::getline(ss, current_path, GetPathSeparator())) {
        expected_test_paths.push_back(current_path);
    }
    TestExtensionsMultiplePaths(env_test_path, soft_test_path, expected_test_paths);
}

TEST(PluginFrameworkDefaultExtensionsPathTest, returns_extensions_under_so_dir)
{
    EXPECT_EQ(BuildDefaultExtensionsPathFromSoDir("/usr/lib64/libnvimgcodec/12"),
        std::string("/usr/lib64/libnvimgcodec/12/extensions"));
    EXPECT_EQ(BuildDefaultExtensionsPathFromSoDir("/home/u/miniconda/envs/codec/lib/libnvimgcodec/12"),
        std::string("/home/u/miniconda/envs/codec/lib/libnvimgcodec/12/extensions"));
    EXPECT_EQ(BuildDefaultExtensionsPathFromSoDir("/home/u/.local/lib/python3.10/site-packages/nvidia/nvimgcodec"),
        std::string("/home/u/.local/lib/python3.10/site-packages/nvidia/nvimgcodec/extensions"));
    EXPECT_EQ(BuildDefaultExtensionsPathFromSoDir("/usr/local/lib64/libnvimgcodec/12"),
        std::string("/usr/local/lib64/libnvimgcodec/12/extensions"));
    EXPECT_EQ(BuildDefaultExtensionsPathFromSoDir("C:/Program Files/NVIDIA nvImageCodec/bin/12"),
        std::string("C:/Program Files/NVIDIA nvImageCodec/bin/12/extensions"));
}

TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_filled_env_and_empty_soft_return_env)
{
    std::string env_test_path{"/usr/env_test_path"};
    std::string soft_test_path{""};
    std::string expected_test_path{env_test_path};
    TestExtensionsOnePath(env_test_path, soft_test_path, expected_test_path);
}

TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_filled_env_and_soft_return_soft)
{
    std::string env_test_path{"/usr/env_test_path"};
    std::string soft_test_path{"/usr/soft_test_path"};
    std::string expected_test_path{soft_test_path};

    TestExtensionsOnePath(env_test_path, soft_test_path, expected_test_path);
}

TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_env_with_multiple_paths)
{
    std::vector<std::string> expected_test_paths{"/usr/env_test_path1", "/usr/env_test_path2", "/usr/env_test_path3"};
    std::stringstream ss;
    ss << expected_test_paths[0] << GetPathSeparator() << expected_test_paths[1] << GetPathSeparator() << expected_test_paths[2];
    std::string env_test_path{ss.str()};

    std::string soft_test_path{""};
    std::string expected_test_path{soft_test_path};

    TestExtensionsMultiplePaths(env_test_path, soft_test_path, expected_test_paths);
}
TEST_F(PluginFrameworkExtensionsPathTest, test_extension_path_for_soft_with_multiple_paths)
{
    std::vector<std::string> expected_test_paths{"/usr/soft_test_path1", "/usr/soft_test_path2", "/usr/soft_test_path3"};
    std::stringstream ss;
    ss << expected_test_paths[0] << GetPathSeparator() << expected_test_paths[1] << GetPathSeparator() << expected_test_paths[2];
    std::string env_test_path{"/usr/env_test_path"};

    std::string soft_test_path{ss.str()};
    std::string expected_test_path{soft_test_path};

    TestExtensionsMultiplePaths(env_test_path, soft_test_path, expected_test_paths);
}

class PluginFrameworkExtensionsVersionTest : public ::testing::Test
{
  public:
    PluginFrameworkExtensionsVersionTest() {}

    void SetUp() override
    {
        env_ = std::make_unique<MockEnvironment>();
        EXPECT_CALL(*env_.get(), getVariable("NVIMGCODEC_EXTENSIONS_PATH")).WillRepeatedly(Return(""));
        directory_scaner_ = std::make_unique<MockDirectoryScaner>();
        EXPECT_CALL(*directory_scaner_.get(), exists(_)).WillRepeatedly(Return(true));
        EXPECT_CALL(*directory_scaner_.get(), hasMore()).WillRepeatedly(Return(false));

        library_loader_ = std::make_unique<MockLibraryLoader>();
        framework_ = std::make_unique<PluginFramework>(
            &logger_, &codec_registry_, std::move(env_), std::move(directory_scaner_), std::move(library_loader_), "");
    }

    void TearDown() override {}

    static nvimgcodecStatus_t ExtCreate(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t ExtDestroy(nvimgcodecExtension_t extension) { return NVIMGCODEC_STATUS_SUCCESS; }

    MockLogger logger_;
    MockCodecRegistry codec_registry_;
    std::unique_ptr<MockEnvironment> env_;
    std::unique_ptr<MockDirectoryScaner> directory_scaner_;
    std::unique_ptr<MockLibraryLoader> library_loader_;
    std::unique_ptr<MockExecutor> executor_;
    std::unique_ptr<PluginFramework> framework_;
};

TEST_F(PluginFrameworkExtensionsVersionTest, test_when_there_is_already_ext_with_the_same_ver_register_returns_invalid)
{
    nvimgcodecExtension_t ext1;
    nvimgcodecExtensionDesc_t ext1_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr, "test_ext", 1000, NVIMGCODEC_VER,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->unregisterExtension(ext1));
}

TEST_F(PluginFrameworkExtensionsVersionTest, test_when_there_is_already_ext_with_the_newer_ver_register_returns_invalid)
{
    nvimgcodecExtension_t ext1;
    nvimgcodecExtensionDesc_t ext1_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr, "test_ext", 1000, NVIMGCODEC_VER,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};
    nvimgcodecExtension_t ext2;

    nvimgcodecExtensionDesc_t ext2_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr, "test_ext", 0100, NVIMGCODEC_VER,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, framework_->registerExtension(&ext2, &ext2_desc));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->unregisterExtension(ext1));
}

TEST_F(PluginFrameworkExtensionsVersionTest, test_when_there_is_already_ext_with_the_older_ver_register_returns_success)
{
    nvimgcodecExtension_t ext1;
    nvimgcodecExtensionDesc_t ext1_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr, "test_ext", 1000, NVIMGCODEC_VER,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};
    nvimgcodecExtension_t ext2;
    nvimgcodecExtensionDesc_t ext2_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr, "test_ext", 2000, NVIMGCODEC_VER,
        &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->registerExtension(&ext2, &ext2_desc));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->unregisterExtension(ext2));
}

TEST_F(PluginFrameworkExtensionsVersionTest, test_when_registering_ext_with_newer_api_ver_register_returns_unsupported)
{
    nvimgcodecExtension_t ext1;
    nvimgcodecExtensionDesc_t ext1_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr, "test_ext", 1000,
        NVIMGCODEC_VER + 1, &PluginFrameworkExtensionsVersionTest::ExtCreate, &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED, framework_->registerExtension(&ext1, &ext1_desc));
}

TEST(PluginFrameworkTest, register_file_extension_via_plugin)
{
    MockLogger logger;
    CodecRegistry registry(&logger);
    auto env = std::make_unique<MockEnvironment>();
    EXPECT_CALL(*env, getVariable("NVIMGCODEC_EXTENSIONS_PATH")).WillRepeatedly(Return(""));
    auto directory_scaner = std::make_unique<MockDirectoryScaner>();
    EXPECT_CALL(*directory_scaner, exists(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(*directory_scaner, hasMore()).WillRepeatedly(Return(false));
    auto library_loader = std::make_unique<MockLibraryLoader>();
    PluginFramework framework(&logger, &registry, std::move(env), std::move(directory_scaner), std::move(library_loader), "");

    auto create = [](void*, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* fw) -> nvimgcodecStatus_t {
        fw->registerFileExtension(fw->instance, "custom", ".cus", 1);
        fw->registerFileExtension(fw->instance, "custom", "CUS2", 1);
        g_too_long_ext.assign(1, '.');
        g_too_long_ext.append(static_cast<size_t>(NVIMGCODEC_MAX_FILE_EXTENSION_SIZE - 1), 'x');
        g_too_long_status = fw->registerFileExtension(fw->instance, "custom", g_too_long_ext.c_str(), 1);
        g_max_ok_ext.assign(1, '.');
        g_max_ok_ext.append(static_cast<size_t>(NVIMGCODEC_MAX_FILE_EXTENSION_SIZE - 2), 'y');
        g_max_ok_status = fw->registerFileExtension(fw->instance, "custom", g_max_ok_ext.c_str(), 1);
        g_too_long_codec.assign(static_cast<size_t>(NVIMGCODEC_MAX_CODEC_NAME_SIZE), 'c');
        g_too_long_codec_status = fw->registerFileExtension(fw->instance, g_too_long_codec.c_str(), ".longc", 1);
        g_max_ok_codec.assign(static_cast<size_t>(NVIMGCODEC_MAX_CODEC_NAME_SIZE - 1), 'c');
        g_max_ok_codec_status = fw->registerFileExtension(fw->instance, g_max_ok_codec.c_str(), ".okc", 1);
        *extension = reinterpret_cast<nvimgcodecExtension_t>(1);
        return NVIMGCODEC_STATUS_SUCCESS;
    };
    auto destroy = [](nvimgcodecExtension_t) -> nvimgcodecStatus_t { return NVIMGCODEC_STATUS_SUCCESS; };

    nvimgcodecExtension_t ext = nullptr;
    nvimgcodecExtensionDesc_t desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr,
        "file_ext_test", 1, NVIMGCODEC_VER, create, destroy};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework.registerExtension(&ext, &desc));
    EXPECT_EQ((std::vector<std::string>{"custom"}), registry.getCodecNames(".cus"));
    EXPECT_EQ((std::vector<std::string>{"custom"}), registry.getCodecNames(".cus2"));
    EXPECT_EQ((std::vector<std::string>{".cus", ".cus2", g_max_ok_ext}), registry.getFileExtensions("custom"));
    ASSERT_EQ(NVIMGCODEC_MAX_FILE_EXTENSION_SIZE, g_too_long_ext.size());
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, g_too_long_status);
    EXPECT_TRUE(registry.getCodecNames(g_too_long_ext).empty());
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, g_max_ok_status);
    EXPECT_EQ((std::vector<std::string>{"custom"}), registry.getCodecNames(g_max_ok_ext));
    ASSERT_EQ(NVIMGCODEC_MAX_CODEC_NAME_SIZE, g_too_long_codec.size());
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, g_too_long_codec_status);
    EXPECT_TRUE(registry.getCodecNames(".longc").empty());
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, g_max_ok_codec_status);
    EXPECT_EQ((std::vector<std::string>{g_max_ok_codec}), registry.getCodecNames(".okc"));

    const nvimgcodecFrameworkDesc_t* fw = framework.getFrameworkDesc();
    int codec_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, fw->getCodecNames(fw->instance, nullptr, &codec_count, nullptr));
    ASSERT_EQ(2, codec_count);
    std::vector<char> codec_names(static_cast<size_t>(codec_count) * NVIMGCODEC_MAX_CODEC_NAME_SIZE);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, fw->getCodecNames(fw->instance, codec_names.data(), &codec_count, nullptr));
    EXPECT_STREQ("custom", codec_names.data());
    EXPECT_STREQ(g_max_ok_codec.c_str(), codec_names.data() + NVIMGCODEC_MAX_CODEC_NAME_SIZE);

    int mapped_codec_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        fw->getCodecNames(fw->instance, nullptr, &mapped_codec_count, ".cus"));
    ASSERT_EQ(1, mapped_codec_count);
    char mapped_codec[NVIMGCODEC_MAX_CODEC_NAME_SIZE] = {};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        fw->getCodecNames(fw->instance, mapped_codec, &mapped_codec_count, ".cus"));
    EXPECT_STREQ("custom", mapped_codec);

    int file_extension_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        fw->getFileExtensions(fw->instance, nullptr, &file_extension_count, "custom"));
    ASSERT_EQ(3, file_extension_count);

    int all_file_extension_count = 0;
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, fw->getFileExtensions(fw->instance, nullptr, &all_file_extension_count, nullptr));
    ASSERT_EQ(4, all_file_extension_count);
}

TEST(PluginFrameworkTest, has_file_extension_api_uses_struct_size)
{
    nvimgcodecFrameworkDesc_t fw{};
    fw.struct_size = offsetof(nvimgcodecFrameworkDesc_t, registerFileExtension);
    EXPECT_FALSE(hasFileExtensionApi(&fw));
    fw.struct_size = offsetof(nvimgcodecFrameworkDesc_t, unregisterFileExtension);
    EXPECT_FALSE(hasFileExtensionApi(&fw));
    fw.struct_size = offsetof(nvimgcodecFrameworkDesc_t, unregisterFileExtension) + sizeof(fw.unregisterFileExtension);
    EXPECT_TRUE(hasFileExtensionApi(&fw));
    EXPECT_FALSE(hasFileExtensionApi(nullptr));
}

TEST(PluginFrameworkTest, register_codec_file_extensions_helper_does_not_steal_default)
{
    MockLogger logger;
    CodecRegistry registry(&logger);
    auto env = std::make_unique<MockEnvironment>();
    EXPECT_CALL(*env, getVariable("NVIMGCODEC_EXTENSIONS_PATH")).WillRepeatedly(Return(""));
    auto directory_scaner = std::make_unique<MockDirectoryScaner>();
    EXPECT_CALL(*directory_scaner, exists(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(*directory_scaner, hasMore()).WillRepeatedly(Return(false));
    auto library_loader = std::make_unique<MockLibraryLoader>();
    PluginFramework framework(&logger, &registry, std::move(env), std::move(directory_scaner), std::move(library_loader), "");

    auto create = [](void*, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* fw) -> nvimgcodecStatus_t {
        registerCodecFileExtensions(fw, "first", {".x"});
        registerCodecFileExtensions(fw, "second", {".x"});
        *extension = reinterpret_cast<nvimgcodecExtension_t>(1);
        return NVIMGCODEC_STATUS_SUCCESS;
    };
    auto destroy = [](nvimgcodecExtension_t) -> nvimgcodecStatus_t { return NVIMGCODEC_STATUS_SUCCESS; };

    nvimgcodecExtension_t ext = nullptr;
    nvimgcodecExtensionDesc_t desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr,
        "default_ext_test", 1, NVIMGCODEC_VER, create, destroy};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework.registerExtension(&ext, &desc));
    EXPECT_EQ((std::vector<std::string>{"first", "second"}), registry.getCodecNames(".x"));
}

// 0.9 packs to 900 both under the encoding it was released with and under the
// current one, so an extension binary built against 0.9 headers keeps loading.
TEST_F(PluginFrameworkExtensionsVersionTest, test_when_registering_ext_built_against_0_9_api_register_returns_success)
{
    ASSERT_EQ(900u, NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 0));
    ASSERT_GE(NVIMGCODEC_VER, 900u);

    nvimgcodecExtension_t ext1;
    nvimgcodecExtensionDesc_t ext1_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, nullptr, "test_ext", 1000,
        NVIMGCODEC_MAKE_SEMANTIC_VERSION(0, 9, 0), &PluginFrameworkExtensionsVersionTest::ExtCreate,
        &PluginFrameworkExtensionsVersionTest::ExtDestroy};

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->registerExtension(&ext1, &ext1_desc));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework_->unregisterExtension(ext1));
}

}} // namespace nvimgcodec::test
