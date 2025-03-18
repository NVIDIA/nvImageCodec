/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include "imgproc/convert.h"
#include "imgproc/math_util.h"
#include "convert_test_static.h"
#include "device_test.h"

namespace nvimgcodec {

__device__ int my_strcmp(const char * s1, const char * s2) {
    for(; *s1 == *s2; ++s1, ++s2) {
        if(*s1 == 0) {
            return 0;
        }
    }

    return *(unsigned char *)s1 < *(unsigned char *)s2 ? -1 : 1;
}
 
#define DEV_EXPECT_STREQ(a, b) { \
  if(my_strcmp(a, b) != 0) { \
    int nmsg = atomicAdd(&test_status.num_messages, 1); \
    if (nmsg < MAX_DEVICE_ERROR_MESSAGES && !test_status.fatal) \
      printf("Check failed at %s:%i:\n%s (= %s) == %s (= %s)\n", \
          __FILE__, __LINE__, \
          #a, a, #b, b); \
    test_status.failed = true; \
  } \
}

DEVICE_TEST(String_Dev, constructor, 1, 1) {
  DeviceString str;
  DEV_EXPECT_STREQ(str.c_str(), "");

  DeviceString str1("hello world!");
  DEV_EXPECT_STREQ(str1.c_str(), "hello world!");

  DeviceString str2(str1);
  DEV_EXPECT_STREQ(str2.c_str(), "hello world!");

  DeviceString& str3(str2);
  DEV_EXPECT_STREQ(str3.c_str(), "hello world!");

  DeviceString str4(str3);
  DEV_EXPECT_STREQ(str4.c_str(), "hello world!");
  
  DeviceString str5("a");
  str5 = str4;
  DEV_EXPECT_STREQ(str5.c_str(), "hello world!");

  DeviceString str6("b");
  str6 = str3;
  DEV_EXPECT_STREQ(str6.c_str(), "hello world!");
}

DEVICE_TEST(String_Dev, operation, 1, 1) {
  DeviceString str("hello world!");

  const char* s1 = str.c_str();
  const char* s2 = str.data();
  DEV_EXPECT_STREQ(s1, "hello world!");
  DEV_EXPECT_STREQ(s2, "hello world!");
  DEV_EXPECT_STREQ(s1, s2);

  size_t size = str.size();
  size_t len = str.length();
  DEV_EXPECT_EQ(size, 12);
  DEV_EXPECT_EQ(len, 12);
  DEV_EXPECT_EQ(size, len);

  char s3[7] = "abcdef";
  str.reset(s3, 6);
  DEV_EXPECT_STREQ(str.c_str(), "abcdef");

  str.clear();
  DEV_EXPECT_EQ(str.size(), 0);

  DeviceString str1("ab");
  DeviceString str2("cde");
  DeviceString str3 = str1 + str2;
  DEV_EXPECT_STREQ(str3.c_str(), "abcde");

  str1 += str3;
  DEV_EXPECT_STREQ(str1.c_str(), "ababcde");

  char c = str1[2];
  DEV_EXPECT_EQ(c, 'a');

  const DeviceString str4("hello world!");
  char c1 = str4[5];
  DEV_EXPECT_EQ(c1, ' ');
}

DEVICE_TEST(String_Dev, dev_to_string, 1, 1) {
  char str[6] = "hello";
  const char* str1 = dev_to_string(str);
  DEV_EXPECT_STREQ(str1, "hello");

  const char* str2 = "world";
  const char* str3 = dev_to_string(str2);
  DEV_EXPECT_STREQ(str3, "world");

  bool b = true;
  DeviceString str4 = dev_to_string(b);
  DEV_EXPECT_STREQ(str4.c_str(), "true");

  b = false;
  str4 = dev_to_string(b);
  DEV_EXPECT_STREQ(str4.c_str(), "false");

  long long l_val = 0;
  str4 = dev_to_string(l_val);
  DEV_EXPECT_STREQ(str4.c_str(), "0");

  l_val = 1234567890;
  str4 = dev_to_string(l_val);
  DEV_EXPECT_STREQ(str4.c_str(), "1234567890");

  l_val = -1234567890;
  str4 = dev_to_string(l_val);
  DEV_EXPECT_STREQ(str4.c_str(), "-1234567890");

  int i_val = 12345;
  str4 = dev_to_string(i_val);
  DEV_EXPECT_STREQ(str4.c_str(), "12345");

  i_val = -12345;
  str4 = dev_to_string(i_val);
  DEV_EXPECT_STREQ(str4.c_str(), "-12345");

  float f_val = 0.0;
  str4 = dev_to_string(f_val);
  DEV_EXPECT_STREQ(str4.c_str(), "0");

  f_val = 769.5;
  str4 = dev_to_string(f_val);
  DEV_EXPECT_STREQ(str4.c_str(), "769.5");

  f_val = -98794.5;
  str4 = dev_to_string(f_val);
  DEV_EXPECT_STREQ(str4.c_str(), "-98794.5");

  f_val = 1.5e10;
  str4 = dev_to_string(f_val);
  DEV_EXPECT_STREQ(str4.c_str(), "1.5e+10");

  f_val = 1.5e-10;
  str4 = dev_to_string(f_val);
  DEV_EXPECT_STREQ(str4.c_str(), "1.5e-10");

  const void* p = nullptr;
  str4 = dev_to_string(p);
  DEV_EXPECT_STREQ(str4.c_str(), "0x0");

  str4 = dev_to_string((const void *)0);
  DEV_EXPECT_STREQ(str4.c_str(), "0x0");

  str4 = dev_to_string((const void *)1);
  DEV_EXPECT_STREQ(str4.c_str(), "0x1");

  str4 = dev_to_string((const void *)2);
  DEV_EXPECT_STREQ(str4.c_str(), "0x2");

  str4 = dev_to_string((const void *)3);
  DEV_EXPECT_STREQ(str4.c_str(), "0x3");

  str4 = dev_to_string((const void *)4);
  DEV_EXPECT_STREQ(str4.c_str(), "0x4");

  str4 = dev_to_string((const void *)5);
  DEV_EXPECT_STREQ(str4.c_str(), "0x5");

  str4 = dev_to_string((const void *)6);
  DEV_EXPECT_STREQ(str4.c_str(), "0x6");

  str4 = dev_to_string((const void *)7);
  DEV_EXPECT_STREQ(str4.c_str(), "0x7");

  str4 = dev_to_string((const void *)8);
  DEV_EXPECT_STREQ(str4.c_str(), "0x8");

  str4 = dev_to_string((const void *)9);
  DEV_EXPECT_STREQ(str4.c_str(), "0x9");

  str4 = dev_to_string((const void *)10);
  DEV_EXPECT_STREQ(str4.c_str(), "0xA");

  str4 = dev_to_string((const void *)11);
  DEV_EXPECT_STREQ(str4.c_str(), "0xB");

  str4 = dev_to_string((const void *)12);
  DEV_EXPECT_STREQ(str4.c_str(), "0xC");

  str4 = dev_to_string((const void *)13);
  DEV_EXPECT_STREQ(str4.c_str(), "0xD");

  str4 = dev_to_string((const void *)14);
  DEV_EXPECT_STREQ(str4.c_str(), "0xE");

  str4 = dev_to_string((const void *)15);
  DEV_EXPECT_STREQ(str4.c_str(), "0xF");

  DeviceString str5 = dev_to_string((const void *)255);
  DEV_EXPECT_STREQ(str5.c_str(), "0xFF");

  str5 = dev_to_string((const void *)256);
  DEV_EXPECT_STREQ(str5.c_str(), "0x100");

  str5 = dev_to_string((const void *)74283);
  DEV_EXPECT_STREQ(str5.c_str(), "0x1222B");

  str5 = dev_to_string((const void *)30589582);
  DEV_EXPECT_STREQ(str5.c_str(), "0x1D2C28E");
}

}  // namespace nvimgcodec
