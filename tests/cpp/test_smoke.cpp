#include <gtest/gtest.h>

#include "vecsearch/version.hpp"

TEST(Smoke, VersionIsSet) { EXPECT_STRNE(vecsearch::version(), ""); }
