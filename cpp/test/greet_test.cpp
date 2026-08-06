#include "dev_repo_template/greet.hpp"

#include <gtest/gtest.h>

TEST(Greet, ReturnsExpectedMessage) {
    EXPECT_EQ(dev_repo_template::greet("Dev-Repo-Template"), "Hello, Dev-Repo-Template!");
}

TEST(Greet, HandlesDefaultCase) {
    EXPECT_EQ(dev_repo_template::greet("world"), "Hello, world!");
}
