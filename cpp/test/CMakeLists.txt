find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(dev_repo_template_tests greet_test.cpp)
target_link_libraries(dev_repo_template_tests PRIVATE dev_repo_template_cpp GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(dev_repo_template_tests)
