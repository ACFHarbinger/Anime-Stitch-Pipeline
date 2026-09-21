include(FetchContent)

find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.7.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

add_executable(animation_tests
    test_seam.cpp
    test_compositing.cpp
    test_bundle_adjust.cpp
    test_matching.cpp
    test_validation.cpp
    test_fuzz.cpp
)
target_link_libraries(animation_tests PRIVATE
    animation_impl
    Catch2::Catch2WithMain
    Python3::Python
)

include(CTest)
# catch_discover_tests concatenates Catch2 case names with ';' into one
# CTest filter; names with commas then match nothing. Run the binary as
# a single CTest entry instead.
add_test(NAME animation_tests COMMAND animation_tests)
