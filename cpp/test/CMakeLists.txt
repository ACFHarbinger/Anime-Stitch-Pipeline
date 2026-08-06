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
    animation/test_seam.cpp
    animation/test_compositing.cpp
    animation/test_bundle_adjust.cpp
    animation/test_matching.cpp
    animation/test_validation.cpp
    animation/test_fuzz.cpp
)
target_link_libraries(animation_tests PRIVATE animation_impl Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(animation_tests)
