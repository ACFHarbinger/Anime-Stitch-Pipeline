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
target_link_libraries(animation_tests PRIVATE animation_impl Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(animation_tests)
