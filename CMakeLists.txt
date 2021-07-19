cmake_minimum_required (VERSION 3.8)

project ("raytracing")

add_executable (raytracing "Source/main.cpp" "Source/main.hpp" "Source/surface.cpp" "Source/surface.hpp" "Source/color.hpp" "Source/ray.hpp")

target_include_directories(raytracing PUBLIC
        "External/glm")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(arch_ptr_size "x64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(arch_ptr_size "x32")
else()
    message(FATAL_ERROR "Can't determine ptr size, please check CMakeLists.txt")
endif()

if(WIN32)
    target_link_libraries(raytracing
            "${CMAKE_SOURCE_DIR}/External/SDL2/lib/${arch_ptr_size}/SDL2.lib"
            "${CMAKE_SOURCE_DIR}/External/SDL2/lib/${arch_ptr_size}/SDL2main.lib")

    target_include_directories(raytracing PUBLIC
            "external/SDL2/include")

    if(NOT(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/SDL2.dll"))
        file(COPY "${CMAKE_SOURCE_DIR}/External/SDL2/lib/${arch_ptr_size}/SDL2.dll" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
endif()
