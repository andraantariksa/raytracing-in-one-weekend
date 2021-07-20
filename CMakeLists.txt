cmake_minimum_required (VERSION 3.8)

project("raytracing" LANGUAGES CXX CUDA)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
else()
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
endif()

add_executable(raytracing
        "Source/main.cpp"
        "Source/Surface.cpp"
        "Source/Surface.hpp"
        "Source/typedef.hpp"
        "Source/Ray.hpp"
        Source/Hit/IHittableObject.hpp
        Source/Hit/HitData.hpp
        Source/Hit/Object/SphereObject.cpp
        Source/Hit/Object/SphereObject.hpp
        Source/Hit/HittableObjects.cpp
        Source/Hit/HittableObjects.hpp
        Source/Misc/Random.hpp
        Source/Camera.cpp
        Source/Camera.hpp
        "Source/Render.cu"
        "Source/Render.cuh")

target_compile_features(raytracing PUBLIC cxx_std_17)
set_target_properties(raytracing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

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
