cmake_minimum_required (VERSION 3.8)

project("raytracing" LANGUAGES CUDA CXX)


add_executable(raytracing
        "Source/main.cu"
        "Source/Surface.cu"
        "Source/Surface.cuh"
        "Source/Typedef.cuh"
        "Source/Ray.cuh"
        Source/Hit/IHittableObject.cuh
        Source/Hit/HitData.cuh
        Source/Hit/Object/SphereObject.cu
        Source/Hit/Object/SphereObject.cuh
        Source/Util/Random.cuh
        Source/Camera.cu
        Source/Camera.cuh
        "Source/Render.cu"
        "Source/Render.cuh"
        Source/Hit/HittableObjectsDevice.cu
        Source/Hit/HittableObjectsDevice.cuh
        Source/Material/IMaterial.cuh
        Source/Material/Metal.cu
        Source/Material/Metal.cuh
        Source/Hit/Object/BoxObject.cu
        Source/Hit/Object/BoxObject.cuh
        Source/Util/Timer.cu
        Source/Util/Timer.cuh
        "Source/Util/SDLHelpers.cuh" Source/Material/Lambertian.cu Source/Material/Lambertian.cuh)

target_compile_definitions(raytracing PRIVATE
        GLM_SWIZZLE
        SDL_MAIN_HANDLED)

target_compile_options(raytracing PRIVATE -expt-relaxed-constexpr)

target_compile_features(raytracing PRIVATE cuda_std_17)

set_target_properties(raytracing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(raytracing PUBLIC
        "External/SDL2/include"
        "External/SDL2/include/SDL2"
        "External/glm")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(arch_ptr_size "x64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(arch_ptr_size "x32")
else()
    message(FATAL_ERROR "Can't determine ptr size, please check CMakeLists.txt")
endif()

set(LINKED_LIBRARIES
        d3d11.lib
        d3dcompiler.lib)

if(WIN32)
    list(APPEND LINKED_LIBRARIES
            "${CMAKE_SOURCE_DIR}/External/SDL2/lib/${arch_ptr_size}/SDL2.lib"
            "${CMAKE_SOURCE_DIR}/External/SDL2/lib/${arch_ptr_size}/SDL2main.lib")

    target_include_directories(raytracing PUBLIC
            "External/SDL2/include"
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    if(NOT(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/SDL2.dll"))
        file(COPY "${CMAKE_SOURCE_DIR}/External/SDL2/lib/${arch_ptr_size}/SDL2.dll" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
endif()

target_link_libraries(raytracing PUBLIC
        ${LINKED_LIBRARIES})
