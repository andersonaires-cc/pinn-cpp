cmake_minimum_required(VERSION 3.18)

project(pinn-cpp
    VERSION 1.0.0
    LANGUAGES CXX
)

# ============================================================
# C++20
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ============================================================
# LibTorch
# ============================================================

list(APPEND CMAKE_PREFIX_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch"
)

find_package(Torch REQUIRED)


# ============================================================
# Flags do LibTorch
# ============================================================

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}"
)


# ============================================================
# Código fonte
# ============================================================

file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)


# ============================================================
# Executável
# ============================================================

add_executable(
    ${PROJECT_NAME}
    ${SOURCES}
)


# ============================================================
# Headers
# ============================================================

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)


# ============================================================
# LibTorch
# ============================================================

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    "${TORCH_LIBRARIES}"
)


# ============================================================
# Windows
# ============================================================

if(MSVC)

    target_compile_options(
        ${PROJECT_NAME}
        PRIVATE
        /W4
    )

endif()


# ============================================================
# DLLs do LibTorch
# ============================================================

if(MSVC)

    file(GLOB TORCH_DLLS
        "${TORCH_INSTALL_PREFIX}/lib/*.dll"
    )

    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD

        COMMAND ${CMAKE_COMMAND}
        -E copy_if_different

        ${TORCH_DLLS}

        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )

endif()