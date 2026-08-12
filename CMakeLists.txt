cmake_minimum_required(VERSION 3.24)
project(IPDE VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Qt6 6.8 REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()

set(IPDE_PYTHON_EXECUTABLE
    "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
    CACHE FILEPATH "Python interpreter used by the IPDE GUI")

qt_add_executable(ipde_gui MACOSX_BUNDLE
    src/gui/main.cpp
)

set(IPDE_PYTHON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/__main__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/cli.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/extractor.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/formats.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/libheif_aux.py"
)

target_link_libraries(ipde_gui PRIVATE Qt6::Widgets)
target_compile_options(ipde_gui PRIVATE -Wall -Wextra -Wpedantic)

if(APPLE)
    # This workstation also has a newer third-party libc++ in /usr/local/lib.
    # Apple Clang injects that search path before the SDK and would otherwise
    # produce an @rpath/libc++.1.dylib dependency that cannot launch. Link the
    # platform libc++ stub explicitly so the app uses macOS's dyld shared cache.
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE IPDE_MACOS_SDK
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE IPDE_XCRUN_RESULT
    )
    if(IPDE_XCRUN_RESULT EQUAL 0 AND EXISTS "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
        target_link_options(ipde_gui PRIVATE -nostdlib++)
        target_link_libraries(ipde_gui PRIVATE "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
    endif()
endif()
target_compile_definitions(ipde_gui PRIVATE
    IPDE_SOURCE_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py"
    IPDE_PYTHON_EXECUTABLE="${IPDE_PYTHON_EXECUTABLE}"
)
set_property(TARGET ipde_gui APPEND PROPERTY LINK_DEPENDS "${IPDE_PYTHON_SOURCES}")

set_target_properties(ipde_gui PROPERTIES
    OUTPUT_NAME "IPDE"
    MACOSX_BUNDLE_BUNDLE_NAME "IPDE"
    MACOSX_BUNDLE_GUI_IDENTIFIER "org.openai.ipde"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
)

if(APPLE)
    set(resource_root "$<TARGET_BUNDLE_DIR:ipde_gui>/Contents/Resources")
    add_custom_command(TARGET ipde_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${resource_root}/src/ipde"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${resource_root}/src/ipde"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py" "${resource_root}/ipde_extract.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/__init__.py" "${resource_root}/src/ipde/__init__.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/__main__.py" "${resource_root}/src/ipde/__main__.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/cli.py" "${resource_root}/src/ipde/cli.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/extractor.py" "${resource_root}/src/ipde/extractor.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/formats.py" "${resource_root}/src/ipde/formats.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/libheif_aux.py" "${resource_root}/src/ipde/libheif_aux.py"
        COMMENT "Embedding the Python extraction engine in IPDE.app"
    )
endif()
