add_executable(jlens-capture
    jlens-capture.cpp
    jlens-model.cpp
    jlens-graph.cpp)
target_link_libraries(jlens-capture PRIVATE ggml)
target_include_directories(jlens-capture PRIVATE .)
install(TARGETS jlens-capture RUNTIME)

add_executable(jlens-fit
    jlens-fit.cpp
    jlens-model.cpp
    jlens-graph.cpp)
target_link_libraries(jlens-fit PRIVATE ggml)
target_include_directories(jlens-fit PRIVATE .)
install(TARGETS jlens-fit RUNTIME)

add_executable(jlens-bwtest
    jlens-bwtest.cpp
    jlens-model.cpp)
target_link_libraries(jlens-bwtest PRIVATE ggml)
target_include_directories(jlens-bwtest PRIVATE .)
install(TARGETS jlens-bwtest RUNTIME)

add_executable(jlens-prep
    jlens-prep.cpp)
target_link_libraries(jlens-prep PRIVATE llama)
target_include_directories(jlens-prep PRIVATE .)
install(TARGETS jlens-prep RUNTIME)

add_executable(jlens-readout
    jlens-readout.cpp
    jlens-model.cpp
    jlens-graph.cpp)
target_link_libraries(jlens-readout PRIVATE ggml)
target_include_directories(jlens-readout PRIVATE .)
install(TARGETS jlens-readout RUNTIME)

add_executable(jlens-fattn-test
    jlens-fattn-test.cpp)
target_link_libraries(jlens-fattn-test PRIVATE ggml)
target_include_directories(jlens-fattn-test PRIVATE .)
install(TARGETS jlens-fattn-test RUNTIME)
