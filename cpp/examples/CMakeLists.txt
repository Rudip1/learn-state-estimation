set(SE_EXAMPLES
    01_probability
    02_dead_reckoning
)

foreach(example ${SE_EXAMPLES})
    add_executable(example_${example} ${example}.cpp)
    target_link_libraries(example_${example} PRIVATE state_estimation)
endforeach()
