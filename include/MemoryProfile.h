#pragma once

#include <cstddef>

struct MemoryProfileSnapshot
{
    unsigned long long startLiveBytes = 0;
    unsigned long long peakLiveBytes = 0;
    unsigned long long endLiveBytes = 0;
    unsigned long long totalAllocatedBytes = 0;
    unsigned long long positionAllocatedBytes = 0;
    unsigned long long positionPeakLiveBytes = 0;
    unsigned long long positionPeakIncreaseBytes = 0;
    unsigned long long copyTableBytes = 0;
    unsigned long long finalOutputBytes = 0;
    unsigned long long branchConcurrentUpperBoundBytes = 0;
    bool available = false;
};

// The allocator-level profiler is compiled only for --memory-profile builds.
// When it is unavailable these functions are harmless and return zeros.
bool memoryProfileAvailable();
void memoryProfileBeginRun(bool enabled);
MemoryProfileSnapshot memoryProfileEndRun();

void memoryProfileBeginPositionPropagation();
void memoryProfileEndPositionPropagation();
void memoryProfileSetCopyTableBytes(unsigned long long bytes);
void memoryProfileSetFinalOutputBytes(unsigned long long bytes);

// The submitted 16-thread schedule evaluates sibling branches one at a time,
// with the full inner worker team assigned to each branch.  During a memory
// profile, these hooks record each branch's incremental peak and sum them to
// obtain a conservative live-memory bound for running all siblings together.
// They do not change the execution schedule.
void memoryProfileBeginBranchGroup(bool branchesAreSequential);
void memoryProfileBeginBranch();
void memoryProfileEndBranch();
void memoryProfileEndBranchGroup();

// Logical storage helpers. They report vector capacity rather than only the
// number of initialized cells, which is closer to the allocated payload.
template <typename TableLike>
unsigned long long memoryProfileTableBytes(const TableLike &table)
{
    unsigned long long bytes =
        static_cast<unsigned long long>(table.capacity()) * sizeof(typename TableLike::value_type);
    for (const auto &row : table)
        bytes += static_cast<unsigned long long>(row.capacity()) * sizeof(typename TableLike::value_type::value_type);
    return bytes;
}
