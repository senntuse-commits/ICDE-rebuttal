#pragma once

#include <vector>

using NonObliviousTable = std::vector<std::vector<int>>;

enum NonObliviousJFYanStatus
{
    NonObliviousJFYanOk = 0,
    NonObliviousJFYanInvalidInput,
    NonObliviousJFYanCountOverflow,
    NonObliviousJFYanOutputTooLarge
};

struct NonObliviousJFYanResult
{
    NonObliviousTable output;
    long long exactRows = 0;
    double setupMs = 0.0;
    double bottomUpCriticalMs = 0.0;
    double materializeCriticalMs = 0.0;
    double parallelEstimateMs = 0.0;
    NonObliviousJFYanStatus status = NonObliviousJFYanOk;
};

// Non-oblivious counterpart of JFYan.
//
// It keeps JFYan's two main ideas:
//   1. compute each tuple's subtree contribution bottom-up;
//   2. allocate the final output once and propagate global positions top-down.
//
// Unlike JFYan, this baseline uses hash indexes, data-dependent branches, and
// direct writes to output[pos]. It therefore must not be used when access
// patterns need to be hidden.
NonObliviousJFYanResult NonObliviousJFYan(
    const std::vector<NonObliviousTable> &tables,
    const std::vector<int> &parent,
    int root,
    const std::vector<int> &joinColInParent,
    const std::vector<int> &joinColInChild,
    const std::vector<int> &tableKeys,
    int maxOutputRows = 2147483647);
